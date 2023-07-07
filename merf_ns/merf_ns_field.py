# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Field for compound nerf model, adds scene contraction and image embeddings to instant ngp
"""


from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import trunc_exp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import Encoding, HashEncoding, SHEncoding
from nerfstudio.field_components.field_heads import (
    DensityFieldHead,
    FieldHead,
    FieldHeadNames,
    PredNormalsFieldHead,
    RGBFieldHead,
    SemanticFieldHead,
    TransientDensityFieldHead,
    TransientRGBFieldHead,
    UncertaintyFieldHead,
)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import (
    SceneContraction,
    SpatialDistortion,
)
from nerfstudio.fields.base_field import Field, shift_directions_for_tcnn
from utils import MeRFNSFieldHeadNames
try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass


class TCNNMeRFNSField(Field):
    """Compound Field that uses TCNN

    Args:
        aabb: parameters of scene aabb bounds
        num_images: number of images in the dataset
        num_layers: number of hidden layers
        hidden_dim: dimension of hidden layers
        geo_feat_dim: output geo feat dimensions
        num_levels: number of levels of the hashmap for the base mlp
        max_res: maximum resolution of the hashmap for the base mlp
        log2_hashmap_size: size of the hashmap for the base mlp
        num_layers_color: number of hidden layers for color network
        num_layers_transient: number of hidden layers for transient network
        hidden_dim_color: dimension of hidden layers for color network
        hidden_dim_transient: dimension of hidden layers for transient network
        appearance_embedding_dim: dimension of appearance embedding
        transient_embedding_dim: dimension of transient embedding
        use_transient_embedding: whether to use transient embedding
        use_semantics: whether to use semantic segmentation
        num_semantic_classes: number of semantic classes
        use_pred_normals: whether to use predicted normals
        use_average_appearance_embedding: whether to use average appearance embedding or zeros for inference
        spatial_distortion: spatial distortion to apply to the scene
    """

    def __init__(
        self,
        aabb: TensorType,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_levels: int = 16,
        max_res: int = 2048,
        log2_hashmap_size: int = 19,
        num_layers_color: int = 3,
        num_layers_transient: int = 2,
        hidden_dim_color: int = 64,
        hidden_dim_transient: int = 64,
        appearance_embedding_dim: int = 32,
        transient_embedding_dim: int = 16,
        use_transient_embedding: bool = False,
        use_semantics: bool = False,
        num_semantic_classes: int = 100,
        pass_semantic_gradients: bool = False,
        use_pred_normals: bool = False,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: SpatialDistortion = None,
    ) -> None:
        super().__init__()
        
        # ******MERF****** 
        self.grid = Grid(level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=512, output_dim=8, num_layers=2, hidden_dim=32)
        
        # triplane
        # if self.opt.use_triplane:
        self.planeXY = Plane(level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=2048, output_dim=8, num_layers=2, hidden_dim=32)
        self.planeYZ = Plane(level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=2048, output_dim=8, num_layers=2, hidden_dim=32)
        self.planeXZ = Plane(level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=2048, output_dim=8, num_layers=2, hidden_dim=32)
        self.view_encoder, self.view_in_dim = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": 4,
            },
        )

        self.view_mlp=tcnn.Network(
           3 + 4 + self.view_in_dim ,
             3,
            {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons":  16,
                "n_hidden_layers": 3
            }
        )
        # ******MERF****** 
        self.register_buffer("aabb", aabb)
        self.geo_feat_dim = geo_feat_dim

        self.register_buffer("max_res", torch.tensor(max_res))
        self.register_buffer("num_levels", torch.tensor(num_levels))
        self.register_buffer("log2_hashmap_size",
                             torch.tensor(log2_hashmap_size))

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(
            self.num_images, self.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_transient_embedding = use_transient_embedding
        self.use_semantics = use_semantics
        self.use_pred_normals = use_pred_normals
        self.pass_semantic_gradients = pass_semantic_gradients

    
    def get_density(self, ray_samples: RaySamples) -> Tuple[TensorType, TensorType]:
        """Computes and returns the densities."""
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(
                ray_samples.frustums.get_positions(), self.aabb)  
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        self._sample_locations = positions
        if not self._sample_locations.requires_grad:
            self._sample_locations.requires_grad = True
        positions_flat = positions.view(-1, 3)
        f_sigma, f_diffuse, f_specular=self.common_forward(positions_flat)
        # sigma = trunc_exp(f_sigma - 1)
        # h = self.mlp_base(positions_flat).view(*ray_samples.frustums.shape, -1)
        # density_before_activation, base_mlp_out = torch.split(
        #     h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = f_sigma

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(f_sigma.to(positions))
        density = density * selector[..., None]
        return density

    
    def quantize_feature(self, f, baking=False):
        f[..., 0] = self.quantize(f[..., 0], 14, baking)
        f[..., 1:] = self.quantize(f[..., 1:], 7, baking)
        return f
    
    def quantize(self, x, m=7, baking=False):
        # x: in real value, to be quantized in to [-m, m]
        x = torch.sigmoid(x)

        if baking: return torch.floor(255 * x + 0.5)
        
        x = x + (torch.floor(255 * x + 0.5) / 255 - x).detach()
        x = 2 * m * x - m
        return x
    def common_forward(self, x):
        
        f = 0
        if self.opt.use_grid:
            f_grid = self.quantize_feature(self.grid(x, self.bound))
            f = f + f_grid
        if self.opt.use_triplane:
            f_plane_01 = self.quantize_feature(self.planeXY(x[..., [0, 1]], self.bound))
            f_plane_12 = self.quantize_feature(self.planeYZ(x[..., [1, 2]], self.bound))
            f_plane_02 = self.quantize_feature(self.planeXZ(x[..., [0, 2]], self.bound))
            f = f + f_plane_01 + f_plane_12 + f_plane_02
        
        f_sigma = f[..., 0]
        f_diffuse = f[..., 1:4]
        f_specular = f[..., 4:]

        return f_sigma, f_diffuse, f_specular
    #　这个地方重载了Field里的forward函数
    # TODO 重载forward 函数
    def forward(self, ray_samples: RaySamples, compute_normals: bool = False) -> Dict[FieldHeadNames, Tensor]:
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        if compute_normals:
            with torch.enable_grad():
                density = self.get_density(ray_samples)
        else:
            density = self.get_density(ray_samples)

        field_outputs = self.get_outputs(ray_samples)
        field_outputs[FieldHeadNames.DENSITY] = density  # type: ignore

        # if compute_normals:
        #     with torch.enable_grad():
        #         normals = self.get_normals()
        #     field_outputs[FieldHeadNames.NORMALS] = normals  # type: ignore
        return field_outputs
    
    # TODO 这个地方import utilis里面的ＭｅＲＦＮＳＦｉｅｌｄＮａｍｅ然后输出一下ＳＨ的参数？
    def get_outputs(
        self, ray_samples: RaySamples
    ) -> Dict[FieldHeadNames, TensorType]:
        # 这部分代码是nerf-w做apperance embedding的
        # assert density_embedding is not None
        outputs = {}
        # if ray_samples.camera_indices is None:
        #     raise AttributeError("Camera indices are not provided.")
        # camera_indices = ray_samples.camera_indices.squeeze()
        if self.spatial_distortion is not None:
            positions = ray_samples.frustums.get_positions()
            positions = self.spatial_distortion(positions)
            positions = (positions + 2.0) / 4.0
        else:
            positions = SceneBox.get_normalized_positions(
                ray_samples.frustums.get_positions(), self.aabb)  
        # Make sure the tcnn gets inputs between 0 and 1.
        selector = ((positions > 0.0) & (positions < 1.0)).all(dim=-1)
        positions = positions * selector[..., None]
        f_sigma, f_diffuse, f_specular = self.common_forward(positions)
        sigma = trunc_exp(f_sigma - 1)
        directions = shift_directions_for_tcnn(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        diffuse = torch.sigmoid(f_diffuse)
        f_specular = torch.sigmoid(f_specular)
        d = self.view_encoder(directions_flat)
        # d = self.direction_encoding(directions_flat)
        specular = torch.cat([diffuse, f_specular, d], dim=-1)
        outputs.update({FieldHeadNames.SH: specular})
        outputs.update({FieldHeadNames.DIFFUSE: diffuse})
        outputs.update({FieldHeadNames.DENSITY: sigma})
        return outputs


class Grid(nn.Module):
    def __init__(self, level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=512, output_dim=8, num_layers=2, hidden_dim=64, interpolation='linear'):
        super().__init__()
        # align corners (index in [0, resolution], resolution + 1 values!)
        self.resolution = desired_resolution
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": 2
            },
        )
        # self.encoder, self.in_dim = get_encoder("hashgrid", input_dim=3, level_dim=level_dim, num_levels=num_levels, log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution + 1, interpolation=interpolation)
        # self.mlp = MLP(self.in_dim, output_dim,
        #                hidden_dim, num_layers, bias=False)
        self.mlp = tcnn.Network(
            self.encoder.n_output_dims,
            output_dim,
            {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers
            }
        )

    def forward(self, xyz, bound):
        # manually perform the interpolation after any nonlinear MLP...
        # this resembles align_corners = True

        xyz = (xyz + bound) / (2 * bound)  # [0, 1]
        coords = xyz * self.resolution  # [0, resolution]
        # float coord
        cx, cy, cz = coords[..., 0], coords[..., 1], coords[..., 2]
        # int coord
        cx0, cy0, cz0 = cx.floor().clamp(0, self.resolution - 1).long(), cy.floor().clamp(0,
                                                                                          self.resolution - 1).long(), cz.floor().clamp(0, self.resolution - 1).long()
        cx1, cy1, cz1 = cx0 + 1, cy0 + 1, cz0 + 1
        # interp weights
        u, v, w = (cx - cx0).unsqueeze(-1), (cy -
                                             cy0).unsqueeze(-1), (cz - cz0).unsqueeze(-1)  # [N, 1] in [0, 1]
        # interp positions
        f000 = self.mlp(self.encoder(torch.stack(
            [cx0, cy0, cz0], dim=-1).float() / self.resolution))
        f001 = self.mlp(self.encoder(torch.stack(
            [cx0, cy0, cz1], dim=-1).float() / self.resolution))
        f010 = self.mlp(self.encoder(torch.stack(
            [cx0, cy1, cz0], dim=-1).float() / self.resolution))
        f011 = self.mlp(self.encoder(torch.stack(
            [cx0, cy1, cz1], dim=-1).float() / self.resolution))
        f100 = self.mlp(self.encoder(torch.stack(
            [cx1, cy0, cz0], dim=-1).float() / self.resolution))
        f101 = self.mlp(self.encoder(torch.stack(
            [cx1, cy0, cz1], dim=-1).float() / self.resolution))
        f110 = self.mlp(self.encoder(torch.stack(
            [cx1, cy1, cz0], dim=-1).float() / self.resolution))
        f111 = self.mlp(self.encoder(torch.stack(
            [cx1, cy1, cz1], dim=-1).float() / self.resolution))
        # interp
        f = (1 - w) * (1 - v) * (1 - u) * f000 + \
            (1 - w) * (1 - v) * u * f100 + \
            (1 - w) * v * (1 - u) * f010 + \
            (1 - w) * v * u * f110 + \
            w * (1 - v) * (1 - u) * f001 + \
            w * (1 - v) * u * f101 + \
            w * v * (1 - u) * f011 + \
            w * v * u * f111
        return f

    def grad_total_variation(self, lambda_tv):
        self.encoder.grad_total_variation(lambda_tv)


class Plane(nn.Module):
    def __init__(self, level_dim=2, num_levels=16, log2_hashmap_size=19, desired_resolution=2048, output_dim=8, num_layers=2, hidden_dim=64, interpolation='linear'):
        super().__init__()
        # align corners (index in [0, resolution], resolution + 1 values!)
        self.resolution = desired_resolution
        self.encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": num_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": 16,
                "per_level_scale": 2
            },
        )
        # self.encoder, self.in_dim = get_encoder("hashgrid", input_dim=2, level_dim=level_dim, num_levels=num_levels,
        #                                         log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, interpolation=interpolation)
        # self.mlp = MLP(self.in_dim, output_dim,
        #                hidden_dim, num_layers, bias=False)
        self.mlp = tcnn.Network(
            self.encoder.n_output_dims,
            output_dim,
            {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": hidden_dim,
                "n_hidden_layers": num_layers
            }
        )

    def forward(self, xy, bound):
        # manually perform the interpolation after any nonlinear MLP...
        # this resembles align_corners = False

        xy = (xy + bound) / (2 * bound)  # [0, 1]
        coords = xy * self.resolution - 0.5  # [-0.5, resolution-0.5]
        coords = coords.clamp(0, self.resolution - 1)  # [0, resolution-1]
        # float coord
        cx, cy = coords[..., 0], coords[..., 1]
        # int coord
        cx0, cy0 = cx.floor().long(), cy.floor().long()
        cx1, cy1 = (cx0 + 1).clamp(0, self.resolution -
                                   1), (cy0 + 1).clamp(0, self.resolution - 1)
        # interp weights
        u, v = (cx - cx0).unsqueeze(-1), (cy -
                                          cy0).unsqueeze(-1)  # [N, 1] in [0, 1]
        # interp positions
        f00 = self.mlp(self.encoder(
            (torch.stack([cx0, cy0], dim=-1).float() + 0.5) / self.resolution))
        f01 = self.mlp(self.encoder(
            (torch.stack([cx0, cy1], dim=-1).float() + 0.5) / self.resolution))
        f10 = self.mlp(self.encoder(
            (torch.stack([cx1, cy0], dim=-1).float() + 0.5) / self.resolution))
        f11 = self.mlp(self.encoder(
            (torch.stack([cx1, cy1], dim=-1).float() + 0.5) / self.resolution))
        # interp
        f = (1 - v) * (1 - u) * f00 + \
            (1 - v) * u * f10 + \
            v * (1 - u) * f01 + \
            v * u * f11
        return f

    def grad_total_variation(self, lambda_tv):
        self.encoder.grad_total_variation(lambda_tv)


field_implementation_to_class: Dict[str, Field] = {"tcnn": TCNNMeRFNSField}
