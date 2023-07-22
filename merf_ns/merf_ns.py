from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type,Union,Optional,Generator

import numpy as np
import torch
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction

from torch.nn import Parameter
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Type

import numpy as np
import torch
from torch.nn import Parameter
from typing_extensions import Literal
import contextlib
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
# from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.model_components.losses import (
    MSELoss,
    distortion_loss,
    interlevel_loss,
    orientation_loss,
    pred_normal_loss,
    
)
import nerfacc
from nerfstudio.utils import colormaps
from nerfstudio.model_components.renderers import SHRenderer, RGBRenderer
from merf_ns.merf_ns_field import TCNNMeRFNSField
from jaxtyping import Float
from torch import Tensor, nn
import math
from nerfstudio.utils.math import components_from_spherical_harmonics, safe_normalize
try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    pass
from jaxtyping import Float, Int

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs/weights')

from nerfstudio.utils import colors
BackgroundColor = Union[Literal["random", "last_sample", "black", "white"], Float[Tensor, "3"]]
BACKGROUND_COLOR_OVERRIDE: Optional[Float[Tensor, "3"]] = None


@contextlib.contextmanager
def background_color_override_context(mode: Float[Tensor, "3"]) -> Generator[None, None, None]:
    """Context manager for setting background mode."""
    global BACKGROUND_COLOR_OVERRIDE
    old_background_color = BACKGROUND_COLOR_OVERRIDE
    try:
        BACKGROUND_COLOR_OVERRIDE = mode
        yield
    finally:
        BACKGROUND_COLOR_OVERRIDE = old_background_color


@dataclass
class MeRFNSConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: MeRFNSModel)
from .utils import MeRFNSFieldHeadNames

class MeRFNSRenderer(SHRenderer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.renderer_rgb = RGBRenderer()
        self.view_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": 4,
            },
        )

        self.view_mlp=tcnn.Network(
           3 + 4 + self.view_encoder.n_output_dims ,
             3,
            {
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons":  16,
                "n_hidden_layers": 3
            }
        )
    # TODO render函数需要修改
    @classmethod
    def combine_rgb(
        cls,
        rgb: Float[Tensor, "*bs num_samples 3"],
        weights: Float[Tensor, "*bs num_samples 1"],
        background_color: BackgroundColor = "random",
        ray_indices: Optional[Int[Tensor, "num_samples"]] = None,
        num_rays: Optional[int] = None,
    ) -> Float[Tensor, "*bs 3"]:
        """Composite samples along ray and render color image

        Args:
            rgb: RGB for each sample
            weights: Weights for each sample
            background_color: Background color as RGB.
            ray_indices: Ray index for each sample, used when samples are packed.
            num_rays: Number of rays, used when samples are packed.

        Returns:
            Outputs rgb values.
        """
        
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            if background_color == "last_sample":
                raise NotImplementedError("Background color 'last_sample' not implemented for packed samples.")
            comp_rgb = nerfacc.accumulate_along_rays(
                weights[..., 0], values=rgb, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulated_weight = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            comp_rgb = torch.sum(weights * rgb, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)

        if BACKGROUND_COLOR_OVERRIDE is not None:
            background_color = BACKGROUND_COLOR_OVERRIDE
        if background_color == "last_sample":
            background_color = rgb[..., -1, :]
        if background_color == "random":
            background_color = torch.rand_like(comp_rgb).to(rgb.device)
        if isinstance(background_color, str) and background_color in colors.COLORS_DICT:
            background_color = colors.COLORS_DICT[background_color].to(rgb.device)

        assert isinstance(background_color, torch.Tensor)
        comp_rgb = comp_rgb + background_color.to(weights.device) * (1.0 - accumulated_weight)

        return comp_rgb
    def forward(self,
                sh: Float[Tensor, "*batch num_samples coeffs"],
                diffuse: Float[Tensor, "*batch num_samples 3"], 
                directions: Float[Tensor, "*batch num_samples 3"],
                weights: Float[Tensor, "*batch num_samples 1"],
                bg_color: Optional[Float[Tensor, "3"]] = None,
                ) -> Float[Tensor, "*batch 3"]:
        
        # TODO [23/7/7] 这里需要对specular进行一次MLP和激活，不然specular和视角无关，到这一步的specular只是和dir拼在一起
        # sh = sh.view(*sh.shape[:-1], 3, sh.shape[-1] // 3)
        # weights_sum = torch.sum(weights, dim=-1)
        # # depth = torch.sum(weights * rays_t, dim=-1)
        # # # rgb=torch.sigmoid(self.view_mlp(sh))
        # # # # levels = int(math.sqrt(sh.shape[-1]))
        # # # # components = components_from_spherical_harmonics(levels=levels, directions=directions)
        
        # # # # rgb = sh * components[..., None, :]
        # # # # rgb = torch.sum(rgb, dim=-1)  # [..., num_samples, 3]
        # # # # if self.activation is not None:
        # # # #     rgb = self.activation(rgb)
        # # # # if not self.training:
        # # # #     rgb = torch.nan_to_num(rgb)
        # # # # rgb=rgb+diffuse
        # # # rgb=(rgb+diffuse).view(*weights.shape[:-1], 3)
        if bg_color is None:
            bg_color = 1
        weights_sum = torch.sum(weights.squeeze(-1), dim=-1) # [N]
        sh=sh.view(*weights.shape[:-1],-1)
        diffuse=diffuse.view(*weights.shape[:-1],-1)
        diffuse = torch.sum(weights* diffuse, dim=-2) # [N, 3]
        specular = torch.sum(weights * sh, dim=-2)
        specular=specular.view([-1,3 + 4 + self.view_encoder.n_output_dims])
        specular = torch.sigmoid(self.view_mlp(specular))
        rgb = (diffuse + specular).clamp(0, 1)
        rgb = rgb + (1 - weights_sum).unsqueeze(-1) * bg_color
        # rgb = MeRFNSRenderer.combine_rgb(rgb, weights, background_color=self.background_color)
        # if not self.training:
        #     torch.clamp_(rgb, min=0.0, max=1.0)
        return rgb
class MeRFNSModel(NerfactoModel):
    config: MeRFNSConfig

    ###临时增加记录step###
    step = 1
    

    def populate_modules(self):
        super().populate_modules()
        if self.config.disable_scene_contraction:
            scene_contraction = None
        else:
            scene_contraction = SceneContraction(order=float("inf"))
        self.field = TCNNMeRFNSField(
            self.scene_box.aabb, spatial_distortion=scene_contraction, num_images=self.num_train_data
        )
        self.renderer_rgb = MeRFNSRenderer()
        pass

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        param_groups["proposal_networks"] = list(
            self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []
        if self.config.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.config.proposal_weights_anneal_max_num_iters

            def set_anneal(step):
                # https://arxiv.org/pdf/2111.12077.pdf eq. 18
                train_frac = np.clip(step / N, 0, 1)
                def bias(x, b): return (b * x) / ((b - 1) * x + 1)
                anneal = bias(
                    train_frac, self.config.proposal_weights_anneal_slope)
                self.proposal_sampler.set_anneal(anneal)

            callbacks.append(
                TrainingCallback(
                    where_to_run=[
                        TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_anneal,
                )
            )
            callbacks.append(
                TrainingCallback(
                    where_to_run=[
                        TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self.proposal_sampler.step_cb,
                )
            )
        return callbacks
    # TODO

    def forward(self, ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        """Run forward starting with a ray bundle. This outputs different things depending on the configuration
        of the model and whether or not the batch is provided (whether or not we are training basically)

        Args:
            ray_bundle: containing all the information needed to render that ray latents included
        """

        if self.collider is not None:
            # 在这里完成了远近平面的设置
            ray_bundle = self.collider(ray_bundle)

        return self.get_outputs(ray_bundle)
    # TODO 这里调用了filed里的getoutput，原本是调用基类的get_output，然后这里我们需要重载一些filed的get_output,然后在这个函数里面
    # 计算merf相关的量？

    def get_outputs(self, ray_bundle: RayBundle):
        # print(self.step)
        self.step += 1  # 临时输出用

        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(
            ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field(
            ray_samples, compute_normals=self.config.predict_normals)
        weights = ray_samples.get_weights(
            field_outputs[MeRFNSFieldHeadNames.DENSITY])
        
        # 输出第64序raysample的density到tensorboard
        nowSample = 63 # 当前使用的射线序号
        density = field_outputs[MeRFNSFieldHeadNames.DENSITY]
        # print(density.shape)
        nowDensity = torch.squeeze(density[nowSample])
        # for index in range(len(nowDensity)):
        #     nowtag = "density"+str(index)
        #     writer.add_scalar(tag=nowtag, # 标签，即数据属于哪一类，不同的标签对应不同的图
        #                 scalar_value=nowDensity[index],  # 纵坐标的值
        #                 global_step=self.step  # 迭代次数，即图的横坐标
        #                 )
        # 输出第64序raysample的weights到tensorboard
        # print(ray_bundle.shape)->128
        # print(weights)->128,48,1
        # print(weights[64].shape)
        # nowWeights = torch.squeeze(weights[nowSample])
        # for index in range(len(nowWeights)):
        #     nowtag = "weight"+str(index)
        #     writer.add_scalar(tag=nowtag, # 标签，即数据属于哪一类，不同的标签对应不同的图
        #                 scalar_value=nowWeights[index],  # 纵坐标的值
        #                 global_step=self.step  # 迭代次数，即图的横坐标
        #                 )

        weights_list.append(weights)
        ray_samples_list.append(ray_samples)
        # 原始的ＲＢＧ计算方式不适用于我们的代码
        # rgb = self.renderer_rgb(
        #     rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        sh=field_outputs[MeRFNSFieldHeadNames.SH]
        diffuse=field_outputs[MeRFNSFieldHeadNames.DIFFUSE]
        
        # weights_sum = torch.sum(weights, dim=-1)
        # depth = torch.sum(weights * rays_t, dim=-1) # [N]
        
        rgb=self.renderer_rgb(sh,diffuse,ray_bundle.directions,weights)
        depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "sh":sh
        }

        # if self.config.predict_normals:
        #     normals = self.renderer_normals(
        #         normals=field_outputs[MeRFNSFieldHeadNames.NORMALS], weights=weights)
        #     pred_normals = self.renderer_normals(
        #         field_outputs[MeRFNSFieldHeadNames.PRED_NORMALS], weights=weights)
        #     outputs["normals"] = self.normals_shader(normals)
        #     outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        # if self.training and self.config.predict_normals:
        #     outputs["rendered_orientation_loss"] = orientation_loss(
        #         weights.detach(
        #         ), field_outputs[MeRFNSFieldHeadNames.NORMALS], ray_bundle.directions
        #     )

        #     outputs["rendered_pred_normal_loss"] = pred_normal_loss(
        #         weights.detach(),
        #         field_outputs[MeRFNSFieldHeadNames.NORMALS].detach(),
        #         field_outputs[MeRFNSFieldHeadNames.PRED_NORMALS],
        #     )
        # 这个地方把propo的depth保存出来了？
        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = {}
        image = batch["image"].to(self.device)
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        if self.training:
            metrics_dict["distortion"] = distortion_loss(
                outputs["weights_list"], outputs["ray_samples_list"])
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        # TODO Need help 
        loss_dict = {}
        image = batch["image"].to(self.device)
        loss_dict["rgb_loss"] = self.rgb_loss(image, outputs["rgb"])
        if self.training:
            loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss(
                outputs["weights_list"], outputs["ray_samples_list"]
            )
            assert metrics_dict is not None and "distortion" in metrics_dict
            loss_dict["distortion_loss"] = self.config.distortion_loss_mult * \
                metrics_dict["distortion"]
            # if self.config.predict_normals:
            #     # orientation loss for computed normals
            #     loss_dict["orientation_loss"] = self.config.orientation_loss_mult * torch.mean(
            #         outputs["rendered_orientation_loss"]
            #     )

            #     # ground truth supervision for normals
            #     loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
            #         outputs["rendered_pred_normal_loss"]
            #     )
            loss_dict['specular_loss'] =1e-5*(outputs['sh']**2).mean()
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(
            psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb,
                       "accumulation": combined_acc, "depth": combined_depth}

        for i in range(self.config.num_proposal_iterations):
            key = f"prop_depth_{i}"
            prop_depth_i = colormaps.apply_depth_colormap(
                outputs[key],
                accumulation=outputs["accumulation"],
            )
            images_dict[key] = prop_depth_i

        return metrics_dict, images_dict
