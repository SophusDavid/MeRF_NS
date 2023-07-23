from typing import Dict

import tyro
# from nerfacc import ContractionType


from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManagerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig, RAdamOptimizerConfig
from nerfstudio.engine.trainer import TrainerConfig

from nerfstudio.plugins.types import MethodSpecification
from merf_ns.merf_ns import MeRFNSConfig

from merf_ns.merf_ns_pipeline import  MeRFNSPipelineConfig
max_num_iterations=30000
MeRFNS_method = MethodSpecification(config=TrainerConfig(
    method_name="MeRFNS",
    steps_per_eval_image=0,
    steps_per_eval_batch=500,
    steps_per_save=2000,
    mixed_precision=True,
    max_num_iterations=max_num_iterations,
    pipeline=MeRFNSPipelineConfig(
        datamanager=VanillaDataManagerConfig(
            dataparser=NerfstudioDataParserConfig(),
            train_num_rays_per_batch=32,
            eval_num_rays_per_batch=32,
            camera_optimizer=CameraOptimizerConfig(
                mode="SO3xR3", optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2)
            ),
        ),
        model=MeRFNSConfig(eval_num_rays_per_chunk=1 << 15,max_num_iterations=max_num_iterations),
    ),
    optimizers={
        "proposal_networks": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
        "fields": {
            "optimizer": AdamOptimizerConfig(lr=1e-2, eps=1e-15),
            "scheduler": None,
        },
    },
    viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
    vis="viewer",
),
  description="Custom description"
)



