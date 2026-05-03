"""
paligemma_vision_backbone.py

Abstract class definition of a Vision Backbone (Visual Featurizer), with full annotations of class methods, utility
functions, and initialization logic.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Dict, Optional, Protocol, Tuple, Union, Type

import torch
import torch.nn as nn
import torchvision.transforms.functional as TVF
from PIL.Image import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from torchvision.transforms import Compose, Resize

from transformers import AutoModel, AutoConfig, AutoProcessor
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.siglip.modeling_siglip import SiglipVisionModel, SiglipEncoderLayer

from prismatic.models.backbones.vision.base_vision import VisionBackbone

class PaliGemmaVisionBackbone(VisionBackbone, ABC):
    def __init__(
        self,
        vision_backbone_id: str,
        image_resize_strategy: str,
        default_image_size: int = 224,
        hf_hub_path: str = None,
        use_flash_attention_2: bool = True,
    ) -> None:
        super().__init__(vision_backbone_id, image_resize_strategy, default_image_size=default_image_size)

        # Initialize Featurizer (ViT)
        paligemma_conf = AutoConfig.from_pretrained(hf_hub_path)
        if use_flash_attention_2:
            paligemma_conf.vision_config.attn_implementation = "flash_attention_2"        
        self.featurizer = AutoModel.from_config(config=paligemma_conf.vision_config)
        # print(f"featurizer with use_flash_attention_2 : {self.featurizer.config.attn_implementation}")
        #
        if True:
        # if False:
            # paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(hf_hub_path)
            # self.featurizer.load_state_dict(paligemma_model.vision_tower.state_dict())
            # del paligemma_model
            self.featurizer.load_state_dict(torch.load(hf_hub_path + "_vision_tower.pth"))
        #
        self.featurizer.embed_dim = paligemma_conf.vision_config.projection_dim
        self.featurizer.eval()

        # Initialize Default Image Transform --> Modified by `self.image_resize_strategy`
        paligemma_processor = AutoProcessor.from_pretrained(hf_hub_path)
        self.image_seq_length = paligemma_processor.image_seq_length
        self.image_resolution = ( 3, paligemma_processor.image_processor.size["height"], paligemma_processor.image_processor.size["width"] )
        default_image_transform = paligemma_processor.image_processor

        # Switch on `image_resize_strategy`
        if self.image_resize_strategy == "resize-naive":
            self.image_transform = default_image_transform
        else:
            raise ValueError(f"Image Resize Strategy `{self.image_resize_strategy}` is not supported!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps each ViT block and then the _entire_ featurizer."""
        siglip_wrap_policy = partial(_module_wrap_policy, module_classes={SiglipVisionModel})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={SiglipEncoderLayer})
        return partial(_or_policy, policies=[siglip_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Runs transformed image/pixel tensor through vision backbone, returning _all_ patch features."""
        return self.featurizer(pixel_values)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return self.image_resolution

    @property
    def embed_dim(self) -> int:
        return self.featurizer.embed_dim

    @property
    def num_patches(self) -> int:
        return -1

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return SiglipEncoderLayer

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
