"""
base_vla.py

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, List, Optional

import torch
import torch.nn as nn

# === Abstract Base Class for ActionHead Models ===
class ActionHeadBackbone(nn.Module, ABC):
    def __init__(
        self,
        action_head_id: str,
    ) -> None:
        super().__init__()
        self.identifier = action_head_id

    @abstractmethod
    def forward(self, image_features): ...

    @abstractmethod
    def get_fsdp_wrapping_policy(self) -> Callable: ...

    # @property
    # @abstractmethod
    # def module_cls(self) -> Type[nn.Module]: ...
