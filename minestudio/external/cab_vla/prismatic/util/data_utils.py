"""
data_utils.py

General utilities and classes for facilitating data loading and collation.
"""

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


@dataclass
class PaddedCollatorForLanguageModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        self.dummy_pixel_values = torch.zeros(self.default_image_resolution, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        pixel_values = [instance["pixel_values"] for instance in instances]
        token_type_ids = [instance["token_type_ids"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=-1)

        # Truncate (if necessary)
        input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
        token_type_ids = token_type_ids[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(pixel_values)) if pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            # pixel_values = torch.stack([self.dummy_pixel_values for _ in range(len(input_ids))])
            pixel_values = None
        elif isinstance(pv_example := pixel_values[multimodal_indices[0]], torch.Tensor):
            pixel_values = torch.stack(
                [
                    pixel_values[idx] if idx in multimodal_indices else self.dummy_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            pixel_values = {
                k: torch.stack(
                    [
                        pixel_values[idx][k] if idx in multimodal_indices else self.dummy_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        return dict(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
            token_type_ids=token_type_ids,
        )

@dataclass
class PaddedCollatorForVideoCaptionModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    default_n_frames: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        video_pixel_values_shape = (self.default_n_frames,) + self.default_image_resolution
        self.dummy_video_pixel_values = torch.zeros(video_pixel_values_shape, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        video_pixel_values = [instance["video_pixel_values"] for instance in instances]
        token_type_ids = [instance["token_type_ids"] for instance in instances]

        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=-1)

        # Truncate (if necessary)
        if self.model_max_length > 0:
            input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
            token_type_ids = token_type_ids[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(video_pixel_values)) if video_pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `video_pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            # video_pixel_values = torch.stack([self.dummy_video_pixel_values for _ in range(len(input_ids))])
            video_pixel_values = None
        elif isinstance(pv_example := video_pixel_values[multimodal_indices[0]], torch.Tensor):
            video_pixel_values = torch.stack(
                [
                    video_pixel_values[idx] if idx in multimodal_indices else self.dummy_video_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            video_pixel_values = {
                k: torch.stack(
                    [
                        video_pixel_values[idx][k] if idx in multimodal_indices else self.dummy_video_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `video_pixel_values` type = {type(video_pixel_values)}")

        return dict(
            video_pixel_values=video_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            multimodal_indices=multimodal_indices,
            token_type_ids=token_type_ids,
        )

@dataclass
class PaddedCollatorForVideoActionModeling:
    model_max_length: int
    pad_token_id: int
    default_image_resolution: Tuple[int, int, int]
    default_n_frames: int
    padding_side: str = "right"
    pixel_values_dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        video_pixel_values_shape = (self.default_n_frames,) + self.default_image_resolution
        self.dummy_video_pixel_values = torch.zeros(video_pixel_values_shape, dtype=self.pixel_values_dtype)

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        video_pixel_values = [instance["video_pixel_values"] for instance in instances]
        token_type_ids = [instance["token_type_ids"] for instance in instances]
        actions = [instance["actions"] for instance in instances]
        labels_weight = [instance["labels_weight"] for instance in instances]

        # #
        actions = torch.stack(actions, dim=0)

        # #
        labels_weight = torch.stack(labels_weight)
        
        # For now, we only support Tokenizers with `padding_side = "right"` during Training (but plan to extend!)
        #   => Handle padding via RNN Utils => `pad_sequence`
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=-1)

        # Truncate (if necessary)
        if self.model_max_length > 0:
            input_ids, labels = input_ids[:, : self.model_max_length], labels[:, : self.model_max_length]
            token_type_ids = token_type_ids[:, : self.model_max_length]

        # Get `attention_mask` by checking for `pad_token_id`
        attention_mask = input_ids.ne(self.pad_token_id)

        # === Handle "unimodal" (language-only) vs. "multimodal" ===

        # Some examples are "language-only" --> build a Tensor of `multimodal_indices` that we can slice into easily
        multimodal_indices = torch.tensor(
            [idx for idx in range(len(video_pixel_values)) if video_pixel_values[idx] is not None], dtype=torch.long
        )

        # Stack all `video_pixel_values` --> depending on type (torch.Tensor, or Dict[str, torch.Tensor]) & presence of None
        if len(multimodal_indices) == 0:
            # video_pixel_values = torch.stack([self.dummy_video_pixel_values for _ in range(len(input_ids))])
            video_pixel_values = None
        elif isinstance(pv_example := video_pixel_values[multimodal_indices[0]], torch.Tensor):
            video_pixel_values = torch.stack(
                [
                    video_pixel_values[idx] if idx in multimodal_indices else self.dummy_video_pixel_values
                    for idx in range(len(input_ids))
                ]
            )
        elif isinstance(pv_example, dict):
            video_pixel_values = {
                k: torch.stack(
                    [
                        video_pixel_values[idx][k] if idx in multimodal_indices else self.dummy_video_pixel_values
                        for idx in range(len(input_ids))
                    ]
                )
                for k in pv_example
            }
        else:
            raise ValueError(f"Unsupported `video_pixel_values` type = {type(video_pixel_values)}")

        return dict(
            video_pixel_values=video_pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            labels_weight=labels_weight,
            multimodal_indices=multimodal_indices,
            token_type_ids=token_type_ids,
            actions=actions,
        )