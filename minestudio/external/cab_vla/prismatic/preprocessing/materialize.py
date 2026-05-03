"""
materialize.py

Factory class for initializing pretraining datasets on a per-VLM basis; provides and exports individual functions for
clear control flow.
"""

from typing import Tuple, Type, Union

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from prismatic.conf import DatasetConfig
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.preprocessing.datasets import AlignDataset, FinetuneDataset
from prismatic.preprocessing.datasets import VideoCaptionWebDataset, MinecraftContractorDataset
from prismatic.util.data_utils import PaddedCollatorForLanguageModeling
from prismatic.util.data_utils import PaddedCollatorForVideoCaptionModeling, PaddedCollatorForVideoActionModeling

# Dataset Initializers =>> Maps Stage --> cls()
DATASET_INITIALIZER = {"align": AlignDataset, "finetune": FinetuneDataset, "full-finetune": FinetuneDataset}


def get_dataset_and_collator(
    stage: str,
    dataset_cfg: DatasetConfig,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
) -> Tuple[Dataset, PaddedCollatorForLanguageModeling]:
    dataset_cls = DATASET_INITIALIZER[stage]
    dataset_root_dir = dataset_cfg.dataset_root_dir
    collator = PaddedCollatorForLanguageModeling(
        tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, padding_side=padding_side
    )

    # Switch on `stage`
    if stage == "align":
        annotation_json, image_dir = dataset_cfg.align_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json, dataset_root_dir / image_dir, image_transform, tokenizer
        )
        return dataset, collator

    elif stage == "finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    elif stage == "full-finetune":
        annotation_json, image_dir = dataset_cfg.finetune_stage_components
        dataset = dataset_cls(
            dataset_root_dir / annotation_json,
            dataset_root_dir / image_dir,
            image_transform,
            tokenizer,
            prompt_builder_fn=prompt_builder_fn,
        )
        return dataset, collator

    else:
        raise ValueError(f"Stage `{stage}` is not supported!")


def get_vla_dataset_and_collator(
    dataset_cfg: DatasetConfig,
    action_head_identifier: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    prompt_builder_fn: Type[PromptBuilder],
    default_image_resolution: Tuple[int, int, int],
    padding_side: str           = "right",
    video_length_factor: int    = 1,
    resampled: bool             = True,
    shardshuffle: bool          = True,
    shuffle_buf_size: int       = 10,
) -> Tuple[Dataset, Union[PaddedCollatorForVideoCaptionModeling, PaddedCollatorForVideoActionModeling]]:

    # Switch on `dataset_id`
    if "contractor_" in dataset_cfg.dataset_id:
        #
        dataset_root_dir = dataset_cfg.dataset_root_dir
        dataset_id       = dataset_cfg.dataset_id
        dataset_dirs = list(dataset_cfg.dataset_components)
        for idx in range(len(dataset_dirs)):
            dataset_dirs[idx] = dataset_root_dir / dataset_dirs[idx]
        event_info_list = list(dataset_cfg.event_info_list)
        min_within      = dataset_cfg.min_within
        win_len         = dataset_cfg.win_len
        win_bias        = dataset_cfg.win_bias
        sample_n_frames = dataset_cfg.sample_n_frames
        sample_stride   = dataset_cfg.sample_stride
        sample_size     = dataset_cfg.sample_size
        #
        video_start_idx     = dataset_cfg.video_start_idx
        video_end_idx       = dataset_cfg.video_end_idx
        action_start_idx    = dataset_cfg.action_start_idx
        action_end_idx      = dataset_cfg.action_end_idx
        action_stride       = dataset_cfg.action_stride
        #
        dataset_sampling_weight = dataset_cfg.dataset_sampling_weight
        dataset_labels_weight   = dataset_cfg.dataset_labels_weight
        dataset_labels_type     = dataset_cfg.dataset_labels_type
        #
        dataset = MinecraftContractorDataset(
            #
            action_head_identifier,
            image_transform,
            tokenizer,
            prompt_builder_fn,
            video_length_factor,
            #
            dataset_id          = dataset_id,
            dataset_dirs        = dataset_dirs,
            event_info_list     = event_info_list,
            min_within          = min_within,
            win_len             = win_len,
            win_bias            = win_bias,
            sample_n_frames     = sample_n_frames,
            sample_stride       = sample_stride,
            sample_size         = sample_size,        
            #
            video_start_idx     = video_start_idx,
            video_end_idx       = video_end_idx,
            action_start_idx    = action_start_idx,
            action_end_idx      = action_end_idx,
            action_stride       = action_stride,          
        )
        dataset.sampling_weight = dataset_sampling_weight
        dataset.labels_weight   = dataset_labels_weight
        dataset.labels_type     = dataset_labels_type
        #
        if tokenizer is not None:
            collator = PaddedCollatorForVideoActionModeling(
                tokenizer.model_max_length, tokenizer.pad_token_id, default_image_resolution, sample_n_frames, padding_side=padding_side
            )
        else:
            collator = PaddedCollatorForVideoActionModeling(
                -1, 0, default_image_resolution, sample_n_frames, padding_side=padding_side
            )
        #       
        return dataset, collator        

    else:
        raise ValueError(f"dataset_id `{dataset_cfg.dataset_id}` is not supported!")
