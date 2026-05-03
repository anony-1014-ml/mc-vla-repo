"""
webdatasets.py

"""

import os
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple, Type
from abc import ABC, abstractmethod

import threading
from functools import partial

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from torchvision import transforms
from torchcodec.decoders import VideoDecoder
from transformers import GemmaTokenizer, GemmaTokenizerFast, CodeGenTokenizerFast, LlamaTokenizerFast, PreTrainedTokenizerBase

import webdataset as wds
import lmdb
import msgpack
import msgpack_numpy

from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.torch_utils import worker_init_function

#-------------------------------------------------------------------------------------------------------------
#
# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"
PAD_TOKEN = "<pad>"
# ACTION_TOKENS = [f"<unused{i}>" for i in range(99)]

#-------------------------------------------------------------------------------------------------------------
#
class BaseVideoWebDataset(IterableDataset[Dict[str, torch.Tensor]], ABC):
    #
    def __init__(
        self,
        shard_dir: Path,
        sample_n_frames: int,
        sample_stride: int,
        sample_size: Tuple[int, int],        
        resampled: bool,
        shardshuffle: bool,
        shuffle_buf_size: int,
    ) -> None:
        super().__init__()

        # #
        self.sample_n_frames    = sample_n_frames
        self.sample_stride      = sample_stride
        self.sample_size        = sample_size
        print(f"sample_n_frames : {sample_n_frames}, sample_stride : {sample_stride}, sample_size : {sample_size}")

        # #
        def read_mp4(sample, sample_n_frames, sample_stride, sample_size, resize_transforms):

            # filename #
            filename = sample["__key__"]
            sample["filename"] = filename

            # check mp4 #
            if "mp4" not in sample or ( mp4 := sample["mp4"] ) is None:
                return sample

            # #
            # video = torchvision.io.VideoReader(mp4, "video")
            video = VideoDecoder(mp4, device="cpu")

            # check frame_num #
            frame_num = video.metadata.num_frames
            required_frame_num = sample_stride * sample_n_frames
            if frame_num < required_frame_num:
                sample["mp4"] = None
                return sample

            # select frame_indices #
            random_range = frame_num - required_frame_num
            start_idx = np.random.randint(0, random_range + 1) if random_range > 0 else 0
            frame_indices = [ np.clip( start_idx + sample_stride*i, 0, frame_num - 1 ) for i in range(sample_n_frames)]
            #
            sample["frame_indices"] = frame_indices

            # #
            try:
                # #
                # frames = []
                # for idx in frame_indices:
                #     if video.metadata.height == sample_size[0] and video.metadata.width == sample_size[1]:
                #         frames.append(video[idx])
                #     else:
                #         frames.append(resize_transforms(video[idx]))
                # #
                # video_frames = torch.stack(frames, 0)

                # #
                video_frames = video.get_frames_at(frame_indices).data
                if video.metadata.height != sample_size[0] or video.metadata.width != sample_size[1]:
                    # video_frames = torch.stack( [ resize_transforms(video_frame) for video_frame in video_frames ], 0 )
                    video_frames = torch.nn.functional.interpolate( video_frames.float(), size=sample_size, mode='bilinear', align_corners=False ).to(torch.uint8)
            except:
                #
                video_frames = None
            #
            sample["mp4"] = video_frames

            # #
            return sample

        # #
        def worker_init_function_in_map(sample):
            info = get_worker_info()
            if info is not None:
                worker_init_function(info.id)
            return sample

        # #
        shards_list = [ str(path) for path in Path(shard_dir).glob('*.tar') ]
        shards_list = sorted(shards_list)
        # print(f"shards_list : {shards_list}")

        # transforms #
        resize_transforms = transforms.Compose([transforms.Resize(sample_size)])

        # #
        dataset = wds.WebDataset(shards_list, resampled=resampled, shardshuffle=shardshuffle, nodesplitter=wds.split_by_node, workersplitter=wds.split_by_worker)
        if shuffle_buf_size > 0: dataset = dataset.shuffle(shuffle_buf_size)
        dataset = dataset.decode()
        if "LOCAL_RANK" in os.environ: dataset = dataset.map( worker_init_function_in_map )
        dataset = dataset.map( partial(read_mp4, sample_n_frames=sample_n_frames, sample_stride=sample_stride, sample_size=sample_size, resize_transforms=resize_transforms) )

        # #
        self.webdataset = dataset

    def __len__(self) -> int:
        return len(self.webdataset)

    def __iter__(self) -> Dict[str, torch.Tensor]:
        # #
        sample_count = 0
        for sample in self.webdataset:
            if sample_count >= len(self.webdataset):
                break
            yield self.get_sample_dict(sample)
            sample_count += 1

    @abstractmethod
    def get_sample_dict(self, sample): ...

#
class VideoCaptionWebDataset(BaseVideoWebDataset):
    #
    def __init__(
        self,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        video_length_factor: int,
        #
        shard_dir: Path,
        lmdb_path: Path,
        sample_n_frames: int,
        sample_stride: int,
        sample_size: Tuple[int, int],        
        resampled: bool,
        shardshuffle: bool,
        shuffle_buf_size: int,
    ) -> None:

        # #
        super().__init__(
            shard_dir,
            sample_n_frames,
            sample_stride,
            sample_size,        
            resampled,
            shardshuffle,
            shuffle_buf_size,
        )

        # #
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.video_length_factor = video_length_factor
        self.dataset_type = "mineclip_caption"

        # Open Annotation DB #
        self.annotation_db = lmdb.open(str(lmdb_path.resolve()), map_size=int(1e12), readonly=True, lock=False, readahead=False, max_readers=2048) # 1(TB)
        # print(f"lmdb_path : {lmdb_path.resolve()}")

        # #
        def add_annotation(sample):

            # filename #
            filename = sample["filename"]

            # caption #
            key_caption = f"{filename}.caption".encode()
            if ( value_caption := get_txn().get(key_caption) ) is None:
                caption = None
            # #
            else:
                # Read caption #
                caption = msgpack.unpackb(value_caption, object_hook=msgpack_numpy.decode)

                # check #
                if caption == "":
                    caption = None
            #
            sample["caption"] = caption

            # #
            return sample        

        # #
        def info_from_json(shard_dir):
            info_file = Path(shard_dir) / 'dataset-size.json'
            if info_file.exists():
                with open(info_file, 'r') as f:
                    info_dic = json.load(f)
                return info_dic['dataset size']
            else:
                return -1

        # Prepare txn #
        thread_local = threading.local()
        def get_txn():
            if not hasattr(thread_local, "txn"):
                thread_local.txn = self.annotation_db.begin(write=False)
            return thread_local.txn

        # #
        self.webdataset = self.webdataset.map(add_annotation).select(lambda x: all(value is not None for value in x.values()))
        self.webdataset = self.webdataset.to_tuple("filename", "mp4", "caption", missing_is_error=False)

        # #
        dataset_size = info_from_json(str(shard_dir.resolve()))
        if dataset_size > 0:
            self.webdataset.with_epoch (dataset_size)
            self.webdataset.with_length(dataset_size)
        print(f"dataset_size : {len(self.webdataset)}")            

    def __del__(self):
        # #
        self.annotation_db.close()

    def get_sample_dict(self, sample):

        # #
        filename, video_frames, caption = sample

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels, token_type_ids = self.prompt_builder_fn(model_family="prismatic"), [], [], []
        if isinstance(self.tokenizer, (GemmaTokenizer, GemmaTokenizerFast)):
            # #
            prefix  = prompt_builder.add_turn_for_caption("human", ""     )
            suffix  = prompt_builder.add_turn_for_caption("gpt",   caption)

            # Tokenize Input IDs
            inputs = self.tokenizer(
                prefix.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.tokenizer.image_seq_length * (self.sample_n_frames // self.video_length_factor)),
                text_pair=suffix,
                return_token_type_ids=True,
                **self.tokenizer.tokenizer_kwargs
            )
            turn_input_ids = inputs.input_ids
            turn_token_type_ids = inputs.token_type_ids

            # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
            turn_labels = [IGNORE_INDEX if turn_token_type_ids[_] == 0 else turn_input_ids[_] for _ in range(len(turn_input_ids)) ]

            # Add to Trackers
            input_ids.extend(turn_input_ids)
            token_type_ids.extend(turn_token_type_ids)
            labels.extend(turn_labels)
        else:
            raise NotImplementedError("Tokenizer other than GemmaTokenizer is not implemented yet")

        # Tensorize =>> Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches after)
        #   - IMPORTANT => IF WE'RE USING HF LLM.forward(... labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels, token_type_ids = torch.tensor(input_ids), torch.tensor(labels), torch.tensor(token_type_ids)

        # Handle Truncation (if necessary)
        input_ids, labels, token_type_ids = input_ids[: self.tokenizer.model_max_length], labels[: self.tokenizer.model_max_length], token_type_ids[: self.tokenizer.model_max_length]

        # Set the <BOS> token's label to IGNORE_INDEX (since we're inserting the image patches right after)
        if not isinstance(self.tokenizer, (GemmaTokenizer, GemmaTokenizerFast)):
            labels[0] = IGNORE_INDEX

        # Process Image --> get "pixel_values" (will either be a torch.Tensor OR a Dict[str,torch.Tensor])
        if isinstance(self.image_transform, transforms.Compose) or hasattr(self.image_transform, "is_prismatic"):
            # This is a standard `torchvision.transforms` object or custom PrismaticVLM wrapper
            video_pixel_values = []
            for video_frame in video_frames:
                video_pixel_values.append( self.image_transform(video_frame) )
            video_pixel_values = torch.stack(video_pixel_values, 0)
        else:
            # Assume `image_transform` is an HF ImageProcessor...
            video_pixel_values = self.image_transform(video_frames, return_tensors="pt")["pixel_values"]

        # #
        return dict(video_pixel_values=video_pixel_values, input_ids=input_ids, labels=labels, token_type_ids=token_type_ids)
