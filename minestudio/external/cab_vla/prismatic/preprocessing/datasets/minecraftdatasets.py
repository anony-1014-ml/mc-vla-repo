"""
webdatasets.py

"""

import os
import math
import random
import copy
import json
from pathlib import Path
from typing import Dict, List, Tuple, Type, Union
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
from prismatic.models import get_action_proc
from prismatic.util.torch_utils import worker_init_function

from prismatic.action import ActionProc

from prismatic.preprocessing.datasets.minecraft import EventDataset
from prismatic.preprocessing.datasets.minecraft.callbacks import (
    ImageKernelCallback, 
    ActionKernelCallback, 
    MetaInfoKernelCallback, 
    SegmentationKernelCallback, 
)

#-------------------------------------------------------------------------------------------------------------
#
# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"
PAD_TOKEN = "<pad>"

# ACTION_TOKENS = [f"<unused{i}>" for i in range(99)]
ACTION_SPECIAL_TOKEN_NUM = 100 #50

#
text_task_token_dict = {
"all"                           : '<unused0>',
"contractor_combat_action"      : '<unused1>',
"contractor_mine_action"        : '<unused2>',
"contractor_use_action"         : '<unused3>',
"contractor_craft_action"       : '<unused4>',
"synthesized_craft_action"      : '<unused5>',
"synthesized_smelt_action"      : '<unused6>',
"combined_craft_smelt_action"   : '<unused7>',
}

#
action_task_token_dict = {
"all"                           :  10,
"contractor_combat_action"      :  11,
"contractor_mine_action"        :  12,
"contractor_use_action"         :  13,
"contractor_craft_action"       :  14,
"synthesized_craft_action"      :  15,
"synthesized_smelt_action"      :  16,
"combined_craft_smelt_action"   :  17,
}

#
text_action_task_token_dict = {
# action_task_token_id -> text_task_token #
10 : '<unused0>',    # "all"  
11 : '<unused1>',    # "contractor_combat_action"
12 : '<unused2>',    # "contractor_mine_action"
13 : '<unused3>',    # "contractor_use_action",
14 : '<unused4>',    # "contractor_craft_action"
15 : '<unused5>',    # "synthesized_craft_action"
16 : '<unused6>',    # "synthesized_smelt_action"
17 : '<unused7>',    # "combined_craft_smelt_action"

# text_task_token -> action_task_token_id #
'<unused0>' : 10,    # "all"  
'<unused1>' : 11,    # "contractor_combat_action"
'<unused2>' : 12,    # "contractor_mine_action"
'<unused3>' : 13,    # "contractor_use_action",
'<unused4>' : 14,    # "contractor_craft_action"
'<unused5>' : 15,    # "synthesized_craft_action"
'<unused6>' : 16,    # "synthesized_smelt_action"
'<unused7>' : 17,    # "combined_craft_smelt_action"
}

#
GOAL_FRAME_BINS     = 11
MAX_GOAL_FRAME_NUM  = 20
GOAL_FRAME_INTERVAL = MAX_GOAL_FRAME_NUM // ( GOAL_FRAME_BINS - 1 )

#
action_goal_token_dict_0c = { idx : 25 + GOAL_FRAME_BINS * 0 + idx for idx in range(GOAL_FRAME_BINS) }
action_goal_token_dict_1c = { idx : 25 + GOAL_FRAME_BINS * 1 + idx for idx in range(GOAL_FRAME_BINS) }
action_goal_token_dict_0c[                0] = 25 + GOAL_FRAME_BINS * 2 - 1
action_goal_token_dict_1c[                0] = 25 + GOAL_FRAME_BINS * 2 - 1
action_goal_token_dict_1c[GOAL_FRAME_BINS-1] = action_goal_token_dict_0c[GOAL_FRAME_BINS-1]

#-------------------------------------------------------------------------------------------------------------
#
def load_json_file(file_path: Union[str, Path], data_type="dict"):
    """
    Load a JSON file from the given path.

    Args:
        file_path (Union[str, Path]): Path to the JSON file.
        data_type (str): Expected data type of the JSON content ("dict" or "list").

    Returns:
        dict or list: Loaded JSON content. Returns an empty dictionary or list if the file does not exist.
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)  # Convert Path to string

    # Initialize an empty object based on the specified data type
    if data_type == "dict":
        json_file = dict()
    elif data_type == "list":
        json_file = list()
    else:
        raise ValueError("Invalid data type. Expected 'dict' or 'list'.")

    # Check if the file exists
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding="utf-8") as f:
                json_file = json.load(f)  # Load JSON content
        except IOError as e:
            print(f"[red]Failed to open file {file_path}: {e}[/red]")
        except json.JSONDecodeError as e:
            print(f"[red]Error parsing JSON file {file_path}: {e}[/red]")
    else:
        print(f"[yellow]File {file_path} does not exist. Returning an empty file...[/yellow]")

    return json_file

#
minecraft_prompt_library = load_json_file(Path(__file__).parent/"minecraft"/"instructions.json")

#       
def convert_text_to_intention(text, dataset_id):
    # #
    text = text.replace("minecraft.", "")
    if "craft_item:" in text:
        prompt = text.replace("craft_item:",  "Craft " ) + "."
        prompt = random.choice(minecraft_prompt_library.get(text, {"instruct":[prompt]})["instruct"])    
    elif "kill_entity:" in text:
        prompt = text.replace("kill_entity:", "Combat ") + "."
        prompt = random.choice(minecraft_prompt_library.get(text, {"instruct":[prompt]})["instruct"])    
    elif "mine_block:" in text:
        prompt = text.replace("mine_block:",  "Mine "  ) + "."
        prompt = random.choice(minecraft_prompt_library.get(text, {"instruct":[prompt]})["instruct"])    
    elif "use_item:" in text:
        prompt = text.replace("use_item:",  "Use "  ) + "."
        prompt = random.choice(minecraft_prompt_library.get(text, {"instruct":[prompt]})["instruct"])    
    else:
        prompt = text

    # #
    if dataset_id in ["contractor_craft_action", "synthesized_craft_action", "synthesized_smelt_action"]:
        dataset_id = str(np.random.choice([dataset_id, "combined_craft_smelt_action"], p=[0.6, 0.4]))

    # #
    prompt = text_task_token_dict.get( dataset_id, '<unused0>' ) + prompt

    # #
    # print(f"prompt : {prompt} / text : {text}")
    return prompt

#       
def convert_text_to_intention_for_inference(text):
    # #
    text = text.replace("minecraft.", "")
    if "craft_item:" in text:
        prompt = text.replace("craft_item:",  "Craft " ) + "."
        prompt = random.choice(minecraft_prompt_library.get(text, {"instruct":[prompt]})["instruct"])    
        prompt_task_id = 17
    elif "kill_entity:" in text:
        prompt = text.replace("kill_entity:", "Combat ") + "."
        prompt = random.choice(minecraft_prompt_library.get(text, {"instruct":[prompt]})["instruct"])    
        prompt_task_id = 11
    elif "mine_block:" in text:
        prompt = text.replace("mine_block:",  "Mine "  ) + "."
        prompt = random.choice(minecraft_prompt_library.get(text, {"instruct":[prompt]})["instruct"])    
        prompt_task_id = 12
    elif "use_item:" in text:
        prompt = text.replace("use_item:",  "Use "  ) + "."
        prompt = random.choice(minecraft_prompt_library.get(text, {"instruct":[prompt]})["instruct"])  
        prompt_task_id = 13
    else:
        prompt = text
        prompt_task_id = 10

    # #
    task_id = prompt_task_id

    # #
    prompt = text_action_task_token_dict.get( task_id, '<unused0>' ) + prompt

    # #
    return prompt

#-------------------------------------------------------------------------------------------------------------
#
def pad_actions(actions, target_rows, pad_value):
    return np.vstack([actions[:target_rows], np.full((max(0, target_rows - len(actions)), actions.shape[1]), pad_value)])

#-------------------------------------------------------------------------------------------------------------
#
class MinecraftSampleDictMixin:
    #
    def get_sample_dict(self, sample):

        # #
        mp4, actions_button, actions_camera, actions_discrete, intention_high, intention_fine, task_progress_0, task_progress_1 = sample

        # sampling video_frames #
        if self.val_flag:
            mp4 = mp4
        else:
            mp4 = mp4[self.video_start_idx :self.video_end_idx, ...]
        #
        video_frames = mp4

        # sampling actions_discrete #
        if self.val_flag:
            actions_discrete = actions_discrete
        else:
            actions_discrete = actions_discrete[self.action_start_idx:self.action_end_idx:self.action_stride, ...]
            actions_discrete = pad_actions(actions_discrete, self.max_action_length, IGNORE_INDEX)
        #
        actions = torch.tensor(actions_discrete)

        # sampling task_progress #
        if self.val_flag:
            task_progress_0 = task_progress_0
            task_progress_1 = task_progress_1
        else:
            task_progress_0 = task_progress_0[self.video_start_idx*self.sample_stride:self.video_end_idx*self.sample_stride:]
            task_progress_1 = task_progress_1[self.video_start_idx*self.sample_stride:self.video_end_idx*self.sample_stride:]
        #
        task_progress_id_0 = np.round(task_progress_0.mean()).astype(int)
        task_progress_id_1 = np.round(task_progress_1.mean()).astype(int)

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels, token_type_ids = self.prompt_builder_fn(model_family="prismatic"), [], [], []
        if isinstance(self.tokenizer, (GemmaTokenizer, GemmaTokenizerFast)):
            # Image + Text #
            if self.labels_type <= 1:

                # Set prefix & suffix
                prefix  = prompt_builder.add_turn_for_intention("human", intention_high)
                suffix  = prompt_builder.add_turn_for_intention("gpt",   intention_fine)

                # Tokenize Input IDs
                inputs = self.tokenizer(
                    prefix.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.tokenizer.image_seq_length * self.video_token_length),
                    text_pair=suffix,
                    return_token_type_ids=True,
                    **self.tokenizer.tokenizer_kwargs
                )

                # set turn_input_ids & turn_token_type_ids
                turn_input_ids = inputs.input_ids
                turn_token_type_ids = inputs.token_type_ids

                # set turn_labels
                turn_labels = [ IGNORE_INDEX if turn_token_type_ids[_] == 0 else turn_input_ids[_] for _ in range(len(turn_input_ids)) ]

                # Add to Trackers
                input_ids.extend(turn_input_ids)
                token_type_ids.extend(turn_token_type_ids)
                labels.extend(turn_labels)

            # Action(causal) #
            if self.labels_type <= 0:

                # set vocab_size
                vocab_size = math.ceil(len(self.tokenizer) / 64 ) * 64
                
                # set turn_input_ids
                turn_input_ids = []
                # <action_bos>
                turn_input_ids.append( 2 + vocab_size )
                # <action_task_token>
                action_task_token = action_task_token_dict.get(  self.dataset_id, IGNORE_INDEX )
                turn_input_ids.append( action_task_token + vocab_size )
                # <action_goal_token>
                # action_goal_token_dict_0c / action_goal_token_dict_1c
                action_goal_token = action_goal_token_dict_0c.get( task_progress_id_0, IGNORE_INDEX ) if task_progress_id_0 >= task_progress_id_1 else action_goal_token_dict_1c.get( task_progress_id_1, IGNORE_INDEX )
                turn_input_ids.append( action_goal_token + vocab_size )
                # <action_token>
                action_length = min( ( actions[:,0] != IGNORE_INDEX ).sum(), self.max_action_length )
                for idx in range( action_length ):
                    if   self.action_proc.method <= 3:
                        turn_input_ids.append( ACTION_SPECIAL_TOKEN_NUM +                                actions[idx,0] + vocab_size )
                        turn_input_ids.append( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + actions[idx,1] + vocab_size )
                    elif self.action_proc.method <= 5:
                        turn_input_ids.append( ACTION_SPECIAL_TOKEN_NUM +                                                                actions[idx,0] + vocab_size )
                        turn_input_ids.append( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim +                                 actions[idx,1] + vocab_size )
                        turn_input_ids.append( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.action_proc.camera_x_dim + actions[idx,2] + vocab_size )
                # <action_eos>
                turn_input_ids.append( 1 + vocab_size )
                # <pad>
                for idx in range( action_length, self.max_action_length ):
                    if   self.action_proc.method <= 3:
                        turn_input_ids.append( self.tokenizer.pad_token_id ) 
                        turn_input_ids.append( self.tokenizer.pad_token_id )
                    elif self.action_proc.method <= 5:
                        turn_input_ids.append( self.tokenizer.pad_token_id ) 
                        turn_input_ids.append( self.tokenizer.pad_token_id )
                        turn_input_ids.append( self.tokenizer.pad_token_id )
                
                # set turn_token_type_ids
                turn_token_type_ids = [ 2 if turn_input_ids[_] != self.tokenizer.pad_token_id else -1 for _ in range(len(turn_input_ids)) ]
                # turn_token_type_ids = np.array(turn_token_type_ids)
                # turn_token_type_ids_mask = (turn_token_type_ids == 2)
                # turn_token_type_ids[turn_token_type_ids_mask] += np.arange(turn_token_type_ids_mask.sum())
                # turn_token_type_ids = turn_token_type_ids.tolist()

                # set turn_labels
                # turn_labels = [ IGNORE_INDEX if turn_token_type_ids[_] == -1 else turn_input_ids[_] for _ in range(len(turn_input_ids)) ]
                turn_labels = [ IGNORE_INDEX if ( turn_token_type_ids[_] == -1 or turn_input_ids[_] == ( 2 + vocab_size ) ) else turn_input_ids[_] for _ in range(len(turn_input_ids)) ]
                # turn_labels = [ IGNORE_INDEX if ( turn_input_ids[_] in ( np.concatenate( [ [ self.tokenizer.pad_token_id ], np.array([2, 3, 4, 5, 6, 7, 8]) + vocab_size ] ) ) ) else turn_input_ids[_] for _ in range(len(turn_input_ids)) ]

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
        labels_weight = torch.tensor(self.labels_weight)

        # #
        return dict(video_pixel_values=video_pixel_values, input_ids=input_ids, labels=labels, token_type_ids=token_type_ids, actions=actions, labels_weight=labels_weight)

#-------------------------------------------------------------------------------------------------------------
#
class MinecraftContractorDataset(Dataset, MinecraftSampleDictMixin):
    #
    def __init__(
        self,
        action_head_identifier: str,
        image_transform: ImageTransform,
        tokenizer: PreTrainedTokenizerBase,
        prompt_builder_fn: Type[PromptBuilder],
        video_length_factor: int,
        #
        dataset_id: str,
        dataset_dirs: List[Path],
        event_info_list: List[Tuple[str,int]],
        min_within: int,
        win_len: int,
        win_bias: int,
        sample_n_frames: int,
        sample_stride: int,
        sample_size: Tuple[int, int],        
        #
        video_start_idx : int,
        video_end_idx   : int,
        action_start_idx: int,
        action_end_idx  : int,
        action_stride   : int,
    ) -> None:
        # #
        super().__init__()

        # #
        self.image_transform, self.tokenizer = image_transform, tokenizer
        self.prompt_builder_fn = prompt_builder_fn
        self.video_length_factor = video_length_factor
        self.dataset_type = "contractor_action"
        self.dataset_id   = dataset_id

        # #
        self.sample_n_frames    = sample_n_frames
        self.sample_stride      = sample_stride
        self.sample_size        = sample_size
        print(f"sample_n_frames : {sample_n_frames}, sample_stride : {sample_stride}, sample_size : {sample_size}")

        # #
        self.video_start_idx     = video_start_idx
        self.video_end_idx       = video_end_idx
        self.action_start_idx    = action_start_idx
        self.action_end_idx      = action_end_idx
        self.action_stride       = action_stride
        self.video_token_length  = ( self.video_end_idx  - self.video_start_idx  ) // self.video_length_factor
        self.max_action_length   = ( self.action_end_idx - self.action_start_idx ) // self.action_stride
        print(f"video_token_length : {self.video_token_length}, max_action_length : {self.max_action_length}")

        # #
        self.sampling_weight            = 1.0
        self.labels_weight              = 1.0        
        self.labels_type                = 0
        self.val_flag                   = False
        self.null_action_filtering_flag = True

        # #
        self.action_proc = get_action_proc(action_head_identifier)

        # Configuration(Base)
        self.win_len        = win_len
        win_bias            = win_bias
        frame_width         = self.sample_size[1]
        frame_height        = self.sample_size[0]
        print(f"win_len : {win_len}, win_bias : {win_bias}")
        
        # Configuration(Event)
        # event_regex = 'minecraft.craft_item:.*'
        # event_regex = '(minecraft.kill_entity:.*)|(minecraft.mine_block:.*)|(minecraft.craft_item:.*)'
        event_regex = "|".join([ f"({event_info[0]})" for event_info in event_info_list ])

        # Define Modal Kernel Callbacks
        modal_kernel_callbacks = [
            ImageKernelCallback(
                num_workers=0,
                frame_width=frame_width, 
                frame_height=frame_height, 
                enable_video_aug=False,
            ), 
            ActionKernelCallback(),
            MetaInfoKernelCallback(),
            # SegmentationKernelCallback(frame_width=frame_width, frame_height=frame_height)
        ]

        # Create EventDataset
        self.event_dataset = EventDataset(
            dataset_dirs=dataset_dirs, 
            modal_kernel_callbacks=modal_kernel_callbacks,
            win_len=win_len,
            bias=win_bias,
            event_regex=event_regex,
            min_nearby=None,
            min_within=min_within,
            max_within_list=event_info_list,
        )
        # print(f"event_list : {self.event_dataset.event_list}")

        # #
        self.actions_button_keys, self.actions_camera_keys = [ key for key in sorted(self.event_dataset[0]['env_action'].keys()) if key != "camera" ], ["camera_0", "camera_1"]

    def __len__(self) -> int:
        return len(self.event_dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        # #
        data = self.event_dataset[idx]

        # #
        video           = data['image']
        timestamp       = data['timestamp']
        episode         = data['episode']
        text            = data['text']
        actions         = data['env_action']
        events          = data['meta_info']["events"]

        # set base_frame_indices #
        base_frame_indices = np.array( [ i for i in range(video.shape[0]) ] )

        ## set intention_high / intention_fine ##
        if   self.labels_type == 0:
            intention_high = "survival skill."
            intention_fine = convert_text_to_intention(text, self.dataset_id)
        elif self.labels_type == 1:
            intention_high = "event prediction."
            intention_fine = convert_text_to_intention(text, self.dataset_id)

        # get task_progress #
        # goal_frame_indices
        goal_frame_indices = []
        for frame_idx, event in enumerate(events):
            if event is None: continue
            for key in event.keys():
                if text in key or text.replace(":",":minecraft.") in key:
                    goal_frame_indices.append(frame_idx)
        # task_progress
        if len(goal_frame_indices) > 0:
            task_progress_0 = np.maximum.reduce( [ np.clip( MAX_GOAL_FRAME_NUM - np.abs(base_frame_indices - goal_frame_idx), 0, None ) * ( base_frame_indices >= goal_frame_idx ) for goal_frame_idx in goal_frame_indices ] )
            task_progress_1 = np.maximum.reduce( [ np.clip( MAX_GOAL_FRAME_NUM - np.abs(base_frame_indices - goal_frame_idx), 0, None ) * ( base_frame_indices <= goal_frame_idx ) for goal_frame_idx in goal_frame_indices ] )
            task_progress_0 = np.round( ( task_progress_0 + 0.01 ) / float(GOAL_FRAME_INTERVAL) ).astype(int)
            task_progress_1 = np.round( ( task_progress_1 + 0.01 ) / float(GOAL_FRAME_INTERVAL) ).astype(int)
        else:
            task_progress_0 = np.repeat(IGNORE_INDEX, len(base_frame_indices))
            task_progress_1 = np.repeat(IGNORE_INDEX, len(base_frame_indices))

        # get actions_button / actions_camera #
        actions_button = np.concatenate( [ actions[key].unsqueeze(0) for key in sorted(actions.keys()) if key != "camera" ], axis=0 ).transpose(1,0)
        actions_camera = actions["camera"]

        # get actions_discrete #
        # actions_env
        actions_button_keys, actions_camera_keys = [ key for key in sorted(actions.keys()) if key != "camera" ], ["camera_0", "camera_1"]
        actions_env = { key : actions_button[:,idx] for idx, key in enumerate(actions_button_keys) }
        actions_env['camera'] = actions_camera.numpy()
        # actions_policy
        actions_policy = self.action_proc.action_env_to_policy(actions_env)
        # actions_discrete
        if   self.action_proc.method <= 3:
            actions_discrete = np.concatenate( [ actions_policy['buttons'], actions_policy['camera' ] ], axis=1 )
        elif self.action_proc.method <= 5:
            actions_discrete = np.concatenate( [ actions_policy['buttons'], actions_policy['camera_x'], actions_policy['camera_y'] ], axis=1 )

        # filter out null_action #
        if self.null_action_filtering_flag:
            # calc valid_actions_indices #
            if   self.action_proc.method == 0 or self.action_proc.method == 1:
                null_action = 0
                valid_actions_indices = np.where( actions_discrete[:, 0] != null_action )[0]
            elif self.action_proc.method == 2 or self.action_proc.method == 3:
                null_buttons_action = 0
                null_camera_action  = self.action_proc.camera_dim // 2
                valid_actions_indices = np.where( ( actions_discrete[:, 0] != null_buttons_action ) | ( actions_discrete[:, 1] != null_camera_action ) )[0]
            elif self.action_proc.method == 4 or self.action_proc.method == 5:
                null_buttons_action   = 0
                null_camera_x_action  = self.action_proc.camera_x_dim // 2
                null_camera_y_action  = self.action_proc.camera_y_dim // 2
                valid_actions_indices = np.where( ( actions_discrete[:, 0] != null_buttons_action ) | ( actions_discrete[:, 1] != null_camera_x_action ) | ( actions_discrete[:, 2] != null_camera_y_action ) )[0]

            # refresh base_frame_indices #
            base_frame_indices  = np.take( base_frame_indices,  valid_actions_indices )

        # set frame_indices #
        random_range = len(base_frame_indices) - ( self.sample_stride * self.sample_n_frames )
        start_idx = np.random.randint(0, random_range + 1) if random_range > 0 else 0
        # frame_indices = [ base_frame_indices[ start_idx + self.sample_stride * i ] for i in range(self.sample_n_frames)]
        frame_indices = [ base_frame_indices[ np.clip( start_idx + self.sample_stride * i, 0, len(base_frame_indices) - 1 ) ] for i in range(self.sample_n_frames)]

        ## set mp4 ##
        video = np.transpose(video,(0,3,1,2))
        mp4 = np.take( video, frame_indices, axis=0 )

        ## set actions_button / actions_camera / actions_discrete ##
        # take 
        actions_button      = np.take( actions_button,   base_frame_indices, axis=0 )[start_idx:, ...]
        actions_camera      = np.take( actions_camera,   base_frame_indices, axis=0 )[start_idx:, ...]
        actions_discrete    = np.take( actions_discrete, base_frame_indices, axis=0 )[start_idx:, ...]
        #
        # pad
        actions_button      = pad_actions(actions_button,   1024, 0             )
        actions_camera      = pad_actions(actions_camera,   1024, 0.0           )
        actions_discrete    = pad_actions(actions_discrete, 1024, IGNORE_INDEX  )

        ## set task_progress ##
        # take
        task_progress_0 = np.take( task_progress_0, base_frame_indices, axis=0 )[start_idx:]
        task_progress_1 = np.take( task_progress_1, base_frame_indices, axis=0 )[start_idx:]
        # pad
        task_progress_0 = np.pad ( task_progress_0, (0, max(1024 - len(task_progress_0), 0)), mode='constant', constant_values=IGNORE_INDEX )
        task_progress_1 = np.pad ( task_progress_1, (0, max(1024 - len(task_progress_1), 0)), mode='constant', constant_values=IGNORE_INDEX )

        # #
        sample = mp4, actions_button, actions_camera, actions_discrete, intention_high, intention_fine, task_progress_0, task_progress_1

        # #
        return self.get_sample_dict(sample)
