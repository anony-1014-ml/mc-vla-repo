import os
import sys
import threading
import time
import re
import random
from einops import rearrange
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import queue
from collections import deque
import gymnasium

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

from transformers import GenerationMixin, PretrainedConfig

from huggingface_hub import PyTorchModelHubMixin
from minestudio.models.base_policy import MinePolicy, dict_map
from minestudio.utils.register import Registers

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'external/cab_vla'))
from prismatic import load_vla
from prismatic.action.action_head import make_action_head as make_action_head_vla
from prismatic.action.action_proc import ActionProc, CameraConfig
from minestudio.utils.vpt_lib.normalize_ewma import NormalizeEwma
from minestudio.utils.vpt_lib.scaled_mse_head import ScaledMSEHead

from scipy.special import softmax
# from scipy.special import expit as sigmoid

#-------------------------------------------------------------------------------------------------------------
#
IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"
ACTION_TOKENS = [f"<unused{i}>" for i in range(99)]

from prismatic.preprocessing.datasets.minecraftdatasets import ACTION_SPECIAL_TOKEN_NUM
from prismatic.preprocessing.datasets.minecraftdatasets import text_action_task_token_dict
from prismatic.preprocessing.datasets.minecraftdatasets import action_goal_token_dict_0c, action_goal_token_dict_1c
from prismatic.preprocessing.datasets.minecraftdatasets import convert_text_to_intention_for_inference

#---------------------------------------------------------------------------------------------
#
class ObservationBuffer:
    #
    def __init__(self, buffer_size):
        self.buffer_size    = buffer_size
        self.buffer         = deque(maxlen=self.buffer_size)
    #
    def append(self, obs, frame_idx):
        self.buffer.append((obs, frame_idx))
    #
    def empty(self):
        return not self.buffer
    #
    def is_ready(self):
        return len(self.buffer) == self.buffer_size
    #
    def get(self):
        return [ _[0] for _ in self.buffer ]

#
@Registers.model.register
class CaBVLAPolicy(MinePolicy, PyTorchModelHubMixin):
    #
    def __init__(self, 
        model_path: str,
        action_space = None,
        temperature  = 1.0, 
        nucleus_prob = 0.85,
    ):
        # #
        super().__init__(hiddim=-1)

        # #
        self.vla, self.image_transform, self.tokenizer, self.action_proc = load_vla(model_path, hf_token="", model_family="prismaticPaliGemma")
        self.prompt_builder_fn = self.vla.llm_backbone.prompt_builder_fn
        self.video_length_factor = self.vla.video_length_factor
        
        # #
        self.temperature, self.nucleus_prob = temperature, nucleus_prob
        print(f"temperature : {self.temperature}, nucleus_prob : {self.nucleus_prob}")
        #
        if   self.vla.action_head.identifier == "paligemma-vpt":
            self.action_proc = ActionProc(method=0, temperature=self.temperature, nucleus_prob=self.nucleus_prob)
        elif self.vla.action_head.identifier == "paligemma-vpt-cd":
            self.action_proc = ActionProc(method=1, temperature=self.temperature, nucleus_prob=self.nucleus_prob)
        elif self.vla.action_head.identifier == "paligemma-vpt-para":
            self.action_proc = ActionProc(method=2, temperature=self.temperature, nucleus_prob=self.nucleus_prob)
        elif self.vla.action_head.identifier == "paligemma-vpt-cd-para":
            self.action_proc = ActionProc(method=3, temperature=self.temperature, nucleus_prob=self.nucleus_prob)
        elif self.vla.action_head.identifier == "paligemma-vpt-paraxy":
            self.action_proc = ActionProc(method=4, temperature=self.temperature, nucleus_prob=self.nucleus_prob)
        elif self.vla.action_head.identifier == "paligemma-vpt-cd-paraxy":
            self.action_proc = ActionProc(method=5, temperature=self.temperature, nucleus_prob=self.nucleus_prob)
        #
        self.pi_head = self.action_proc.pi_head
        #
        self.value_head = ScaledMSEHead(self.vla.config.hidden_size, 1, norm_type="ewma", norm_kwargs=None)

        # #
        self.action_length          =  4 #  1
        self.video_length           =  4 #  4
        self.video_append_stride    =  1 #  1
        self.video_inference_stride =  4 #  1
        self.inference_wait_num     =  0 #  0

        # #
        self.action_gen_method      = 3 # [ default : 0, action_task_token : 1, action_goal_token : 2, action_task&goal_token : 3, action_task&goal_token(x5) : 4 ]
        self.output_hidden_states   = False #True

        # #
        self.action_goal_param_dict = {
        2 : { "window_size" : 10, "score_th" : 5.66 * 10.0 }, # AB   For Craft & Smelt
        }
        self.action_goal_token_dict_0c_inv  = { v : k for k, v in action_goal_token_dict_0c.items() }
        self.action_goal_token_dict_1c_inv  = { v : k for k, v in action_goal_token_dict_1c.items() }
        self.action_goal_token_dict_01c_inv = self.action_goal_token_dict_0c_inv | self.action_goal_token_dict_1c_inv
        #
        # full
        self.goal_weight_vec_zeroshot_full = np.array([ self.action_goal_token_dict_01c_inv[key] for idx, key in enumerate(sorted(self.action_goal_token_dict_01c_inv.keys()))]).astype(np.float64)
        # after
        self.goal_weight_vec_zeroshot_after = self.goal_weight_vec_zeroshot_full.copy()
        self.goal_weight_vec_zeroshot_after [ np.arange(len(self.goal_weight_vec_zeroshot_full)) >= len(self.goal_weight_vec_zeroshot_full) / 2     ] = 0.0
        # before
        self.goal_weight_vec_zeroshot_before = self.goal_weight_vec_zeroshot_full.copy()
        self.goal_weight_vec_zeroshot_before[ np.arange(len(self.goal_weight_vec_zeroshot_full)) <  len(self.goal_weight_vec_zeroshot_full) / 2 - 1 ] = 0.0
        # const
        self.goal_weight_vec_zeroshot_const = np.zeros(self.goal_weight_vec_zeroshot_full.shape)
        self.goal_weight_vec_zeroshot_const[:-1] = 10.0
        #
        self.goal_weight_vec_dict = {
            0 : self.goal_weight_vec_zeroshot_after,    # A
            1 : self.goal_weight_vec_zeroshot_before,   # B
            2 : self.goal_weight_vec_zeroshot_full,     # AB
            3 : self.goal_weight_vec_zeroshot_const,    # C
        }
        self.goal_logit_T = 1.0
        #
        print(f"action_goal_param_dict : {self.action_goal_param_dict}")
        print(f"goal_weight_vec_dict : {self.goal_weight_vec_dict}")

        # #
        self.inference_input_queue  = queue.Queue()
        self.inference_output_queue = queue.Queue()
        self.thread = threading.Thread(target=self.inference_worker)
        self.thread.start()

        # #
        self.clear_agent()

    #
    def clear_agent(self):

        # #
        self.frame_idx              =  0
        self.inference_elapsed_num  = -1
        self.observation_buffer     = ObservationBuffer(self.video_length)
        self.scheduled_actions      = {}

        # #
        self.intention_idx          = -1
        self.intention_completion   = False
        self.intention_goal_type    = -1
        self.intention_dict         = {}
        self.intention_history      = []

    #
    def stop(self):
        self.inference_input_queue.put((None,None,None,None,None))
        self.thread.join()

    #
    def __del__(self):
        self.stop()

    #
    def set_intention(self, intention):

        # set intention_high/fine #
        assert type(intention) == list and len(intention) == 2 and type(intention[0]) == str and type(intention[1]) == str
        intention_high = str(intention[0])
        intention_fine = str(intention[1])

        # set intention_goal_type #
        intention_goal_type = 2

        # set intention_dict #
        self.intention_idx += 1
        self.intention_completion = False
        self.intention_goal_type = intention_goal_type
        self.intention_dict[ self.intention_idx ] = [intention_high, intention_fine]
        print(f"intention_dict [ {self.intention_idx} ] : {intention_high} -> {intention_fine}")

    #
    def forward(self, input: Dict, memory: Optional[List[torch.Tensor]] = None) -> Dict:

        # Append Image for Action #
        if ( self.frame_idx % self.video_append_stride ) == 0:
            # #
            image = input['image']
            image = rearrange(input['image'], 'b t h w c -> b t c h w')

            # #
            observation_append_size = 1 if not self.observation_buffer.empty() else self.observation_buffer.buffer_size
            for _ in range(observation_append_size):
                self.observation_buffer.append( self.image_transform(image[0,0], return_tensors="pt")["pixel_values"], self.frame_idx )

        # Inference for Action #
        if ( self.frame_idx % self.video_inference_stride ) == 0:
           
            # #
            intention_high, intention_fine = self.intention_dict[ self.intention_idx ]

            # #
            if self.observation_buffer.is_ready():
                # get video_pixel_values #
                video_pixel_values = torch.cat( self.observation_buffer.get(), dim=0 )
                # print(f"video_pixel_values : {video_pixel_values.shape} {type(video_pixel_values)}")

                # put inference_input_queue #
                self.inference_input_queue.put((self.frame_idx, self.intention_idx, video_pixel_values, intention_high, intention_fine))

                # init inference_elapsed_num #
                self.inference_elapsed_num = 0

        # check inference_elapsed_num #
        if self.inference_elapsed_num >= self.inference_wait_num:
        # while not self.inference_output_queue.empty():

            # get inference_output_queue #
            if self.output_hidden_states:
                inference_frame_idx, inference_intention_idx, inference_action_logits_list, inference_action_goal, inference_action_hidden_state = self.inference_output_queue.get()
            else:
                inference_frame_idx, inference_intention_idx, inference_action_logits_list, inference_action_goal                                = self.inference_output_queue.get()
            self.inference_elapsed_num = -1

            # set new scheduled_actions #
            for idx in range(self.action_length):
                self.scheduled_actions[ inference_frame_idx + idx ] = [ inference_action_logits[0,idx] for inference_action_logits in inference_action_logits_list ]
                self.scheduled_actions[ inference_frame_idx + idx ].extend( [ inference_intention_idx, inference_action_goal ] )
                if self.output_hidden_states: self.scheduled_actions[ inference_frame_idx + idx ].extend( [ inference_action_hidden_state ] )

        # set current scheduled_action & delete old scheduled_actions #
        scheduled_action = self.scheduled_actions.pop(self.frame_idx, None)
        for key in list(self.scheduled_actions.keys()):
            if key < self.frame_idx:
                del self.scheduled_actions[key]

        # Set pi_logits # 
        #
        if   self.action_proc.method <= 3:
            if scheduled_action is not None:
                scheduled_action_logits_list = scheduled_action[:2]
                button_logits = scheduled_action_logits_list[0].unsqueeze(0).unsqueeze(0)
                camera_logits = scheduled_action_logits_list[1].unsqueeze(0).unsqueeze(0)
            else:
                button_logits = torch.zeros((1, 1, self.action_proc.buttons_dim), dtype=torch.float32)
                button_logits[0,0,0] = 1000.0
                camera_logits = torch.zeros((1, 1, self.action_proc.camera_dim ), dtype=torch.float32)
                camera_logits[0,0,self.action_proc.camera_dim // 2] = 1000.0
            #
            pi_logits = { "buttons" : button_logits, "camera" : camera_logits }
        #
        elif self.action_proc.method <= 5:        
            if scheduled_action is not None:
                scheduled_action_logits_list = scheduled_action[:3]
                button_logits   = scheduled_action_logits_list[0].unsqueeze(0).unsqueeze(0)
                camera_x_logits = scheduled_action_logits_list[1].unsqueeze(0).unsqueeze(0)
                camera_y_logits = scheduled_action_logits_list[2].unsqueeze(0).unsqueeze(0)
            else:
                button_logits   = torch.zeros((1, 1, self.action_proc.buttons_dim  ), dtype=torch.float32)
                button_logits[0,0,0] = 1000.0
                camera_x_logits = torch.zeros((1, 1, self.action_proc.camera_x_dim ), dtype=torch.float32)
                camera_x_logits[0,0,self.action_proc.camera_x_dim // 2] = 1000.0
                camera_y_logits = torch.zeros((1, 1, self.action_proc.camera_y_dim ), dtype=torch.float32)
                camera_y_logits[0,0,self.action_proc.camera_y_dim // 2] = 1000.0
            #
            pi_logits = { "buttons" : button_logits, "camera_x" : camera_x_logits, "camera_y" : camera_y_logits }

        # Set intention_history #
        if self.output_hidden_states:
            scheduled_intention_idx, scheduled_action_goal, scheduled_action_hidden_state = -1, -1, -1
            if scheduled_action is not None: scheduled_intention_idx, scheduled_action_goal, scheduled_action_hidden_state = scheduled_action[-3:]
            self.intention_history.append( [ self.frame_idx, scheduled_intention_idx, scheduled_action_goal, scheduled_action_hidden_state ] )
        else:
            scheduled_intention_idx, scheduled_action_goal = -1, -1
            if scheduled_action is not None: scheduled_intention_idx, scheduled_action_goal = scheduled_action[-2:]
            self.intention_history.append( [ self.frame_idx, scheduled_intention_idx, scheduled_action_goal ] )

        # Check intention_completion #
        if not self.intention_completion:
            action_goal_history = [ _[2] for _ in self.intention_history[-1000:] if _[1] == self.intention_idx ]
            if len(action_goal_history) >= self.action_goal_param_dict[ self.intention_goal_type ]["window_size"]:
                # action_goal_score = float(np.sum([ _[ self.intention_goal_type ] for _ in action_goal_history[-self.action_goal_param_dict[ self.intention_goal_type ]["window_size"]:] ]))
                action_goal_score = float(np.sum([ self.goal_weight_vec_dict[ self.intention_goal_type ] * softmax(np.array(goal_logit_vec) / self.goal_logit_T) for goal_logit_vec in action_goal_history[-self.action_goal_param_dict[ self.intention_goal_type ]["window_size"]:] ]))
                if action_goal_score >= self.action_goal_param_dict[ self.intention_goal_type ]["score_th"]:
                    self.intention_completion = True

        # Update frame_idx & inference_elapsed_num #
        self.frame_idx += 1
        if self.inference_elapsed_num >= 0: self.inference_elapsed_num += 1

        # Return #
        return {"pi_logits": pi_logits, "vpred": None}, [None]

    #
    def initial_state(self, batch_size: int = None) -> List[torch.Tensor]:
        pass

    #
    def inference_worker(self):
        # #
        while True:

            # get inference_input_queue #
            frame_idx, intention_idx, video_pixel_values, intention_high, intention_fine = self.inference_input_queue.get()
            if frame_idx is None and intention_idx is None and video_pixel_values is None and intention_high is None and intention_fine is None:
                break

            # do inference #
            output = self.do_inference(video_pixel_values, intention_high, intention_fine)

            # put inference_output_queue #
            if self.output_hidden_states:
                self.inference_output_queue.put((frame_idx, intention_idx, output.action_logits_list, output.action_goal, output.action_hidden_state))
            else:
                self.inference_output_queue.put((frame_idx, intention_idx, output.action_logits_list, output.action_goal))

    #
    def do_inference(self, video_pixel_values, intention_high : str, intention_fine : str):

        # prepare inference #
        input_ids, labels, token_type_ids, attention_mask, action_task_token_id = self.prepare_inference(intention_high, intention_fine)
        # print(f"input_ids : {input_ids.shape} {labels.shape} {token_type_ids.shape} {attention_mask.shape}")

        # Action(bi-directional) #
        if False:
            # do inference #
            start_inference = time.perf_counter()
            with torch.autocast("cuda", dtype=self.vla.llm_backbone.half_precision_dtype, enabled=self.vla.enable_mixed_precision_training):
                # #
                video_pixel_values = video_pixel_values.unsqueeze(0).to(self.vla.device)
                input_ids = input_ids.unsqueeze(0).to(self.vla.device)
                token_type_ids = token_type_ids.unsqueeze(0).to(self.vla.device)
                attention_mask = attention_mask.unsqueeze(0).to(self.vla.device)

                # #
                output: CausalLMOutputWithPast = self.vla(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    video_pixel_values=video_pixel_values,
                )
            end_inference = time.perf_counter()
            # print(f"Inference Time: {end_inference - start_inference:.6f} (sec)")

        # Action(causal) #
        else:
            # do inference #
            start_inference = time.perf_counter()
            with torch.autocast("cuda", dtype=self.vla.llm_backbone.half_precision_dtype, enabled=self.vla.enable_mixed_precision_training):
                # #
                video_pixel_values = video_pixel_values.unsqueeze(0).to(self.vla.device)
                input_ids = input_ids.unsqueeze(0).to(self.vla.device)
                token_type_ids = token_type_ids.unsqueeze(0).to(self.vla.device)
                attention_mask = attention_mask.unsqueeze(0).to(self.vla.device)

                # make action_constraint_fn #
                if   self.action_proc.method <= 3:
                    # default #
                    if   self.action_gen_method == 0:
                        #
                        def make_action_constraint_fn(input_length, action_length):
                            def constraint(batch_id, input_ids):
                                generated_length = input_ids.shape[-1] - input_length
                                if generated_length == action_length:
                                    return [1 + self.vla.vocab_size]
                                elif generated_length % 2 == 0:
                                    constraint_ids = np.arange( ACTION_SPECIAL_TOKEN_NUM                               , ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim                               )
                                    constraint_ids = constraint_ids + self.vla.vocab_size
                                    return constraint_ids.tolist()
                                else:
                                    constraint_ids = np.arange( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim, ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.action_proc.camera_dim )
                                    constraint_ids = constraint_ids + self.vla.vocab_size
                                    return constraint_ids.tolist()
                            return constraint
                        #
                        gen_action_length = self.action_length * 2
                        action_constraint_fn = make_action_constraint_fn(input_ids.shape[1], gen_action_length)      

                    # action_task_token #              
                    elif self.action_gen_method == 1:
                        #
                        def make_action_constraint_fn(input_length, action_length):
                            def constraint(batch_id, input_ids):
                                generated_length = input_ids.shape[-1] - input_length
                                if   generated_length == action_length:
                                    return [1 + self.vla.vocab_size]
                                elif generated_length == 0:
                                    return [action_task_token_id + self.vla.vocab_size]
                                elif ( generated_length - 1 ) % 2 == 0:
                                    constraint_ids = np.arange( ACTION_SPECIAL_TOKEN_NUM                               , ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim                               )
                                    constraint_ids = constraint_ids + self.vla.vocab_size
                                    return constraint_ids.tolist()
                                else:
                                    constraint_ids = np.arange( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim, ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.action_proc.camera_dim )
                                    constraint_ids = constraint_ids + self.vla.vocab_size
                                    return constraint_ids.tolist()
                            return constraint
                        #
                        gen_action_length = self.action_length * 2 + 1
                        action_constraint_fn = make_action_constraint_fn(input_ids.shape[1], gen_action_length)

                    # action_goal_token #              
                    elif self.action_gen_method == 2:
                        #
                        def make_action_constraint_fn(input_length, action_length):
                            def constraint(batch_id, input_ids):
                                generated_length = input_ids.shape[-1] - input_length
                                if   generated_length == action_length:
                                    return [1 + self.vla.vocab_size]
                                elif generated_length == 0:
                                    constraint_ids = np.arange( 25, 25 + 11 )
                                    constraint_ids = constraint_ids + self.vla.vocab_size
                                    return constraint_ids.tolist()
                                elif ( generated_length - 1 ) % 2 == 0:
                                    constraint_ids = np.arange( ACTION_SPECIAL_TOKEN_NUM                               , ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim                               )
                                    constraint_ids = constraint_ids + self.vla.vocab_size
                                    return constraint_ids.tolist()
                                else:
                                    constraint_ids = np.arange( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim, ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.action_proc.camera_dim )
                                    constraint_ids = constraint_ids + self.vla.vocab_size
                                    return constraint_ids.tolist()
                            return constraint
                        #
                        gen_action_length = self.action_length * 2 + 1
                        action_constraint_fn = make_action_constraint_fn(input_ids.shape[1], gen_action_length)

                    # action_task&goal_token #
                    elif self.action_gen_method == 3:
                        #
                        def make_action_constraint_fn(input_length, action_length):
                            def constraint(batch_id, input_ids):
                                generated_length = input_ids.shape[-1] - input_length
                                if   generated_length == action_length:
                                    return [1 + self.vla.vocab_size]
                                elif generated_length == 0:
                                    return [action_task_token_id + self.vla.vocab_size]
                                elif generated_length == 1:
                                    constraint_ids = np.array( sorted( [ k for k in self.action_goal_token_dict_01c_inv.keys() ] ) )
                                    constraint_ids = constraint_ids + self.vla.vocab_size
                                    return constraint_ids.tolist()
                                elif generated_length % 2 == 0:
                                    constraint_ids = np.arange( ACTION_SPECIAL_TOKEN_NUM                               , ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim                               )
                                    constraint_ids = constraint_ids + self.vla.vocab_size
                                    return constraint_ids.tolist()
                                else:
                                    constraint_ids = np.arange( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim, ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.action_proc.camera_dim )
                                    constraint_ids = constraint_ids + self.vla.vocab_size
                                    return constraint_ids.tolist()
                            return constraint
                        #
                        gen_action_length = self.action_length * 2 + 2
                        action_constraint_fn = make_action_constraint_fn(input_ids.shape[1], gen_action_length)
                #
                elif self.action_proc.method <= 5:
                    #
                    def make_action_constraint_fn(input_length, action_length):
                        def constraint(batch_id, input_ids):
                            generated_length = input_ids.shape[-1] - input_length
                            if generated_length == action_length:
                                return [1 + self.vla.vocab_size]
                            elif generated_length % 3 == 0:
                                constraint_ids = np.arange( ACTION_SPECIAL_TOKEN_NUM                               , ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim                               )
                                constraint_ids = constraint_ids + self.vla.vocab_size
                                return constraint_ids.tolist()
                            elif generated_length % 3 == 1:
                                constraint_ids = np.arange( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim, ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.action_proc.camera_x_dim )
                                constraint_ids = constraint_ids + self.vla.vocab_size
                                return constraint_ids.tolist()
                            else:
                                constraint_ids = np.arange( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.action_proc.camera_x_dim, ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.action_proc.camera_x_dim + self.action_proc.camera_y_dim )
                                constraint_ids = constraint_ids + self.vla.vocab_size
                                return constraint_ids.tolist()
                        return constraint
                    #
                    gen_action_length = self.action_length * 3
                    action_constraint_fn = make_action_constraint_fn(input_ids.shape[1], gen_action_length)

                # #
                if False:
                    output = self.vla.generate(
                        video_pixel_values=video_pixel_values,
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        eos_token_id=[1 + self.vla.vocab_size],
                        max_new_tokens=( self.action_length * 2 + 1 ),
                        #
                        do_sample=False,
                        prefix_allowed_tokens_fn=action_constraint_fn,
                        #
                        return_dict_in_generate=True,
                        output_logits=True,
                    )
                else:
                    output = self.vla.generate(
                        video_pixel_values=video_pixel_values,
                        input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask,
                        eos_token_id=[1 + self.vla.vocab_size],
                        max_new_tokens=( gen_action_length + 1 ),
                        #
                        do_sample=True,
                        top_p=self.nucleus_prob,
                        temperature=self.temperature,
                        prefix_allowed_tokens_fn=action_constraint_fn,
                        #
                        return_dict_in_generate=True,
                        output_logits=True,
                        output_hidden_states=self.output_hidden_states,
                    )
            end_inference = time.perf_counter()
            print(f"Inference Time: {end_inference - start_inference:.6f} (sec)")

        # set action_logits_list : (batch_size, action_length, buttons_dim), (batch_size, action_length, camera_dim) #
        # with output.logits #
        if False:
            # #
            action_logits_list_0 = torch.stack(
                [
                    logits[:,self.vla.vocab_size:][:,ACTION_SPECIAL_TOKEN_NUM:ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim].to("cpu")
                    for logits_idx, logits in enumerate(output.logits[:-1]) if logits_idx % 2 == 0
                ],
                dim=1
            )
            action_logits_list_1 = torch.stack(
                [
                    logits[:,self.vla.vocab_size:][:,ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim:ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.action_proc.camera_dim].to("cpu")
                    for logits_idx, logits in enumerate(output.logits[:-1]) if logits_idx % 2 == 1
                ],
                dim=1
            )

            # #
            output.action_logits_list = [action_logits_list_0.cpu(), action_logits_list_1.cpu()]                

        # with output.sequences #
        else:
            if   self.action_proc.method <= 3:
                # default #
                if   self.action_gen_method == 0:
                    # #
                    gen_ids = output.sequences[..., input_ids.shape[1] :]
                    gen_ids_buttons = gen_ids[:, :-1:2] - ( ACTION_SPECIAL_TOKEN_NUM                                + self.vla.vocab_size )
                    gen_ids_camera  = gen_ids[:,1:-1:2] - ( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.vla.vocab_size )
                    gen_ids_task    = -1
                    gen_ids_goal    = -1

                # action_task_token #
                elif self.action_gen_method == 1:
                    # #
                    gen_ids = output.sequences[..., input_ids.shape[1] :]
                    gen_ids_buttons = gen_ids[:,1:-1:2] - ( ACTION_SPECIAL_TOKEN_NUM                                + self.vla.vocab_size )
                    gen_ids_camera  = gen_ids[:,2:-1:2] - ( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.vla.vocab_size )
                    gen_ids_task    = gen_ids[:,0] - self.vla.vocab_size
                    gen_ids_goal    = -1

                # action_goal_token #
                elif self.action_gen_method == 2:
                    # #
                    gen_ids = output.sequences[..., input_ids.shape[1] :]
                    gen_ids_buttons = gen_ids[:,1:-1:2] - ( ACTION_SPECIAL_TOKEN_NUM                                + self.vla.vocab_size )
                    gen_ids_camera  = gen_ids[:,2:-1:2] - ( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.vla.vocab_size )
                    gen_ids_task    = -1
                    gen_ids_goal    = int(( gen_ids[:,0] - self.vla.vocab_size - 25 ).cpu())

                # action_task&goal_token #
                elif self.action_gen_method == 3:
                    # #
                    gen_ids = output.sequences[..., input_ids.shape[1] :]
                    gen_ids_buttons = gen_ids[:,2:-1:2] - ( ACTION_SPECIAL_TOKEN_NUM                                + self.vla.vocab_size )
                    gen_ids_camera  = gen_ids[:,3:-1:2] - ( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.vla.vocab_size )
                    gen_ids_task    = gen_ids[:,0] - self.vla.vocab_size
                    # gen_ids_goal_token = int(( gen_ids[:, 1] - self.vla.vocab_size ).cpu())
                    # gen_ids_goal_a   =      self.action_goal_token_dict_0c_inv .get( gen_ids_goal_token, 0 )
                    # gen_ids_goal_b   =      self.action_goal_token_dict_1c_inv .get( gen_ids_goal_token, 0 )
                    # gen_ids_goal_ab  =      self.action_goal_token_dict_01c_inv.get( gen_ids_goal_token, 0 )
                    # gen_ids_goal_ac  = int( self.action_goal_token_dict_0c_inv .get( gen_ids_goal_token, 0 ) > 0 )
                    # gen_ids_goal_bc  = int( self.action_goal_token_dict_1c_inv .get( gen_ids_goal_token, 0 ) > 0 )
                    # gen_ids_goal_abc = int( self.action_goal_token_dict_01c_inv.get( gen_ids_goal_token, 0 ) > 0 )
                    # gen_ids_goal     = [ gen_ids_goal_a, gen_ids_goal_b, gen_ids_goal_ab, gen_ids_goal_ac, gen_ids_goal_bc, gen_ids_goal_abc ]
                    goal_token_losits = output.logits[1][0, sorted( [ k + self.vla.vocab_size for k in self.action_goal_token_dict_01c_inv.keys() ] )] # output.logits  : seq_length x ( batch_size, combined_vocab_size=(vocab_size + action_vocab_size) )
                    gen_ids_goal = goal_token_losits.cpu().tolist()

                    # ??? : 1 is the token index for predicting goal_token : ( boa(0) -> task(1) -> goal(2) -> action(3) -> ... ) #
                    if self.output_hidden_states:
                        gen_hidden_state_goal = output.hidden_states[1][-1].squeeze()

                print(f"action_task : {gen_ids_task}, action_goal : {gen_ids_goal}")

                # #
                gen_batch_size, gen_action_size = gen_ids_buttons.shape[0], gen_ids_buttons.shape[1]
                action_logits_list_0 = torch.zeros((gen_batch_size, gen_action_size, self.action_proc.buttons_dim), dtype=torch.float32)
                action_logits_list_1 = torch.zeros((gen_batch_size, gen_action_size, self.action_proc.camera_dim ), dtype=torch.float32)
                batch_idx  = torch.arange(gen_batch_size ).unsqueeze(1).expand(-1, gen_action_size) # shape: (batch_size, action_length)
                action_idx = torch.arange(gen_action_size).unsqueeze(0).expand(gen_batch_size,  -1) # shape: (batch_size, action_length)
                action_logits_list_0[ batch_idx, action_idx, gen_ids_buttons] = 1000.0
                action_logits_list_1[ batch_idx, action_idx, gen_ids_camera ] = 1000.0

                # #
                output.action_logits_list = [action_logits_list_0.cpu(), action_logits_list_1.cpu()]      
                output.action_goal = gen_ids_goal
                if self.output_hidden_states:
                    # output.action_hidden_state = gen_hidden_state_goal.cpu().tolist()
                    output.action_hidden_state = [ round(_, 6) for _ in gen_hidden_state_goal.cpu().tolist() ]
            #
            elif self.action_proc.method <= 5:                    
                # #
                gen_ids = output.sequences[..., input_ids.shape[1] :]
                gen_ids_buttons  = gen_ids[:, :-1:3] - ( ACTION_SPECIAL_TOKEN_NUM                                                                + self.vla.vocab_size )
                gen_ids_camera_x = gen_ids[:,1:-1:3] - ( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim                                 + self.vla.vocab_size )
                gen_ids_camera_y = gen_ids[:,2:-1:3] - ( ACTION_SPECIAL_TOKEN_NUM + self.action_proc.buttons_dim + self.action_proc.camera_x_dim + self.vla.vocab_size )

                # #
                gen_batch_size, gen_action_size = gen_ids_buttons.shape[0], gen_ids_buttons.shape[1]
                action_logits_list_0 = torch.zeros((gen_batch_size, gen_action_size, self.action_proc.buttons_dim ), dtype=torch.float32)
                action_logits_list_1 = torch.zeros((gen_batch_size, gen_action_size, self.action_proc.camera_x_dim), dtype=torch.float32)
                action_logits_list_2 = torch.zeros((gen_batch_size, gen_action_size, self.action_proc.camera_y_dim), dtype=torch.float32)
                batch_idx  = torch.arange(gen_batch_size ).unsqueeze(1).expand(-1, gen_action_size) # shape: (batch_size, action_length)
                action_idx = torch.arange(gen_action_size).unsqueeze(0).expand(gen_batch_size,  -1) # shape: (batch_size, action_length)
                action_logits_list_0[ batch_idx, action_idx, gen_ids_buttons ] = 1000.0
                action_logits_list_1[ batch_idx, action_idx, gen_ids_camera_x] = 1000.0
                action_logits_list_2[ batch_idx, action_idx, gen_ids_camera_y] = 1000.0

                # #
                output.action_logits_list = [action_logits_list_0.cpu(), action_logits_list_1.cpu(), action_logits_list_2.cpu()]
        # #
        return output

    #
    def prepare_inference(self, intention_high : str, intention_fine : str):

        # Create Prompt Builder --> add each message sequentially
        prompt_builder, input_ids, labels, token_type_ids = self.prompt_builder_fn(model_family="prismatic"), [], [], []
        if True:#isinstance(self.tokenizer, (GemmaTokenizer, GemmaTokenizerFast)):
            # Image + Text #
            # Set prefix & suffix
            prefix  = prompt_builder.add_turn_for_intention("human", intention_high)
            suffix  = prompt_builder.add_turn_for_intention("gpt",   intention_fine)

            # Tokenize Input IDs
            inputs = self.tokenizer(
                prefix.replace(IMAGE_TOKEN, IMAGE_TOKEN * self.tokenizer.image_seq_length * ( self.video_length // self.video_length_factor)),
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

            # Action(bi-directional) #
            if False:
                # Set action_text
                action_text = ""
                for idx in range(self.action_length): action_text += ACTION_TOKENS[idx]

                # Tokenize Input IDs
                inputs = self.tokenizer(
                    action_text,
                    return_token_type_ids=False,#True,
                    **self.tokenizer.tokenizer_kwargs
                )
                turn_input_ids = inputs.input_ids
                # turn_token_type_ids = inputs.token_type_ids
                turn_token_type_ids = np.full_like(inputs.input_ids, 2)

                # [CRITICAL] We do not want to take the loss for the "USER: <msg>" prompts =>> just the responses!
                # turn_labels = [IGNORE_INDEX if turn_token_type_ids[_] == 0 else turn_input_ids[_] for _ in range(len(turn_input_ids)) ]
                turn_labels = [IGNORE_INDEX if turn_token_type_ids[_] == 2 else turn_input_ids[_] for _ in range(len(turn_input_ids)) ]

            # Action(causal) #
            else:
                # set turn_input_ids
                turn_input_ids = []
                # <action_bos>
                turn_input_ids.append( 2 + self.vla.vocab_size )

                # set turn_token_type_ids
                turn_token_type_ids = [ 2 if turn_input_ids[_] != self.tokenizer.pad_token_id else -1 for _ in range(len(turn_input_ids)) ]

                # set turn_labels
                turn_labels = [ IGNORE_INDEX if turn_token_type_ids[_] == -1 else turn_input_ids[_] for _ in range(len(turn_input_ids)) ]

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

        # #
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # set action_task_token_id #
        text_task_token = re.match( r"<unused\d+>", intention_fine )
        if text_task_token:
            text_task_token = text_task_token.group()
            action_task_token_id = text_action_task_token_dict.get( text_task_token, 10 )
        else:
            action_task_token_id = 10

        # #
        return input_ids, labels, token_type_ids, attention_mask, action_task_token_id

@Registers.model_loader.register
def load_cab_vla_policy(ckpt_path: Optional[str] = None):
    return None