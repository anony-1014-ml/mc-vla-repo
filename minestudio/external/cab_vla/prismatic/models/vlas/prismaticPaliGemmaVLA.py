"""
prismaticPaliGemmaVLA.py

"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from collections import OrderedDict

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers import GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import is_torchdynamo_compiling, ModelOutput
from transformers.cache_utils import Cache, HybridCache, StaticCache

from prismatic.models.vlms.base_vlm import ProjectorBackbone
from prismatic.models.vlas.base_vla import ActionHeadBackbone
from prismatic.models.vlms.prismaticPaliGemma import PrismaticPaliGemmaVLM
from prismatic.overwatch import initialize_overwatch

from einops import rearrange

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"

#-------------------------------------------------------------------------------------------------------------
#
class PaliGemmaCausalActionHeadBackbone(ActionHeadBackbone):
    def __init__(
        self,
        action_head_id:     str,
        vocab_size:         int,
        vocab_config:       List[int],
        hidden_size:        int,
        pad_token_id:       int,
        initializer_range:  int = 0.02,
    ) -> None:
        # #
        super().__init__(action_head_id)

        # #
        self.vocab_size     = vocab_size
        self.vocab_config   = vocab_config

        # #
        self.embed_tokens   = nn.Embedding(vocab_size, hidden_size, pad_token_id)
        self.action_head    = nn.Linear(hidden_size, vocab_size, bias=False)

        # #
        self.action_weight = torch.full((vocab_size,), 1.0)

        # #
        self.action_ignore_classes = []#[60+1]

        # gui_action_indexes #
        # spacial_token
        gui_action_indexes = np.arange(0, vocab_config[0]).tolist()
        # button_token
        if   vocab_config[1] == 8641:
            gui_action_indexes.extend( ( np.array([ 0, 4, 16, 20, 1, 5, 17, 21, 8640 ]) + vocab_config[0] ).tolist() )
        elif vocab_config[1] == 4321:   
            gui_action_indexes.extend( ( np.array([ 0, 2,  8, 10,               4320 ]) + vocab_config[0] ).tolist() )
        else:
            raise NotImplementedError
        # camera_token
        gui_action_indexes.extend( np.arange(vocab_config[0] + vocab_config[1], vocab_config[0] + vocab_config[1] + vocab_config[2]).tolist() )
        #
        self.gui_action_indexes = gui_action_indexes

        # # unmask_dict #
        # unmask_dict = OrderedDict()
        # for token_type_id in range(-1, 100000):
        #     if   token_type_id <= 5:
        #         unmask_dict[token_type_id] = [ 0, 0 ]
        #     elif token_type_id <= 7:
        #         unmask_dict[token_type_id] = [ 2, token_type_id -  6 ] 
        #     elif token_type_id <= 9:
        #         unmask_dict[token_type_id] = [ 4, token_type_id -  8 ]
        #     elif token_type_id <= 11:
        #         unmask_dict[token_type_id] = [ 6, token_type_id - 10 ]
        #     elif token_type_id <= 13:
        #         unmask_dict[token_type_id] = [ 8, token_type_id - 12 ]
        #     else:
        #         unmask_dict[token_type_id] = [10, token_type_id - 14 ]  
        #         # unmask_dict[token_type_id] = [ 0, 0 ]
        # self.unmask_dict = unmask_dict

        # # action_input_mapping #
        # action_input_mapping = { 4 : xxx, 5 : yyy, 6 : zzz } # { token_type_id : action_input_id } 
        # self.action_input_mapping = action_input_mapping

        # #
        self.apply(partial(self._init_weights, std=initializer_range))

    def _init_weights(self, module, std):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(self, hidden_state):
        # #
        action_logits = self.action_head(hidden_state)

        # #
        return action_logits

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps action_head."""
        action_head_wrap_policy = partial(_module_wrap_policy, module_classes={PaliGemmaCausalActionHeadBackbone})
        return action_head_wrap_policy

    # @property
    # def module_cls(self) -> Type[nn.Module]:
    #     return None

#-------------------------------------------------------------------------------------------------------------
#
@dataclass
class PrismaticPaliGemmaVLAOutputWithPast(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    loss_list: Optional[List[torch.FloatTensor]] = None
    logits: Optional[torch.FloatTensor] = None
    action_logits_list: Optional[List[torch.FloatTensor]] = None
    past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

#
class PrismaticPaliGemmaCausalVLA(PrismaticPaliGemmaVLM):

    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True

    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        projector: ProjectorBackbone,
        action_head: ActionHeadBackbone,
        enable_mixed_precision_training: bool = True,
    ) -> None:
        #---------------------------------------------------------------------
        # PrismaticPaliGemmaVLM
        #---------------------------------------------------------------------
        super().__init__(
            model_id,
            vision_backbone,
            llm_backbone,
            projector,
            enable_mixed_precision_training,
        )

        #---------------------------------------------------------------------
        # PrismaticPaliGemmaCausalVLA
        #---------------------------------------------------------------------
        # #
        self.video_spatial_pooling_size     = 1 #1
        self.video_temporal_pooling_size    = 1 #2
        self.video_length_factor = self.video_spatial_pooling_size**2 * self.video_temporal_pooling_size
        if self.video_spatial_pooling_size >= 2:
            self.video_spatial_pooling = nn.AvgPool2d(kernel_size=self.video_spatial_pooling_size, stride=self.video_spatial_pooling_size)
        if self.video_temporal_pooling_size >= 2:
            self.video_temporal_pooling = nn.AvgPool1d(kernel_size=self.video_temporal_pooling_size, stride=self.video_temporal_pooling_size)

        # #
        self.action_head = action_head
        self.all_module_keys.append("action_head")

        # #
        self.loss_text_weight   = 1.0
        self.loss_action_weight = 2.0

        # register_buffer #
        # text_vocab_mask : (1, 1, combined_vocab_size) #
        mask = torch.zeros(self.vocab_size + self.action_head.vocab_size, dtype=torch.bool)
        mask[:self.vocab_size] = True
        mask = mask.unsqueeze(0).unsqueeze(0)
        self.register_buffer("text_vocab_mask", mask)
        # action_vocab_mask : (1, 1, combined_vocab_size) #
        mask = torch.zeros(self.vocab_size + self.action_head.vocab_size, dtype=torch.bool)
        mask[self.vocab_size:] = True
        mask = mask.unsqueeze(0).unsqueeze(0)        
        self.register_buffer("action_vocab_mask", mask)

        # # unmask_dict #
        # self.unmask_dict = self.action_head.unmask_dict

        # # action_input_mapping / action_input_mapping_lut #
        # self.action_input_mapping = self.action_head.action_input_mapping
        # self.action_input_mapping_lut = torch.full(100000, -1, device=self.device, dtype=torch.long)
        # for k, v in self.action_input_mapping.items(): self.action_input_mapping_lut[k] = v

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        projector: ProjectorBackbone,
        action_head: ActionHeadBackbone,
        enable_mixed_precision_training: bool = True,
    ) -> PrismaticPaliGemmaCausalVLA:

        """Initialize a PrismaticPaliGemmaCausalVLA from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vla = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            projector,
            action_head,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

         # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        if "vision_backbone" in model_state_dict.keys():
            vla.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])
        if "llm_backbone" in model_state_dict.keys():
            vla.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "projector" in model_state_dict.keys():
            vla.projector.load_state_dict(model_state_dict["projector"])
        if "action_head" in model_state_dict.keys():
            vla.action_head.load_state_dict(model_state_dict["action_head"])

        # Freeze Weights
        vla.requires_grad_(False)
        vla.eval()

        return vla

    def freeze_backbones(self, stage: str) -> None:

        if stage == "finetune":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.action_head.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "llm_backbone", "action_head"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.projector.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Action Head `{self.action_head.identifier}`", ctx_level=1)

        elif stage == "full-finetune":
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)
            self.action_head.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone", "action_head"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.projector.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Action Head `{self.action_head.identifier}`", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < finetune | full-finetune >")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()
        # projector_fsdp_wrapping_policy = self.projector.get_fsdp_wrapping_policy()
        # action_head_fsdp_wrapping_policy = self.action_head.get_fsdp_wrapping_policy()

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                # projector_fsdp_wrapping_policy,
                # action_head_fsdp_wrapping_policy,
            ],
        )

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids=None,
        past_key_values=None,
        cache_position=None,
        input_tensor=None,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            # if attention_mask is not None and 0.0 in attention_mask:
            if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2 and (attention_mask == 0).any():
                return attention_mask
            return None

        min_dtype = torch.finfo(self.dtype).min
        if input_tensor is None:
            input_tensor = attention_mask

        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if   isinstance(past_key_values, StaticCache):
            target_length = past_key_values.get_max_cache_shape()
        elif isinstance(past_key_values, HybridCache):
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else cache_position[0] + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            return attention_mask

        causal_mask = torch.full(
            (sequence_length, target_length), fill_value=min_dtype, dtype=self.dtype, device=cache_position.device
        )
        # Causal diagonal mask only if training, otherwise attend to the whole prefix. Training-specific attn for prefix is handled below
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)

        # Apply bidirectional mask on images if token type ids are provided
        if token_type_ids is not None and sequence_length != 1:
            token_type_mask = token_type_ids.unsqueeze(1) == token_type_ids.unsqueeze(2)
            # token_type_mask[token_type_ids ==  2] = False  # if text token do not change anything (For Causal-Action)
            # token_type_mask[token_type_ids ==  1] = False  # if text token do not change anything
            # token_type_mask[token_type_ids == -1] = False  # if text token do not change anything
            token_type_mask[token_type_ids != 0] = False  # if text token do not change anything
            token_type_mask = token_type_mask.unsqueeze(1).to(causal_mask.device, dtype=torch.bool)
            causal_mask = causal_mask.clone()
            causal_mask[:, :, :, :sequence_length] = causal_mask[:, :, :, :sequence_length].masked_fill(
                token_type_mask, 0.0
            )

        # # === NEW: For each row, based on the dict mapping token_type_id → (k, o) (unmask_dict),
        # #          forbid the range of "k consecutive positions offset by o from the current position". ===
        # #
        # # Training time (sequence_length > 1):
        # #   For each row q, use token_type_ids[:,  q] → k, and forbid the range [cur_q - o - k, cur_q - o).
        # #
        # # Generation time (sequence_length == 1):
        # #   For each batch, use token_type_ids[:, -1] → k, and forbid the range [cur   - o - k, cur   - o).
        # #
        # unmask_dict = getattr(self, "unmask_dict", None)
        # if token_type_ids is not None and unmask_dict is not None:
        #     #
        #     B = inputs_lead_dim
        #     Tq = sequence_length
        #     Tk = target_length
        #     device = causal_mask.device

        #     #
        #     tt_slice = token_type_ids[:, :Tq].to(torch.long).to(device)  # [B, Tq]
        #     k_per_query = torch.zeros((B, Tq), dtype=torch.long, device=device)
        #     o_per_query = torch.zeros((B, Tq), dtype=torch.long, device=device)
        #     for ttype in tt_slice.unique().tolist():
        #         kval, oval = unmask_dict.get(int(ttype))
        #         kval = int(kval)
        #         oval = int(oval)
        #         if kval <= 0: continue
        #         mask_t = (tt_slice == int(ttype))
        #         if mask_t.any():
        #             k_per_query = torch.where(mask_t, torch.tensor(kval, device=device, dtype=torch.long), k_per_query)
        #             o_per_query = torch.where(mask_t, torch.tensor(oval, device=device, dtype=torch.long), o_per_query)
        #     #
        #     k_per_query = k_per_query.clamp_min(0).clamp_max(Tk)
        #     o_per_query = o_per_query.clamp_min(0).clamp_max(Tk)

        #     #
        #     if (k_per_query > 0).any():
        #         # 
        #         cur_row = cache_position.to(torch.long).to(device)  # [Tq]

        #         # unmask each row in [left, right] = [cur - o - k, cur - o)
        #         cur = cur_row.view(1, 1, Tq, 1)          # [1,1,Tq,1]
        #         k   = k_per_query.view(B, 1, Tq, 1)      # [B,1,Tq,1]
        #         o   = o_per_query.view(B, 1, Tq, 1)      # [B,1,Tq,1]
        #         left  = (cur - o - k).clamp_min(0)       # [B,1,Tq,1]
        #         right = (cur - o    )                    # [1,1,Tq,1]  ※clampしない（cur==0で自動的に空集合になる）

        #         key_idx = torch.arange(Tk, device=device).view(1, 1, 1, Tk)  # [1,1, 1,Tk]
        #         ban_recent = (key_idx >= left) & (key_idx < right)           # [B,1,Tq,Tk] bool
        #         causal_mask = causal_mask.clone().masked_fill(ban_recent, min_dtype)

        #         # # unmask each row in [left, right] = [cur - o - k, cur - o) (without broadcast)
        #         # cm_view = causal_mask  # [B,1,Tq,Tk]
        #         # for b in range(B):
        #         #     for q in (k_per_query[b] > 0).nonzero(as_tuple=False).flatten().tolist():
        #         #         k = int(k_per_query[b, q].item())
        #         #         o = int(o_per_query[b, q].item())
        #         #         cur = int(cur_row[q].item())
        #         #         left, right = cur - o - k, cur - o
        #         #         left = max(0, left)
        #         #         if right >= 0 and left < right:
        #         #             cm_view[b, 0, q, left:right] = min_dtype

        # Apply attention mask if attention_mask are provided
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]

            # Then apply padding mask (will mask pad tokens)
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        # print(f"token_type_ids(l) : {token_type_ids[0,-30:]}")
        # print(f"causal_mask(l) : {causal_mask[0, 0,-30:,-30:] > -1.0}")
        # print(f"token_type_ids(m) : {token_type_ids[0,-60:-30]}")
        # print(f"causal_mask(m) : {causal_mask[0, 0,-60:-30,-60:-30] > -1.0}")
        return causal_mask

    def get_video_features(self, video_pixel_values: torch.FloatTensor):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            video_pixel_values (`torch.FloatTensor]` of shape `(batch_size, video_length, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            video_features (`torch.Tensor`): Image feature tensor of shape `(batch_size, image_seq_length * ( video_length // video_length_factor ), embed_dim)`).
        """
        video_b, video_f, video_c, video_h, video_w = video_pixel_values.shape
        video_pixel_values = rearrange(video_pixel_values, "b f c h w -> (b f) c h w")
        video_outputs = self.vision_backbone(video_pixel_values)
        selected_video_feature = video_outputs.last_hidden_state

        if self.video_spatial_pooling_size >= 2:
            patch_num = selected_video_feature.shape[-2]
            ph, pw = int(np.sqrt(patch_num)), int(np.sqrt(patch_num))
            selected_video_feature = rearrange(selected_video_feature, "bf ( ph pw ) c -> bf c ph pw", ph=ph, pw=pw)   
            selected_video_feature = self.video_spatial_pooling(selected_video_feature)
            selected_video_feature = rearrange(selected_video_feature, "bf c ph pw -> bf ( ph pw ) c")   

        if self.video_temporal_pooling_size >= 2:
            patch_num = selected_video_feature.shape[-2]
            selected_video_feature = rearrange(selected_video_feature, "( b f ) p c -> b ( p c ) f", f=video_f)   
            selected_video_feature = self.video_temporal_pooling(selected_video_feature)
            selected_video_feature = rearrange(selected_video_feature, " b ( p c ) f -> b ( f p ) c", p=patch_num)   
        else:
            selected_video_feature = rearrange(selected_video_feature, "( b f ) p c -> b ( f p ) c", f=video_f)

        video_features = self.projector(selected_video_feature)
        video_features = video_features / (self.config.hidden_size**0.5)
        return video_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        video_pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_weight: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        multimodal_indices: Optional[torch.LongTensor] = None,
        **lm_kwargs,
    ) -> Union[Tuple, PrismaticPaliGemmaVLAOutputWithPast]:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )
        output_hidden_states = True
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        is_training = token_type_ids is not None and labels is not None

        # Replace image id woth PAD if the image token if OOV, to avoid index-errors
        if input_ids is not None and self.config.image_token_index >= self.vocab_size:
            special_image_mask = input_ids == self.config.image_token_index
            llm_input_ids = input_ids.clone()
            llm_input_ids[special_image_mask] = 0
        else:
            llm_input_ids = input_ids

        if inputs_embeds is None:

            # input_ids_mask : (batch_size, seq_length) #
            input_ids_mask = ( token_type_ids >= 2 ).to(llm_input_ids.device)

            # set text_input_ids / action_input_ids : (batch_size, seq_length) #
            text_input_ids   = llm_input_ids.masked_fill(  input_ids_mask, self.pad_token_id )
            action_input_ids = llm_input_ids.masked_fill( ~input_ids_mask, self.pad_token_id )
            # action_input_ids[ input_ids_mask ] = action_input_ids[ input_ids_mask ] - self.vocab_size
            action_input_ids[ input_ids_mask & ( action_input_ids != self.pad_token_id ) ] = action_input_ids[ input_ids_mask & ( action_input_ids != self.pad_token_id ) ] - self.vocab_size

            # # refresh action_input_ids #
            # #
            # action_input_mask = torch.zeros_like(token_type_ids, dtype=torch.bool)
            # for k in self.action_input_mapping.keys():
            #     action_input_mask |= (token_type_ids == k)
            # #
            # action_input_mapping_rows, action_input_mapping_cols = action_input_mask.nonzero(as_tuple=True)
            # action_input_mapping_values = self.action_input_mapping_lut[token_type_ids][action_input_mapping_rows, action_input_mapping_cols]
            # #
            # action_input_ids.index_put_((action_input_mapping_rows, action_input_mapping_cols), action_input_mapping_values)   # accumulate=False (default)   

            # get text_inputs_embeds / action_inputs_embeds : (batch_size, seq_length, hidden_size) #
            text_inputs_embeds   = self.get_input_embeddings()(text_input_ids)     
            action_inputs_embeds = self.action_head.get_input_embeddings()(action_input_ids)

            # inputs_embeds : (batch_size, seq_length, hidden_size) #
            inputs_embeds = text_inputs_embeds + action_inputs_embeds

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0) + 1  # Paligemma positions are 1-indexed

        # Merge text and images
        if pixel_values is not None:
            image_features = self.get_image_features(pixel_values)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
                )
            else:
                special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {image_features.shape[0] * image_features.shape[1]} "
                    "tokens from image embeddings."
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        # Merge text and video
        elif video_pixel_values is not None:
            video_features = self.get_video_features(video_pixel_values)

            if input_ids is None:
                special_image_mask = inputs_embeds == self.get_input_embeddings()(
                    torch.tensor(self.config.image_token_index, dtype=torch.long, device=inputs_embeds.device)
                )
            else:
                special_image_mask = (input_ids == self.config.image_token_index).unsqueeze(-1)
                special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)

            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != video_features.numel():
                image_tokens_in_text = (special_image_mask).sum(dim=1).sum(dim=0)[0]
                raise ValueError(
                    f"Number of images does not match number of special image tokens in the input text. "
                    f"Got {image_tokens_in_text} image tokens in the text but {video_features.shape[0] * video_features.shape[1]} "
                    "tokens from image embeddings."
                )
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, video_features)

        # mask out pad-token-ids in labels for BC
        if labels is not None and self.pad_token_id in labels:
            logger.warning_once(
                "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. "
                "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
            )
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

        # Augmentation #
        if is_training and ( pixel_values is not None or video_pixel_values is not None ):
        # if False:
            # set aug_seq_length #
            if pixel_values is not None:
                aug_seq_length = (special_image_mask[:,:,0]).sum(dim=1).min()
            elif video_pixel_values is not None:
                aug_seq_length = (special_image_mask[:,:,0]).sum(dim=1).min()
            # print(f"aug_seq_length : {aug_seq_length} / {self.llm_backbone.tokenizer.image_seq_length}")

            # set aug_indices #
            aug_rate = np.random.uniform(low=0.40, high=0.60)
            aug_size = int(aug_seq_length * aug_rate)
            aug_indices_image = torch.randperm(aug_seq_length, device=inputs_embeds.device)[:aug_size].sort().values
            aug_indices_text  = torch.arange(aug_seq_length, inputs_embeds.shape[1], device=inputs_embeds.device)
            aug_indices = torch.cat([aug_indices_image, aug_indices_text], dim=0)

            # Augment Inputs (wo cache_position) #
            attention_mask = attention_mask[:, aug_indices].contiguous()
            token_type_ids = token_type_ids[:, aug_indices].contiguous()
            inputs_embeds  = inputs_embeds [:, aug_indices].contiguous()
            labels         = labels        [:, aug_indices].contiguous()
            # cache_position = cache_position[   aug_indices].contiguous()
            position_ids   = position_ids  [:, aug_indices].contiguous()

            # ReCalc cache_position #
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # #
        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds
        )
        outputs: CausalLMOutputWithPast = self.llm_backbone.llm(
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **lm_kwargs,
        )

        # #
        if True:
            # combined_logits : (batch_size, seq_length, combined_vocab_size=(vocab_size + action_vocab_size) ) #
            text_logit      = outputs.logits
            action_logits   = self.action_head(outputs.hidden_states[-1][:, -outputs.logits.shape[1]:])
            combined_logits = torch.cat([text_logit, action_logits], dim=2)

            # base_logits_mask : (batch_size, seq_length, 1) #
            base_logits_mask = ( token_type_ids[:, -outputs.logits.shape[1]:] >= 2 ).unsqueeze(-1) # ( token_type_ids >= 2 ) is action_token

            # logits_mask : (batch_size, seq_length, combined_vocab_size) #
            logits_mask = torch.where(
                base_logits_mask,
                self.action_vocab_mask.broadcast_to(combined_logits.shape),
                self.text_vocab_mask.broadcast_to(combined_logits.shape),
                )

            # apply logits_mask to combined_logits #
            combined_logits.masked_fill_(~logits_mask, float('-inf'))

            # refresh outputs.logits #
            outputs.logits = combined_logits

        # loss_text #
        loss_list = []
        if labels is not None:

            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = outputs.logits.float()

            # input_ids_mask : (batch_size, seq_length - 1) #
            input_ids_mask = ( token_type_ids >= 2 )

            # text_mask, action_mask #
            if attention_mask is not None:
                text_mask     = ~input_ids_mask.bool() & attention_mask.bool()
                action_mask   =  input_ids_mask.bool() & attention_mask.bool()
            else:
                text_mask     = ~input_ids_mask.bool()
                action_mask   =  input_ids_mask.bool()

            # #
            if labels_weight is not None:
                # print(f"labels_weight : {labels_weight}")
                labels_weight = labels_weight.unsqueeze(1).broadcast_to(labels.shape)

            # #
            loss_text_fct = nn.CrossEntropyLoss(reduction='none')
            if text_mask.any():
                # shift_text_logits : (N, vocab_size),  shift_text_labels : (N,) #
                shift_text_logits = logits[text_mask.to(logits.device)][ :-1, :self.vocab_size].contiguous()
                shift_text_labels = labels[text_mask.to(labels.device)][1:                    ].contiguous()
                if labels_weight is not None:
                    shift_text_labels_weight = labels_weight[text_mask.to(labels_weight.device)][1:                    ].contiguous()

                # #
                flat_text_logits = shift_text_logits.view(-1, self.vocab_size)
                flat_text_labels = shift_text_labels.view(-1).to(shift_text_logits.device)
                if labels_weight is not None:
                    flat_text_labels_weight = shift_text_labels_weight.view(-1).to(shift_text_logits.device)
                loss_text_full = loss_text_fct(flat_text_logits, flat_text_labels)

                # #
                ignore_classes = [IGNORE_INDEX]
                mask = ~torch.isin(flat_text_labels, torch.tensor(ignore_classes).to(self.device))
                if mask.any():
                    if False:#labels_weight is not None:
                        loss_text = ( loss_text_full[mask] * flat_text_labels_weight[mask] ).mean() * self.loss_text_weight
                    else:
                        loss_text = loss_text_full[mask].mean() * self.loss_text_weight
                else:
                    loss_text = ( torch.nan_to_num(loss_text_full, nan=0.0) * 0.0 ).mean()
            else:
                # #
                dummy_text_logits = logits[..., :self.vocab_size].contiguous()
                # dummy_text_labels = torch.zeros(dummy_text_logits.size(0), dtype=torch.long, device=dummy_text_logits.device)
                dummy_text_labels = torch.zeros(dummy_text_logits.shape[:2], dtype=torch.long, device=dummy_text_logits.device)

                # #
                flat_text_logits = dummy_text_logits.view(-1, self.vocab_size)
                flat_text_labels = dummy_text_labels.view(-1).to(dummy_text_logits.device)
                loss_text_full = loss_text_fct(flat_text_logits, flat_text_labels)

                # #
                loss_text = ( torch.nan_to_num(loss_text_full, nan=0.0) * 0.0 ).mean()
            #
            loss_list.append(loss_text)

            # #
            loss_action_fct = nn.CrossEntropyLoss(weight=self.action_head.action_weight.to(self.device), reduction='none')
            if action_mask.any():
                # shift_action_logits : (N, action_vocab_size),  shift_action_labels : (N,) #
                shift_action_logits = logits[action_mask.to(logits.device)][ :-1, self.vocab_size:].contiguous()
                shift_action_labels = labels[action_mask.to(labels.device)][1:                    ].contiguous()
                # shift_action_labels = shift_action_labels - self.vocab_size
                shift_action_labels[ shift_action_labels != IGNORE_INDEX ] = shift_action_labels[ shift_action_labels != IGNORE_INDEX ] - self.vocab_size
                if labels_weight is not None:
                    shift_action_labels_weight = labels_weight[action_mask.to(labels_weight.device)][1:                    ].contiguous()

                # flat_action_logits : (N, action_vocab_size), flat_action_labels : (N,) #
                flat_action_logits = shift_action_logits.view(-1, self.action_head.vocab_size)
                flat_action_labels = shift_action_labels.view(-1).to(shift_action_logits.device)
                if labels_weight is not None:
                    # #
                    flat_action_labels_weight = shift_action_labels_weight.view(-1).to(shift_action_logits.device)      

                # #
                loss_action_full = loss_action_fct(flat_action_logits, flat_action_labels)

                # #
                ignore_classes = self.action_head.action_ignore_classes + [IGNORE_INDEX]
                mask = ~torch.isin(flat_action_labels, torch.tensor(ignore_classes).to(self.device))
                if mask.any():
                    if labels_weight is not None:
                        # #
                        flat_action_labels_weight = torch.abs(flat_action_labels_weight)
                        loss_action = ( loss_action_full[mask] * flat_action_labels_weight[mask] ).mean() * self.loss_action_weight

                        # # #
                        # if ( labels_weight.mean(dim=1) < 0.0 ).all():
                        #     lambda_pen = 1.0 #1e-2
                        #     loss_action = loss_action + ( lambda_pen * action_logits_pen )
                    else:
                        loss_action = loss_action_full[mask].mean() * self.loss_action_weight
                else:
                    loss_action = ( torch.nan_to_num(loss_action_full, nan=0.0) * 0.0 ).mean()
            else:
                # #
                dummy_action_logits = logits[..., self.vocab_size:].contiguous()
                # dummy_action_labels = torch.zeros(dummy_action_logits.size(0), dtype=torch.long, device=dummy_action_logits.device)
                dummy_action_labels = torch.zeros(dummy_action_logits.shape[:2], dtype=torch.long, device=dummy_action_logits.device)

                # #
                flat_action_logits = dummy_action_logits.view(-1, self.action_head.vocab_size)
                flat_action_labels = dummy_action_labels.view(-1).to(dummy_action_logits.device)
                loss_action_full = loss_action_fct(flat_action_logits, flat_action_labels)

                # #
                loss_action = ( torch.nan_to_num(loss_action_full, nan=0.0) * 0.0 ).mean()
            #
            loss_list.append(loss_action)

        # #
        loss = None
        for loss_idx in range(len(loss_list)):
            loss = loss + loss_list[ loss_idx ] if loss is not None else loss_list[ loss_idx ]
            loss_list[ loss_idx ] = loss_list[ loss_idx ].detach().clone().cpu()

        # #
        output = PrismaticPaliGemmaVLAOutputWithPast(
            loss=loss,
            loss_list=loss_list,
            logits=outputs.logits,
            action_logits_list=None,#action_logits_list,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output if return_dict else output.to_tuple()

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
        video_pixel_values=None,
        attention_mask=None,
        token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        **kwargs,
    ):
        # Overwritten -- custom `position_ids` and `pixel_values` handling
        model_inputs = self.llm_backbone.llm.prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cache_position=cache_position,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            token_type_ids=token_type_ids,
            **kwargs,
        )

        # #
        #
        def _current_len_from_model_inputs(mi):
            if mi.get("inputs_embeds") is not None:
                return mi["inputs_embeds"].shape[1]
            for k in ("input_ids", "decoder_input_ids"):
                if mi.get(k) is not None:
                    return mi[k].shape[1]
            return input_ids.shape[1]
        #        
        device = input_ids.device
        B = input_ids.size(0)
        L = _current_len_from_model_inputs(model_inputs)

        # position_ids in Paligemma are 1-indexed
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1

        # # generate token_type_ids
        # if model_inputs.get("token_type_ids") is not None:
        #     # print(f"L : {L} <--> {token_type_ids.shape[1]}")

        #     # if tti.shape[1] < L:
        #     #     add_tokens = L - tti.shape[1]
        #     #     next_type_provider = getattr(self, "next_type_provider", None)
        #     #     # next_type_provider : 
        #     #     if callable(next_type_provider):
        #     #         q_type = next_type_provider(token_type_ids).to(device=device, dtype=torch.long)
        #     #     # default : copy previous value
        #     #     else:
        #     #         q_type = tti[:, -1] #if tti.shape[1] > 0 else torch.zeros((B,), dtype=torch.long, device=device)
        #     #     q_type = q_type.view(B, 1).expand(B, add_tokens)
        #     #     tti = torch.cat([tti, q_type], dim=1)
        #     # elif tti.shape[1] > L:
        #     #     tti = tti[:, -L:].contiguous()

        #     # next_type_provider : 
        #     if L == 1:
        #         q_type = self.next_type_provider(token_type_ids).to(device=device, dtype=torch.long)
        #         model_inputs["token_type_ids"] = q_type.unsqueeze(1)
        #         token_type_ids[:,-1] = q_type

        #     # print(f'input_ids : {input_ids.shape} : {model_inputs["input_ids"].shape} : {input_ids[:,-20:]} : {model_inputs["input_ids"]}')
        #     # print(f'token_type_ids : {token_type_ids.shape} : {model_inputs["token_type_ids"].shape} : {token_type_ids[:,-20:]} : {model_inputs["token_type_ids"]}')

        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
            model_inputs["video_pixel_values"] = video_pixel_values

        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
            causal_mask = self._update_causal_mask(
                attention_mask, token_type_ids, past_key_values, cache_position, input_tensor
            )
            model_inputs["attention_mask"] = causal_mask

        return model_inputs


    def next_type_provider(
        self,
        token_type_ids: torch.Tensor,
        keep_threshold: int = 1,         # <= keep this value
        inc: int = 1,                    # otherwise increment by +inc
        default_type: int = 0,           # fallback when T == 0
        max_type: Optional[int] = None,  # e.g., 10000; if None, allow unbounded growth
    ) -> torch.Tensor:
        """
        token_type_ids: Expected to be a LongTensor of shape [B, T].
        Returns: A LongTensor of shape [B], representing the next token's type for each batch.
        Spec: Refers to the last column's type; values <= keep_threshold are kept,
            others are increased by +inc.
        """
        if token_type_ids.dim() != 2:
            raise ValueError(f"token_type_ids must be 2D [B, T], got {token_type_ids.shape}")

        B, T = token_type_ids.shape
        device = token_type_ids.device

        if T == 0:
            # If the sequence length is zero (e.g., first step), return the fallback value.
            return torch.full((B,), default_type, dtype=torch.long, device=device)

        last = token_type_ids[:, -1].to(dtype=torch.long)  # [B]
        # Keep values <= keep_threshold; otherwise increment by +inc.
        out = torch.where(last <= keep_threshold, last, last + inc)

        if max_type is not None:
            out = torch.clamp(out, max=max_type)
            # Or if wrap-around behavior is preferred:
            # out = keep_threshold + 1 + ((out - (keep_threshold + 1)) % (max_type - keep_threshold))

        # #
        return out

