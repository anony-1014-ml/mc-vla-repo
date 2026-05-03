"""
prismatic.py

PyTorch Module defining a PrismaticPaliGemmaVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers import GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import is_torchdynamo_compiling
from transformers.cache_utils import Cache, HybridCache, StaticCache

from transformers import AutoModel, AutoConfig, AutoProcessor
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import ProjectorBackbone
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

from einops import rearrange

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100
IMAGE_TOKEN = "<image>"

#-------------------------------------------------------------------------------------------------------------
#
class PaliGemmaProjectorBackbone(ProjectorBackbone):
    def __init__(
        self,
        projector_id: str,
        hf_hub_path: str = None,
    ) -> None:
        # #
        super().__init__(projector_id)

        # #
        paligemma_conf = AutoConfig.from_pretrained(hf_hub_path)
        self.projector = PaliGemmaMultiModalProjector(paligemma_conf)
        #
        # if True:
        if False:
            paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(hf_hub_path)
            self.projector.load_state_dict(paligemma_model.multi_modal_projector.state_dict())
            del paligemma_model

    def forward(self, image_features):
        hidden_states = self.projector(image_features)
        return hidden_states

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a simple FSDP policy that wraps projector."""
        projector_wrap_policy = partial(_module_wrap_policy, module_classes={PaliGemmaProjectorBackbone})
        return projector_wrap_policy

    # @property
    # def module_cls(self) -> Type[nn.Module]:
    #     return PaliGemmaMultiModalProjector

#-------------------------------------------------------------------------------------------------------------
#
class PrismaticPaliGemmaVLM(nn.Module, GenerationMixin):

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
        enable_mixed_precision_training: bool = True,
    ) -> None:
        #---------------------------------------------------------------------
        # VLM
        #---------------------------------------------------------------------
        super().__init__()
        self.model_family, self.model_id = "prismatic", model_id
        self.vision_backbone, self.llm_backbone, self.projector = vision_backbone, llm_backbone, projector
        self.enable_mixed_precision_training = enable_mixed_precision_training

        # Instance Attributes for a generic VLM
        self.all_module_keys, self.trainable_module_keys = None, None

        # === GenerationMixin Expected Attributes =>> *DO NOT MODIFY* ===
        self.generation_config = self.llm_backbone.llm.generation_config
        self.main_input_name = "input_ids"

        #---------------------------------------------------------------------
        # PrismaticVLM
        #---------------------------------------------------------------------
        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)

        # Trackers
        self.vision_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector"]
        self.trainable_module_keys = []

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

        #---------------------------------------------------------------------
        # PrismaticPaliGemmaVLM
        #---------------------------------------------------------------------
        self.vocab_size = self.llm_backbone.llm.config.vocab_size
        self.pad_token_id = self.llm_backbone.llm.config.pad_token_id

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        projector: ProjectorBackbone,
        enable_mixed_precision_training: bool = True,
    ) -> PrismaticPaliGemmaVLM:
        """Initialize a PrismaticPaliGemmaVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            projector,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

         # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        if "vision_backbone" in model_state_dict.keys():
            vlm.vision_backbone.load_state_dict(model_state_dict["vision_backbone"])
        if "llm_backbone" in model_state_dict.keys():
            vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])
        if "projector" in model_state_dict.keys():
            vlm.projector.load_state_dict(model_state_dict["projector"])

        # Freeze Weights
        vlm.requires_grad_(False)
        vlm.eval()

        return vlm

    @property
    def device(self) -> torch.device:
        """Borrowed from `transformers.modeling_utils.py` -- checks parameter device; assumes model on *ONE* device!"""
        return next(self.parameters()).device

    # === GenerationMixin Expected Properties & Methods (DO NOT MODIFY) ===
    @staticmethod
    def can_generate() -> bool:
        return True

    @property
    def config(self) -> PretrainedConfig:
        return self.llm_backbone.llm.config

    @property
    def dtype(self):
        return self.llm_backbone.llm.dtype        

    # => Beam Search Utility
    def _reorder_cache(self, past_key_values, beam_idx):
        return self.llm_backbone.llm._reorder_cache(past_key_values, beam_idx)        

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" >
        """
        if stage == "align":
        # if stage == "align" or stage == "finetune":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.projector.identifier}`", ctx_level=1)

        elif stage == "finetune":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.projector.identifier}`", ctx_level=1)

        elif stage == "full-finetune":
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.projector.identifier}`", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()
        # projector_fsdp_wrapping_policy = self.projector.get_fsdp_wrapping_policy()

        # Return union (_or_) over constituent policies
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                # projector_fsdp_wrapping_policy,
            ],
        )

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings with Llava->PaliGemma
    def get_input_embeddings(self):
        return self.llm_backbone.llm.get_input_embeddings()

    def _update_causal_mask(
        self,
        attention_mask,
        token_type_ids=None,
        past_key_values=None,
        cache_position=None,
        input_tensor=None,
        is_training: Optional[bool] = None,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None
        is_training = is_training if is_training is not None else self.training
        using_static_cache = isinstance(past_key_values, StaticCache)
        min_dtype = torch.finfo(self.dtype).min
        if input_tensor is None:
            input_tensor = attention_mask

        inputs_lead_dim, sequence_length = input_tensor.shape[:2]
        if using_static_cache:
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
            if is_training:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            else:
                causal_mask[:, :sequence_length] = 0.0

        causal_mask *= torch.arange(target_length, device=cache_position.device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(inputs_lead_dim, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]

            # First unmask prefix tokens during training
            if is_training:
                if token_type_ids is None:
                    raise ValueError("Token type ids must be provided during training")
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    token_type_ids[:, None, None, :].to(causal_mask.device) == 0, 0
                )

            # Then apply padding mask (will mask pad tokens)
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )

        return causal_mask

    def get_image_features(self, pixel_values: torch.FloatTensor):
        """
        Obtains image last hidden states from the vision tower and apply multimodal projection.

        Args:
            pixel_values (`torch.FloatTensor]` of shape `(batch_size, channels, height, width)`)
               The tensors corresponding to the input images.
        Returns:
            image_features (`torch.Tensor`): Image feature tensor of shape `(batch_size, image_seq_length, embed_dim)`).
        """
        image_outputs = self.vision_backbone(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.projector(selected_image_feature)
        image_features = image_features / (self.config.hidden_size**0.5)
        return image_features

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[List[torch.FloatTensor], Cache]] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        multimodal_indices: Optional[torch.LongTensor] = None,
        **lm_kwargs,
    ) -> Union[Tuple, PaliGemmaCausalLMOutputWithPast]:

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
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
            inputs_embeds = self.get_input_embeddings()(llm_input_ids)

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

        # mask out pad-token-ids in labels for BC
        if labels is not None and self.pad_token_id in labels:
            logger.warning_once(
                "`labels` contains `pad_token_id` which will be masked with `config.ignore_index`. "
                "You have to mask out `pad_token_id` when preparing `labels`, this behavior will be removed in v.4.46.",
            )
            labels = torch.where(input_ids == self.pad_token_id, self.config.ignore_index, labels)

        # #
        causal_mask = self._update_causal_mask(
            attention_mask, token_type_ids, past_key_values, cache_position, inputs_embeds, is_training
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

        logits = outputs[0]
        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            shift_logits = logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            if attention_mask is not None:
                # we use the input attention mask to shift the logits and labels, because it is 2D.
                # we also crop attn mask in case it is longer, which happens in PrefixTuning with peft
                shift_attention_mask = attention_mask[:, -shift_logits.shape[1] :].to(logits.device)
                shift_logits = shift_logits[shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = shift_labels[shift_attention_mask.to(shift_labels.device) != 0].contiguous()
            else:
                shift_logits = shift_logits.contiguous()
                shift_labels = shift_labels.contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()

            flat_logits = shift_logits.view(-1, self.vocab_size)
            flat_labels = shift_labels.view(-1).to(shift_logits.device)
            loss = loss_fct(flat_logits, flat_labels)

            outputs.loss = loss

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        pixel_values=None,
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

        # position_ids in Paligemma are 1-indexed
        if model_inputs.get("position_ids") is not None:
            model_inputs["position_ids"] += 1
        # If we're in cached decoding stage, pixel values should be None because input ids do not contain special image token anymore
        # Otherwise we need pixel values to be passed to model. NOTE: use_cache=False needs pixel_values always
        if cache_position[0] == 0:
            model_inputs["pixel_values"] = pixel_values
        is_training = token_type_ids is not None and labels is not None
        if cache_position[0] == 0 and isinstance(past_key_values, HybridCache):
            input_tensor = inputs_embeds if inputs_embeds is not None else input_ids
            causal_mask = self._update_causal_mask(
                attention_mask, token_type_ids, past_key_values, cache_position, input_tensor, is_training
            )
            model_inputs["attention_mask"] = causal_mask

        return model_inputs    

    @torch.inference_mode()
    def generate_batch(
        self,
        batch_pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        batch_text: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        #
        inputs = tokenizer(
            [ text.replace(IMAGE_TOKEN, IMAGE_TOKEN * tokenizer.image_seq_length)  for text in batch_text],
            text_pair=None,
            return_token_type_ids=False,
            **tokenizer.tokenizer_kwargs,
            truncation=True, return_tensors="pt"
        )  
        batch_input_ids = inputs.input_ids.to(self.device)
        batch_attention_mask = inputs.attention_mask.to(self.device)
        #
        if isinstance(batch_pixel_values, torch.Tensor):
            # batch_pixel_values = batch_pixel_values[None, ...].to(self.device)
            batch_pixel_values = batch_pixel_values.to(self.device)
        else:
            raise ValueError(f"Unsupported `batch_pixel_values` type = {type(batch_pixel_values)}")

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx in range(len(batch_text)):
                #
                input_ids = batch_input_ids[idx].unsqueeze(0)
                attention_mask = batch_attention_mask[idx].unsqueeze(0)
                if isinstance(batch_pixel_values, torch.Tensor):
                    pixel_values = batch_pixel_values[idx].unsqueeze(0)
                else:
                    raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, attention_mask=attention_mask, **kwargs)
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]
                    gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                    # print(f"texts : {batch_text[idx]} -> gen_text : {gen_text}")

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(gen_text)
                else:
                    full_out_dict = super().generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    # @torch.inference_mode()
    # def generate(self, image: Image, prompt_text: str, **kwargs: str) -> str:
    #     # For now, only support generation with a batch size of 1 for simplicity
    #     image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

    #     # Prepare Inputs
    #     #
    #     input_ids = tokenizer(prompt_text.replace(IMAGE_TOKEN, IMAGE_TOKEN * tokenizer.image_seq_length), truncation=True, return_tensors="pt").input_ids.to(self.device)
    #     #
    #     pixel_values = image_transform(image)
    #     if isinstance(pixel_values, torch.Tensor):
    #         pixel_values = pixel_values[None, ...].to(self.device)
    #     elif isinstance(pixel_values, dict):
    #         pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
    #     else:
    #         raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

    #     # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
    #     autocast_dtype = self.llm_backbone.half_precision_dtype
    #     with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
    #         # fmt: off
    #         generated_ids = super().generate(
    #             input_ids=input_ids,            # Shape: [1, seq]
    #             pixel_values=pixel_values,      # Shape: [1, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
    #             **kwargs
    #         )
    #         # fmt: on

    #     generated_text = tokenizer.decode(generated_ids[0, input_ids.shape[1] :], skip_special_tokens=True).strip()

    #     return generated_text
