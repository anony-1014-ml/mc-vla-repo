"""
materialize.py

Factory class for initializing Vision Backbones, LLM Backbones, and VLMs from a set registry; provides and exports
individual functions for clear control flow.
"""
import os
from pathlib import Path
PRISMATIC_MODEL_DIR = os.environ.get('PRISMATIC_MODEL_DIR', "../checkpoints")
print(f"PRISMATIC_MODEL_DIR : {PRISMATIC_MODEL_DIR}")

from typing import Optional, Tuple

from transformers import PreTrainedTokenizerBase

from prismatic.models.backbones.llm import LLMBackbone, LLaMa2LLMBackbone, MistralLLMBackbone, PhiLLMBackbone, PaliGemmaTextBackbone
from prismatic.models.backbones.vision import (
    VisionBackbone,
    CLIPViTBackbone,
    DinoCLIPViTBackbone,
    DinoSigLIPViTBackbone,
    DinoV2ViTBackbone,
    ImageTransform,
    IN1KViTBackbone,
    SigLIPViTBackbone,
    PaliGemmaVisionBackbone,
)
from prismatic.models.vlms import ProjectorBackbone, PaliGemmaProjectorBackbone
from prismatic.models.vlms import PrismaticVLM, PrismaticPaliGemmaVLM
from prismatic.models.vlas import ActionHeadBackbone, PaliGemmaCausalActionHeadBackbone
from prismatic.models.vlas import PrismaticPaliGemmaCausalVLA
from prismatic.action import ActionProc

# === Registries =>> Maps ID --> {cls(), kwargs} :: Different Registries for Vision Backbones, LLM Backbones, VLMs ===
# fmt: off

# === Vision Backbone Registry ===
VISION_BACKBONES = {
    # === 224px Backbones ===
    "clip-vit-l": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-so400m": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "dinov2-vit-l": {"cls": DinoV2ViTBackbone, "kwargs": {"default_image_size": 224}},
    "in1k-vit-l": {"cls": IN1KViTBackbone, "kwargs": {"default_image_size": 224}},
    "dinosiglip-vit-so-224px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 224}},

    # === Assorted CLIP Backbones ===
    "clip-vit-b": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "clip-vit-l-336px": {"cls": CLIPViTBackbone, "kwargs": {"default_image_size": 336}},

    # === Assorted SigLIP Backbones ===
    "siglip-vit-b16-224px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 224}},
    "siglip-vit-b16-256px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 256}},
    "siglip-vit-b16-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},
    "siglip-vit-so400m-384px": {"cls": SigLIPViTBackbone, "kwargs": {"default_image_size": 384}},

    # === Fused Backbones ===
    "dinoclip-vit-l-336px": {"cls": DinoCLIPViTBackbone, "kwargs": {"default_image_size": 336}},
    "dinosiglip-vit-so-384px": {"cls": DinoSigLIPViTBackbone, "kwargs": {"default_image_size": 384}},

    # === paligemma vision Backbones ===
    "paligemma-3b-pt-224":     {"cls": PaliGemmaVisionBackbone, "kwargs": {"default_image_size": 224, "hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma-3b-pt-224" )}},
    "paligemma-3b-mix-224":    {"cls": PaliGemmaVisionBackbone, "kwargs": {"default_image_size": 224, "hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma-3b-mix-224" )}},
    "paligemma-3b-ft-gqa-224": {"cls": PaliGemmaVisionBackbone, "kwargs": {"default_image_size": 224, "hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma-3b-ft-gqa-224" )}},

    # === paligemma2 vision Backbones ===
    "paligemma2-3b-pt-224":     {"cls": PaliGemmaVisionBackbone, "kwargs": {"default_image_size": 224, "hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma2-3b-pt-224" )}},
    "paligemma2-3b-mix-224":    {"cls": PaliGemmaVisionBackbone, "kwargs": {"default_image_size": 224, "hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma2-3b-mix-224" )}},
}


# === Language Model Registry ===
LLM_BACKBONES = {
    # === LLaMa-2 Pure (Non-Chat) Backbones ===
    "llama2-7b-pure": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "llama2-13b-pure": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === LLaMa-2 Chat Backbones ===
    "llama2-7b-chat": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "llama2-13b-chat": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === Vicuna-v1.5 Backbones ===
    "vicuna-v15-7b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},
    "vicuna-v15-13b": {"cls": LLaMa2LLMBackbone, "kwargs": {}},

    # === Mistral v0.1 Backbones ===
    "mistral-v0.1-7b-pure": {"cls": MistralLLMBackbone, "kwargs": {}},
    "mistral-v0.1-7b-instruct": {"cls": MistralLLMBackbone, "kwargs": {}},

    # === Phi-2 Backbone ===
    "phi-2-3b": {"cls": PhiLLMBackbone, "kwargs": {}},

    # === paligemma vision Backbones ===
    "paligemma-3b-pt-224":     {"cls": PaliGemmaTextBackbone, "kwargs": {"hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma-3b-pt-224" )}},
    "paligemma-3b-mix-224":    {"cls": PaliGemmaTextBackbone, "kwargs": {"hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma-3b-mix-224" )}},
    "paligemma-3b-ft-gqa-224": {"cls": PaliGemmaTextBackbone, "kwargs": {"hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma-3b-ft-gqa-224" )}},

    # === paligemma2 vision Backbones ===
    "paligemma2-3b-pt-224":     {"cls": PaliGemmaTextBackbone, "kwargs": {"hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma2-3b-pt-224" )}},
    "paligemma2-3b-mix-224":    {"cls": PaliGemmaTextBackbone, "kwargs": {"hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma2-3b-mix-224" )}},
}

# === Projector Model Registry ===
PROJECTORS = {
    # === paligemma vision Backbones ===
    "paligemma-3b-pt-224":     {"cls": PaliGemmaProjectorBackbone, "kwargs": {"hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma-3b-pt-224" )}},
    "paligemma-3b-mix-224":    {"cls": PaliGemmaProjectorBackbone, "kwargs": {"hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma-3b-mix-224" )}},
    "paligemma-3b-ft-gqa-224": {"cls": PaliGemmaProjectorBackbone, "kwargs": {"hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma-3b-ft-gqa-224" )}},

    # === paligemma2 vision Backbones ===
    "paligemma2-3b-pt-224":     {"cls": PaliGemmaProjectorBackbone, "kwargs": {"hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma2-3b-pt-224" )}},
    "paligemma2-3b-mix-224":    {"cls": PaliGemmaProjectorBackbone, "kwargs": {"hf_hub_path": os.path.join( PRISMATIC_MODEL_DIR, "paligemma2-3b-mix-224" )}},
}

# === ActionHead Model Registry ===
ACTION_HEADS = {
    # === causal ===
    "paligemma-vpt"             : {"cls": PaliGemmaCausalActionHeadBackbone, "kwargs": {"vocab_size": -(-(100+8641+121)//64)*64, "vocab_config": [100,8641,121], "hidden_size": 2048, "pad_token_id": 0}},
    "paligemma-vpt-cd"          : {"cls": PaliGemmaCausalActionHeadBackbone, "kwargs": {"vocab_size": -(-(100+8641+441)//64)*64, "vocab_config": [100,8641,441], "hidden_size": 2048, "pad_token_id": 0}},
    "paligemma-vpt-para"        : {"cls": PaliGemmaCausalActionHeadBackbone, "kwargs": {"vocab_size": -(-(100+4321+121)//64)*64, "vocab_config": [100,4321,121], "hidden_size": 2048, "pad_token_id": 0}},
    "paligemma-vpt-cd-para"     : {"cls": PaliGemmaCausalActionHeadBackbone, "kwargs": {"vocab_size": -(-(100+4321+441)//64)*64, "vocab_config": [100,4321,441], "hidden_size": 2048, "pad_token_id": 0}},
    "paligemma-vpt-paraxy"      : {"cls": PaliGemmaCausalActionHeadBackbone, "kwargs": {"vocab_size": -(-(100+4321+11+11)//64)*64, "vocab_config": [100,4321,22], "hidden_size": 2048, "pad_token_id": 0}},
    "paligemma-vpt-cd-paraxy"   : {"cls": PaliGemmaCausalActionHeadBackbone, "kwargs": {"vocab_size": -(-(100+4321+21+21)//64)*64, "vocab_config": [100,4321,42], "hidden_size": 2048, "pad_token_id": 0}},
}

# fmt: on


def get_vision_backbone_and_transform(
    vision_backbone_id: str, image_resize_strategy: str
) -> Tuple[VisionBackbone, ImageTransform]:
    """Instantiate a Vision Backbone, returning both the nn.Module wrapper class and default Image Transform."""
    if vision_backbone_id in VISION_BACKBONES:
        vision_cfg = VISION_BACKBONES[vision_backbone_id]
        vision_backbone: VisionBackbone = vision_cfg["cls"](
            vision_backbone_id, image_resize_strategy, **vision_cfg["kwargs"]
        )
        image_transform = vision_backbone.get_image_transform()
        return vision_backbone, image_transform

    else:
        raise ValueError(f"Vision Backbone `{vision_backbone_id}` is not supported!")


def get_llm_backbone_and_tokenizer(
    llm_backbone_id: str,
    llm_max_length: int = 2048,
    hf_token: Optional[str] = None,
    inference_mode: bool = False,
) -> Tuple[LLMBackbone, PreTrainedTokenizerBase]:
    if llm_backbone_id in LLM_BACKBONES:
        llm_cfg = LLM_BACKBONES[llm_backbone_id]
        llm_backbone: LLMBackbone = llm_cfg["cls"](
            llm_backbone_id,
            llm_max_length=llm_max_length,
            hf_token=hf_token,
            inference_mode=inference_mode,
            **llm_cfg["kwargs"],
        )
        tokenizer = llm_backbone.get_tokenizer()
        return llm_backbone, tokenizer

    else:
        raise ValueError(f"LLM Backbone `{llm_backbone_id}` is not supported!")


def get_projector(
    projector_id: str,
):
    if projector_id in PROJECTORS:
        projector_cfg = PROJECTORS[projector_id]
        projector = projector_cfg["cls"](
            projector_id, **projector_cfg["kwargs"]
        )
        return projector

    else:
        raise ValueError(f"Projector `{projector_id}` is not supported!")


def get_action_head(
    action_head_id: str,
):
    if action_head_id in ACTION_HEADS:
        action_head_cfg = ACTION_HEADS[action_head_id]
        action_head = action_head_cfg["cls"](
            action_head_id, **action_head_cfg["kwargs"]
        )
        return action_head

    else:
        raise ValueError(f"ActionHead `{action_head_id}` is not supported!")


def get_action_proc(
    action_head_id: str,
    temperature: float = 1.0,
    nucleus_prob: float = 0.99,
):
    # ActionProc #
    if   action_head_id == "paligemma-vpt":
        action_proc = ActionProc(method=0, temperature=temperature, nucleus_prob=nucleus_prob)
    elif action_head_id == "paligemma-vpt-cd":
        action_proc = ActionProc(method=1, temperature=temperature, nucleus_prob=nucleus_prob)
    elif action_head_id == "paligemma-vpt-para":
        action_proc = ActionProc(method=2, temperature=temperature, nucleus_prob=nucleus_prob)
    elif action_head_id == "paligemma-vpt-cd-para":
        action_proc = ActionProc(method=3, temperature=temperature, nucleus_prob=nucleus_prob)
    elif action_head_id == "paligemma-vpt-paraxy":
        action_proc = ActionProc(method=4, temperature=temperature, nucleus_prob=nucleus_prob)
    elif action_head_id == "paligemma-vpt-cd-paraxy":
        action_proc = ActionProc(method=5, temperature=temperature, nucleus_prob=nucleus_prob)    
    else:
        raise NotImplementedError(f"{action_head_id} is not implemented yet")

    # #
    return action_proc


def get_vlm(
    model_id: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    projector: ProjectorBackbone,
    arch_specifier: str,
    enable_mixed_precision_training: bool = True,
    model_family: str = "prismatic"
) -> PrismaticVLM:
    """Lightweight wrapper around initializing a VLM, mostly for future-proofing (if one wants to add a new VLM)."""
    if model_family == "prismaticPaliGemma":
        return PrismaticPaliGemmaVLM(
                model_id,
                vision_backbone,
                llm_backbone,
                projector,
                enable_mixed_precision_training=enable_mixed_precision_training,
        )
    elif model_family == "prismatic":
        return PrismaticVLM(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
        )


def get_vla(
    run_dir: Path,
    model_id: str,
    vision_backbone: VisionBackbone,
    llm_backbone: LLMBackbone,
    projector: ProjectorBackbone,
    action_head: ActionHeadBackbone,
    enable_mixed_precision_training: bool = True,
    model_family: str = "prismaticPaliGemma"
) -> PrismaticVLM:
    """Lightweight wrapper around initializing a VLM, mostly for future-proofing (if one wants to add a new VLM)."""
    if model_family == "prismaticPaliGemma":
        checkpoint_pt = run_dir / "checkpoints" / "latest-checkpoint.pt"
        if checkpoint_pt.exists():
            return PrismaticPaliGemmaCausalVLA.from_pretrained(
                checkpoint_pt,
                model_id,
                vision_backbone,
                llm_backbone,
                projector,
                action_head,
                enable_mixed_precision_training=enable_mixed_precision_training,
            )
        else:
            return PrismaticPaliGemmaCausalVLA(
                model_id,
                vision_backbone,
                llm_backbone,
                projector,
                action_head,
                enable_mixed_precision_training=enable_mixed_precision_training,
            )
    else:
        raise NotImplementedError("vla other than prismaticPaliGemma is not implemented yet")
