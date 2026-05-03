"""
fsdp.py

Core class definition for a strategy implementing Torch native Fully Sharded Data Parallel Training (with support for
fine-grained control over wrapping policies and mixed precision per component).
"""

import math
import shutil
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Callable, Optional
import gc
import re

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    fully_shard,
    MixedPrecisionPolicy,
    CPUOffloadPolicy,
    FSDPModule
)
from torch.optim import AdamW, SGD, ASGD, Adafactor
from prismatic.training.strategies.adafactorP import Adafactor as AdafactorP
from prismatic.training.strategies.adafactorT import Adafactor as AdafactorT
from prismatic.training.strategies.apollo import AdamW as APOLLOAdamW
from transformers.optimization import get_cosine_schedule_with_warmup

from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch
from prismatic.training.strategies.base_strategy import TrainingStrategy

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


class FSDP2Strategy(TrainingStrategy):
    def __init__(
        self,
        vlm: PrismaticVLM,
        device_id: int,
        epochs: int,
        max_steps: Optional[int],
        global_batch_size: int,
        per_device_batch_size: int,
        learning_rate: float,
        weight_decay: float,
        max_grad_norm: float,
        lr_scheduler_type: str,
        warmup_ratio: float,
        optimizer_type: str,
        checkpointing_accumulation_steps: Optional[int] = -1,
        enable_gradient_checkpointing: bool = True,
        enable_mixed_precision_training: bool = True,
        reduce_in_full_precision: bool = False,
        mixed_precision_dtype: torch.dtype = torch.bfloat16,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        sharding_strategy: str = "shard-grad-op",
    ) -> None:
        super().__init__(
            vlm=vlm,
            device_id=device_id,
            epochs=epochs,
            max_steps=max_steps,
            global_batch_size=global_batch_size,
            per_device_batch_size=per_device_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            lr_scheduler_type=lr_scheduler_type,
            warmup_ratio=warmup_ratio,
            optimizer_type=optimizer_type,
            checkpointing_accumulation_steps=checkpointing_accumulation_steps,
            enable_gradient_checkpointing=enable_gradient_checkpointing,
            enable_mixed_precision_training=enable_mixed_precision_training,
            reduce_in_full_precision=reduce_in_full_precision,
            mixed_precision_dtype=mixed_precision_dtype,
            worker_init_fn=worker_init_fn,
        )

        # FSDP-Specific Parameters
        if sharding_strategy == "shard-grad-op":
            self.reshard_after_forward = False
        elif sharding_strategy == "full-shard":
            self.reshard_after_forward = True
        else:
            raise ValueError(f"FSDP Sharding Strategy {sharding_strategy} is not supported!")

    def save_checkpoint(
        self,
        run_dir: Path,
        global_step: int,
        epoch: int,
        train_loss: Optional[float] = None,
        only_trainable: bool = True,
    ) -> None:
        """Save a checkpoint to the `run_dir` only containing the state_dicts for trainable parameters by default."""
        assert isinstance(self.vlm, FSDPModule), "FSDPStrategy.save_checkpoint assumes VLM is already wrapped in FSDP!"

        # #
        model_state_dicts = {
            mkey: OrderedDict() for mkey in (self.trainable_module_keys if only_trainable else self.all_module_keys)
        }
        sharded_sd = self.vlm.state_dict()
        for param_name, sharded_param in sharded_sd.items():
            if hasattr(sharded_param, "full_tensor"):
                full_param = sharded_param.full_tensor()
            else:
                full_param = sharded_param
            keep_flag = False
            if overwatch.is_rank_zero():
                for mkey in model_state_dicts:
                    if param_name.startswith(mprefix := f"{mkey}."):
                        model_state_dicts[mkey][param_name.removeprefix(mprefix)] = full_param.cpu()
                        keep_flag = True
                        break
            if not keep_flag:
                del full_param

        # Save on rank zero *only*
        if overwatch.is_rank_zero():
            checkpoint_dir = run_dir / "checkpoints"
            if train_loss is None:
                checkpoint_path = checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss=inf.pt"
            else:
                checkpoint_path = (
                    checkpoint_dir / f"step-{global_step:06d}-epoch-{epoch:02d}-loss={train_loss:.4f}.pt"
                )

            # Save Checkpoint & Copy Latest to `latest-checkpoint.pt`
            torch.save({"model": model_state_dicts}, checkpoint_path)
            shutil.copy(checkpoint_path, checkpoint_dir / "latest-checkpoint.pt")

        # #
        gc.collect()
        torch.cuda.empty_cache()

    def run_setup(self, run_dir: Path, n_train_examples: int) -> None:

        # Assemble the Default FSDP Mixed Precision Policy
        if self.enable_mixed_precision_training and self.mixed_precision_dtype == torch.bfloat16:
            # MixedPrecision `param_dtype` specifies *compute* dtype (for forward/backward only)
            reduce_buffer_dtype = torch.bfloat16 if not self.reduce_in_full_precision else torch.float32
            fsdp_precision_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=reduce_buffer_dtype,
            )
        else:
            # If we're not using mixed precision, everything is in default full precision!
            fsdp_precision_policy = MixedPrecisionPolicy(
                param_dtype=torch.float32,
                reduce_dtype=torch.float32,
            )

        # Gradient Checkpoint Setup
        if self.enable_gradient_checkpointing:
            # For Gradient Checkpointing under FSDP --> we make the same assumption as in the DDP/other strategies; the
            #   bulk of activation memory is taken up by the LLM activations. However, unlike other strategies, we
            #   cannot rely on the HF Transformers default `gradient_checkpointing_enable()` --> FSDP breaks semantics!
            #
            # Instead, we need to write our own *NO-REENTRANT* wrapper, and apply it to the LLM's Transformer Layer.
            non_reentrant_wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)

            def check_fn(submodule: nn.Module) -> bool:
                # return isinstance(submodule, self.llm_transformer_layer_cls)
                return isinstance(submodule, ( self.llm_transformer_layer_cls, self.vision_transformer_layer_cls ) )

            # Note that the terms "activation checkpointing" and "gradient checkpointing" are synonymous!
            apply_activation_checkpointing(self.vlm, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

        # <FSDP> => note that FSDP will automatically take care of device placement (similar to `autocast`)
        #
        fsdp_kwargs = {}
        fsdp_kwargs["reshard_after_forward"] = self.reshard_after_forward
        fsdp_kwargs["mp_policy"] = fsdp_precision_policy
        # fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()
        #
        self.vlm = self.vlm.to(self.device_id)#torch.cuda.current_device())
        #
        for module in self.vlm.modules():
            #
            if isinstance(module, self.llm_transformer_layer_cls):
                fully_shard(module, **fsdp_kwargs)
            #
            if isinstance(module, self.vision_transformer_layer_cls):
                fully_shard(module, **fsdp_kwargs)
            # #
            # if isinstance(module, self.projector_module_cls):
            #     fully_shard(module, **fsdp_kwargs)
            # #
            # if isinstance(module, self.action_head_module_cls):
            #     fully_shard(module, **fsdp_kwargs)
        #
        fully_shard(self.vlm, **fsdp_kwargs)

        # # torch.compile #
        # self.vlm = torch.compile(self.vlm, mode="default")  

        # Barrier =>> Sharding takes a minute?
        dist.barrier()

        # Create Optimizer and LR Scheduler =>> note that most of the LR Schedulers we use require `max_steps/epochs`
        #   => Optimizer should only operate on parameters that are *unfrozen* / trainable!
        if self.lr_scheduler_type == "linear-warmup+cosine-decay":
            n_train_examples = math.ceil(n_train_examples / self.global_batch_size) * self.global_batch_size
            if self.max_steps is None:
                num_training_steps = (n_train_examples * self.epochs) // self.global_batch_size
            else:
                num_training_steps = self.max_steps

            # Set warmup steps (floor) based on `warmup_ratio` (should be 0.03 - 0.05)
            num_warmup_steps = int(num_training_steps * self.warmup_ratio)

            # Default AdamW w/ specified LR & Linear Warmup / Cosine Decay & Weight Decay
            #   => Create Parameter Groups --> bias terms, normalization layer parameters shouldn't be decayed!
            decay, no_decay = [], []
            for name, param in self.vlm.named_parameters():
                #
                if not param.requires_grad:
                    continue
                # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                if param.ndim <= 1 or name.endswith(".bias"):
                    no_decay.append(param)
                else:
                    decay.append(param)

            # Build Parameter Groups
            groups = [{"params": decay, "weight_decay": self.weight_decay}, {"params": no_decay, "weight_decay": 0.0}]

            # Create Optimizer & LR Scheduler
            #
            self.use_lr_scheduler = True
            if   self.optimizer_type == "AdamW":
                self.optimizer = AdamW(groups, lr=self.learning_rate, fused=True)
            elif self.optimizer_type == "SGD":
                self.optimizer = SGD(groups, lr=self.learning_rate, fused=True)
            elif self.optimizer_type == "SGDM":
                self.optimizer = SGD(groups, lr=self.learning_rate, momentum=0.9, fused=True)
            elif self.optimizer_type == "ASGD":
                self.optimizer = ASGD(groups, lr=self.learning_rate)
            elif self.optimizer_type == "AdafactorP":
                self.optimizer = AdafactorP(groups, lr=self.learning_rate, scale_parameter=True, relative_step=False)
            elif self.optimizer_type == "AdafactorT":
                self.optimizer = AdafactorT(groups, lr=self.learning_rate, scale_parameter=True, relative_step=False, warmup_init=False)
            elif self.optimizer_type == "Adafactor":
                self.optimizer = Adafactor(groups)
                self.use_lr_scheduler = False
            elif self.optimizer_type == "APOLLO":
                # #
                lowrank_params, non_lowrank_params = [], []
                patterns = [r".*(\.attn|\.self_attn).*", r".*\.mlp.*"]
                for name, param in self.vlm.named_parameters():
                    #
                    if not param.requires_grad:
                        continue
                    # Check on any parameters with fewer than 2 dimensions or with "bias" in the name
                    if any(re.match(pattern, name) for pattern in patterns):
                        lowrank_params.append(param)
                    else:
                        non_lowrank_params.append(param)                
                # #
                groups = [
                    {"params": non_lowrank_params, "weight_decay": 0.0},
                    {"params":     lowrank_params, "weight_decay": self.weight_decay,
                     'rank': 256,#1, 
                     'proj': 'random', 
                     'scale_type': 'channel',#'tensor', 
                     'scale': 1,#128,
                     'update_proj_gap': 200, 
                     'proj_type': 'std'
                    }
                    ]
                self.optimizer = APOLLOAdamW(groups, lr=self.learning_rate)
            else:
                raise ValueError(f"Optimizer `{self.optimizer_type}` is not supported!")
            #
            if self.use_lr_scheduler:
                self.lr_scheduler = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps, num_training_steps)                
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = 0.0

        else:
            raise ValueError(f"Learning Rate Schedule with type `{self.lr_scheduler_type}` is not supported!")

        # Finalize Setup =>> Log!
        overwatch.info(
            "FSDP Full-Shard Strategy =>> Finalized Training Setup:\n"
            f"         |-> Global (Effective) Batch Size = {self.global_batch_size}\n"
            f"         |-> Per-Device Batch Size = {self.per_device_batch_size}\n"
            f"         |-> Distributed World Size = {overwatch.world_size()}\n"
            f"         |-> Gradient Accumulation Steps = {self.grad_accumulation_steps}\n\n"
            f"         |-> LLM Backbone FSDP Gradient Checkpointing = {self.enable_gradient_checkpointing}\n"
            f"         |-> Use FSDP Mixed Precision = {self.enable_mixed_precision_training}\n"
            f"                 |-> Parameter Precision = {fsdp_precision_policy.param_dtype}\n"
            f"                 |-> Reduction Precision = {fsdp_precision_policy.reduce_dtype}\n"
            # f"                 |-> Buffer Precision = {fsdp_precision_policy.buffer_dtype}\n\n"
            f"         |-> Default AdamW LR = {self.learning_rate}\n"
            f"         |-> AdamW Weight Decay = {self.weight_decay}\n"
            f"         |-> LR Scheduler Type = {self.lr_scheduler_type}\n"
            f"         |-> LR Scheduler Warmup Steps (Ratio) = {num_warmup_steps} ({self.warmup_ratio})\n"
            f"         |-> Dataset Size = {n_train_examples} Examples\n"
            f"         |-> Max Steps = {num_training_steps}\n"
        )

    def clip_grad_norm(self) -> None:
        torch.nn.utils.clip_grad_norm_(self.vlm.parameters(), max_norm=self.max_grad_norm)
