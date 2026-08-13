from __future__ import annotations

import contextlib
import os
import random
import typing
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.utils.data
from torch.nn.parallel import DistributedDataParallel


class Accelerator:
    """
    Simplified accelerator that mirrors the behaviour of the minicpm-audio
    training utilities. It initializes a distributed process group when
    ``torchrun`` is used and exposes helpers for AMP, gradient scaling and
    preparing models/dataloaders for DDP or FSDP.
    """

    def __init__(
        self,
        amp: bool = False,
        seed: int = 42,
        distributed_strategy: str = "ddp",
        activation_checkpointing: bool = False,
    ):
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.world_size = int(os.getenv("WORLD_SIZE", "1"))
        strategy = str(distributed_strategy).strip().lower()
        if strategy not in {"ddp", "fsdp"}:
            raise ValueError(f"Unsupported distributed_strategy={distributed_strategy!r}; expected 'ddp' or 'fsdp'")
        self.distributed_strategy = strategy
        self.activation_checkpointing = bool(activation_checkpointing)

        # Model constructors may allocate runtime caches on the current CUDA
        # device before prepare_model() is called, so select the torchrun-local
        # device immediately instead of allowing every rank to use cuda:0.
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)

        if self.world_size > 1 and not dist.is_initialized():
            dist.init_process_group("nccl", init_method="env://")

        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.amp = amp

        # Set random seed to ensure model initialization consistency
        self._set_seed(seed)

        class DummyScaler:
            def step(self, optimizer):
                optimizer.step()

            def scale(self, loss):
                return loss

            def unscale_(self, optimizer):
                return optimizer

            def update(self):
                pass

        self.scaler = torch.amp.GradScaler("cuda") if (amp and torch.cuda.is_available()) else DummyScaler()
        self.device_ctx = torch.cuda.device(self.local_rank) if torch.cuda.is_available() else None
        self._ddp_model = None  # For no_sync support
        self._fsdp_model = None

    def _set_seed(self, seed: int):
        """Set random seed to ensure model initialization consistency across multiple GPUs"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def __enter__(self):
        if self.device_ctx is not None:
            self.device_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.device_ctx is not None:
            self.device_ctx.__exit__(exc_type, exc_value, traceback)

    def barrier(self):
        """Synchronize all processes"""
        if dist.is_initialized():
            dist.barrier()

    def all_reduce(self, tensor: torch.Tensor, op=dist.ReduceOp.AVG):
        """All-reduce tensor across processes"""
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor

    # ------------------------------------------------------------------ #
    # Model helpers
    # ------------------------------------------------------------------ #
    def prepare_model(self, model: torch.nn.Module, **kwargs):
        if hasattr(model, "device"):  # make sure the matrix will be moved to the correct device
            model.device = self.device

        if self.world_size > 1 and self.distributed_strategy == "fsdp":
            from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                CheckpointImpl,
                apply_activation_checkpointing,
                checkpoint_wrapper,
            )
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            from voxcpm.modules.minicpm4.model import MiniCPMDecoderLayer

            auto_wrap_policy = partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={MiniCPMDecoderLayer},
            )
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
            model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                device_id=self.device,
                sync_module_states=True,
                use_orig_params=True,
                limit_all_gathers=True,
                **kwargs,
            )
            if self.activation_checkpointing:
                checkpoint_wrapper_fn = partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
                apply_activation_checkpointing(
                    model,
                    checkpoint_wrapper_fn=checkpoint_wrapper_fn,
                    check_fn=lambda module: isinstance(module, MiniCPMDecoderLayer),
                )
            self._fsdp_model = model
        else:
            model = model.to(self.device)

        if self.world_size > 1 and self.distributed_strategy == "ddp":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = DistributedDataParallel(model, device_ids=[self.local_rank], **kwargs)
            self._ddp_model = model  # Save DDP model reference for no_sync support
        return model

    @contextlib.contextmanager
    def no_sync(self):
        """
        Context manager to skip gradient synchronization during gradient accumulation.
        Only used outside the last micro-batch.
        """
        if self._ddp_model is not None:
            with self._ddp_model.no_sync():
                yield
        else:
            yield

    @property
    def is_fsdp(self) -> bool:
        return self._fsdp_model is not None

    def clip_grad_norm_(self, model: torch.nn.Module, max_norm: float):
        """Clip gradients correctly for replicated or sharded parameters."""

        if self.is_fsdp:
            # FSDP computes the global norm collectively across parameter
            # shards. torch.nn.utils.clip_grad_norm_ would only see one rank.
            return model.clip_grad_norm_(max_norm)
        return torch.nn.utils.clip_grad_norm_(self.unwrap(model).parameters(), max_norm=max_norm)

    @contextlib.contextmanager
    def summon_full_params(self, model: torch.nn.Module):
        """Expose an unwrapped full model for deterministic FSDP inference.

        Every rank must enter this context. Parameters are materialized on all
        ranks and every rank executes the same custom ``generate`` calls so
        nested FSDP wrappers perform matching collectives. Only rank zero
        writes the resulting artifacts.
        """

        if self.is_fsdp:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

            with FSDP.summon_full_params(
                model,
                recurse=True,
                writeback=False,
                rank0_only=False,
                offload_to_cpu=False,
            ):
                yield self.unwrap(model)
        else:
            yield self.unwrap(model)

    @property
    def device(self):
        if torch.cuda.is_available():
            return torch.device("cuda", self.local_rank)
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    # ------------------------------------------------------------------ #
    # AMP helpers
    # ------------------------------------------------------------------ #
    def autocast(self, *args, **kwargs):
        return torch.amp.autocast("cuda", enabled=self.amp, *args, **kwargs)

    def backward(self, loss: torch.Tensor):
        self.scaler.scale(loss).backward()

    def step(self, optimizer: torch.optim.Optimizer):
        self.scaler.step(optimizer)

    def update(self):
        self.scaler.update()

    # ------------------------------------------------------------------ #
    # Data helpers
    # ------------------------------------------------------------------ #
    def prepare_dataloader(
        self,
        dataset: typing.Iterable,
        *,
        batch_size: int,
        num_workers: int = 0,
        shuffle: bool = True,
        collate_fn=None,
        drop_last: bool = False,
    ) -> torch.utils.data.DataLoader:
        if self.world_size > 1:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.world_size, rank=self.rank, shuffle=shuffle
            )
            shuffle = False
        else:
            sampler = None

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle if sampler is None else False,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            drop_last=drop_last,
            pin_memory=True,
        )

    @staticmethod
    def unwrap(model: torch.nn.Module) -> torch.nn.Module:
        return model.module if hasattr(model, "module") else model
