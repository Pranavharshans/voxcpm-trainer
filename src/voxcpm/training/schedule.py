"""Pure helpers for translating dataset epochs into optimizer steps."""

from __future__ import annotations

import math


def resolve_grad_accum_steps(
    *,
    global_batch_size: int,
    batch_size: int,
    world_size: int,
    fallback_grad_accum_steps: int,
) -> int:
    """Resolve accumulation while preserving a requested global batch size.

    A value of zero for ``global_batch_size`` keeps the configured fallback.
    Otherwise the requested global batch must divide exactly across the
    per-rank batch and distributed world size; silently rounding would change
    the optimizer schedule and learning dynamics.
    """

    fallback = int(fallback_grad_accum_steps)
    if int(global_batch_size) <= 0:
        if fallback <= 0:
            raise ValueError("grad_accum_steps must be positive when global_batch_size is disabled")
        return fallback

    denominator = int(batch_size) * int(world_size)
    if denominator <= 0:
        raise ValueError("batch_size and world_size must be positive")
    if int(global_batch_size) % denominator != 0:
        raise ValueError(
            f"global_batch_size={global_batch_size} must be divisible by "
            f"batch_size * world_size ({batch_size} * {world_size} = {denominator})"
        )

    resolved = int(global_batch_size) // denominator
    if resolved <= 0:
        raise ValueError(
            f"global_batch_size={global_batch_size} is smaller than the distributed micro-batch {denominator}"
        )
    return resolved


def resolve_data_resume_position(
    *,
    start_step: int,
    grad_accum_steps: int,
    batches_per_epoch: int,
) -> tuple[int, int]:
    """Map an optimizer resume step to sampler epoch and batch offset.

    Each optimizer step consumes ``grad_accum_steps`` dataloader batches on
    every rank. The returned offset is rank-local because distributed samplers
    expose one rank-local sequence each epoch.
    """

    values = {
        "start_step": start_step,
        "grad_accum_steps": grad_accum_steps,
        "batches_per_epoch": batches_per_epoch,
    }
    negative = [name for name, value in values.items() if int(value) < 0]
    if negative:
        raise ValueError(f"Resume schedule values cannot be negative: {', '.join(negative)}")
    if int(grad_accum_steps) == 0 or int(batches_per_epoch) == 0:
        raise ValueError("grad_accum_steps and batches_per_epoch must be positive")

    consumed_batches = int(start_step) * int(grad_accum_steps)
    return divmod(consumed_batches, int(batches_per_epoch))


def build_epoch_end_schedule(
    *,
    num_samples: int,
    batch_size: int,
    grad_accum_steps: int,
    world_size: int,
    num_epochs: int,
) -> tuple[int, dict[int, int]]:
    """Return total optimizer steps and zero-based step-to-epoch boundaries.

    Epoch boundaries are calculated from consumed sample presentations rather
    than rounded ``steps_per_epoch`` values. This keeps later boundaries exact
    when the dataset size is not divisible by the effective global batch.
    """

    values = {
        "num_samples": num_samples,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "world_size": world_size,
        "num_epochs": num_epochs,
    }
    invalid = [name for name, value in values.items() if int(value) <= 0]
    if invalid:
        raise ValueError(f"Epoch schedule values must be positive: {', '.join(invalid)}")

    effective_batch = int(batch_size) * int(grad_accum_steps) * int(world_size)
    boundaries = {
        math.ceil(epoch * int(num_samples) / effective_batch) - 1: epoch
        for epoch in range(1, int(num_epochs) + 1)
    }
    total_steps = math.ceil(int(num_epochs) * int(num_samples) / effective_batch)
    return total_steps, boundaries
