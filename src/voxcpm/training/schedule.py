"""Pure helpers for translating dataset epochs into optimizer steps."""

from __future__ import annotations

import math


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
