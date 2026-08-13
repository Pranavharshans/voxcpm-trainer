from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCHEDULE_MODULE = ROOT / "src" / "voxcpm" / "training" / "schedule.py"
spec = importlib.util.spec_from_file_location("voxcpm_training_schedule", SCHEDULE_MODULE)
schedule_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = schedule_module
assert spec.loader is not None
spec.loader.exec_module(schedule_module)

build_epoch_end_schedule = schedule_module.build_epoch_end_schedule
resolve_data_resume_position = schedule_module.resolve_data_resume_position
resolve_grad_accum_steps = schedule_module.resolve_grad_accum_steps


def test_builds_exact_boundaries_for_non_divisible_dataset():
    total_steps, boundaries = build_epoch_end_schedule(
        num_samples=71_108,
        batch_size=1,
        grad_accum_steps=16,
        world_size=1,
        num_epochs=3,
    )

    assert total_steps == 13_333
    assert boundaries == {4_444: 1, 8_888: 2, 13_332: 3}


def test_rejects_non_positive_schedule_values():
    with pytest.raises(ValueError, match="num_epochs"):
        build_epoch_end_schedule(
            num_samples=10,
            batch_size=1,
            grad_accum_steps=1,
            world_size=1,
            num_epochs=0,
        )


@pytest.mark.parametrize(
    ("world_size", "expected"),
    [(1, 16), (2, 8), (4, 4), (8, 2)],
)
def test_preserves_global_batch_across_supported_world_sizes(world_size, expected):
    assert (
        resolve_grad_accum_steps(
            global_batch_size=16,
            batch_size=1,
            world_size=world_size,
            fallback_grad_accum_steps=16,
        )
        == expected
    )


def test_rejects_world_size_that_cannot_preserve_global_batch():
    with pytest.raises(ValueError, match="must be divisible"):
        resolve_grad_accum_steps(
            global_batch_size=16,
            batch_size=1,
            world_size=6,
            fallback_grad_accum_steps=16,
        )


def test_uses_fallback_when_global_batch_is_disabled():
    assert (
        resolve_grad_accum_steps(
            global_batch_size=0,
            batch_size=2,
            world_size=4,
            fallback_grad_accum_steps=3,
        )
        == 3
    )


@pytest.mark.parametrize(
    ("start_step", "grad_accum_steps", "batches_per_epoch", "expected"),
    [
        (0, 4, 17_777, (0, 0)),
        (500, 4, 17_777, (0, 2_000)),
        (4_445, 4, 17_777, (1, 3)),
        (8_889, 4, 17_777, (2, 2)),
    ],
)
def test_resolves_exact_dataloader_resume_position(
    start_step, grad_accum_steps, batches_per_epoch, expected
):
    assert (
        resolve_data_resume_position(
            start_step=start_step,
            grad_accum_steps=grad_accum_steps,
            batches_per_epoch=batches_per_epoch,
        )
        == expected
    )
