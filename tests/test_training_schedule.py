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
