from __future__ import annotations

import pytest

from voxcpm.training.accelerator import OffsetDistributedSampler


def test_offset_distributed_sampler_resumes_without_loading_skipped_rows():
    dataset = list(range(20))
    sampler = OffsetDistributedSampler(
        dataset,
        num_replicas=4,
        rank=2,
        shuffle=True,
        seed=42,
    )
    sampler.set_epoch(3)
    full_epoch = list(sampler)

    sampler.set_start_index(2)
    assert list(sampler) == full_epoch[2:]
    assert len(sampler) == len(full_epoch) - 2

    sampler.set_epoch(4)
    assert sampler.start_index == 0
    assert len(sampler) == len(full_epoch)


def test_offset_distributed_sampler_rejects_invalid_position():
    sampler = OffsetDistributedSampler(list(range(8)), num_replicas=2, rank=0)
    with pytest.raises(ValueError, match="start_index"):
        sampler.set_start_index(sampler.num_samples + 1)
