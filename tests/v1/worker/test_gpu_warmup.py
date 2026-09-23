# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCachePoolSpec,
)
from vllm.v1.worker.gpu.warmup import (
    _cap_warmup_requests_by_pool,
    _pool_block_totals,
    _WarmupBlockAllocator,
)


def _independent_pool_config() -> KVCacheConfig:
    groups = [
        KVCacheGroupSpec(
            [f"layer.{group_id}"],
            FullAttentionSpec(
                block_size=16,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float16,
            ),
        )
        for group_id in range(3)
    ]
    return KVCacheConfig(
        num_blocks=7,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        kv_cache_pools=[
            KVCachePoolSpec(num_blocks=17, group_ids=[0]),
            KVCachePoolSpec(num_blocks=17, group_ids=[1]),
            KVCachePoolSpec(num_blocks=257, group_ids=[2]),
        ],
    )


def test_warmup_block_ids_are_local_to_independent_pools() -> None:
    allocator = _WarmupBlockAllocator(_independent_pool_config())

    assert allocator.allocate(0, 4) == [1, 2, 3, 4]
    assert allocator.allocate(1, 4) == [1, 2, 3, 4]
    assert allocator.allocate(2, 2) == [1, 2]
    assert allocator.allocate(0, 2) == [5, 6]


def test_warmup_concurrency_is_capped_per_pool() -> None:
    config = _independent_pool_config()

    assert _pool_block_totals(config, [4, 4, 64]) == [4, 4, 64]
    # Each 17-block recurrent-like pool reserves block zero and admits four
    # four-page request bundles; the larger attention pool is non-limiting.
    assert _cap_warmup_requests_by_pool(config, [4, 4, 64], 8) == 4


def test_shared_pool_still_sums_group_block_ids() -> None:
    config = _independent_pool_config()
    config = KVCacheConfig(
        num_blocks=257,
        kv_cache_tensors=[],
        kv_cache_groups=config.kv_cache_groups,
    )

    assert _pool_block_totals(config, [4, 4, 64]) == [72]
    assert _cap_warmup_requests_by_pool(config, [4, 4, 64], 8) == 3
