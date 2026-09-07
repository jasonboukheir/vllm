# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Lifecycle tests for physically independent KV-cache block pools."""

from collections.abc import Sequence

import pytest
import torch

from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import KVCacheBlock, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCachePoolSpec,
    MambaSpec,
)

from .test_prefix_caching import make_request

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 4


def _make_manager(
    pool_capacities: tuple[int, int] = (5, 2),
) -> KVCacheManager:
    groups = [
        KVCacheGroupSpec(
            [f"layer.{group_id}"],
            FullAttentionSpec(
                block_size=BLOCK_SIZE,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        )
        for group_id in range(2)
    ]
    config = KVCacheConfig(
        # Legacy aggregate view: the limiting physical pool.
        num_blocks=min(pool_capacities),
        kv_cache_tensors=[],
        kv_cache_groups=groups,
        kv_cache_pools=[
            KVCachePoolSpec(num_blocks=capacity, group_ids=[pool_id])
            for pool_id, capacity in enumerate(pool_capacities)
        ],
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=64,
        scheduler_block_size=BLOCK_SIZE,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )


def _request(request_id: str, num_tokens: int = 1):
    init_none_hash(sha256)
    return make_request(
        request_id,
        list(range(num_tokens)),
        BLOCK_SIZE,
        sha256,
    )


def _pool_snapshot(pool) -> tuple:
    """State which a failed admission must leave untouched."""
    return (
        pool.get_num_free_blocks(),
        tuple(
            (
                block.block_id,
                block.pool_id,
                block.ref_cnt,
                block.block_hash,
                block.block_hash_num_tokens,
            )
            for block in pool.blocks
        ),
    )


def _ids(blocks: Sequence[KVCacheBlock]) -> list[tuple[int, int]]:
    return [(block.pool_id, block.block_id) for block in blocks]


def _manager_snapshot(manager) -> tuple:
    return tuple(
        (
            tuple(
                (request_id, tuple(_ids(blocks)))
                for request_id, blocks in sorted(single.req_to_blocks.items())
            ),
            tuple(sorted(single.num_cached_block.items())),
        )
        for single in manager.coordinator.single_type_managers
    )


def test_unequal_capacities_have_overlapping_local_block_ids():
    manager = _make_manager(pool_capacities=(5, 2))
    pools = manager.coordinator.block_pools

    assert pools[0] is not pools[1]
    assert [pool.num_gpu_blocks for pool in pools] == [5, 2]

    request = _request("request")
    new_blocks = manager.allocate_slots(request, num_new_tokens=1)
    assert new_blocks is not None

    # Block 0 is each pool's null block. Real block IDs are also pool-local,
    # so both first allocations legitimately use ID 1.
    assert _ids(new_blocks.blocks[0]) == [(0, 1)]
    assert _ids(new_blocks.blocks[1]) == [(1, 1)]
    assert [pool.get_num_free_blocks() for pool in pools] == [3, 0]


def test_limiting_pool_rejects_without_mutating_other_pool():
    manager = _make_manager(pool_capacities=(5, 2))
    pools = manager.coordinator.block_pools

    first = _request("first")
    assert manager.allocate_slots(first, num_new_tokens=1) is not None
    before = tuple(_pool_snapshot(pool) for pool in pools)
    manager_state_before = _manager_snapshot(manager)

    # Pool 0 has three free blocks, but pool 1 is exhausted. Aggregate free
    # space is therefore not a valid admission criterion.
    second = _request("second")
    assert manager.allocate_slots(second, num_new_tokens=1) is None

    assert tuple(_pool_snapshot(pool) for pool in pools) == before
    assert _manager_snapshot(manager) == manager_state_before
    assert all(
        second.request_id not in single.req_to_blocks
        for single in manager.coordinator.single_type_managers
    )


def test_free_returns_blocks_to_their_pool_and_reuses_local_ids():
    manager = _make_manager(pool_capacities=(5, 2))
    pools = manager.coordinator.block_pools
    initial_free = [pool.get_num_free_blocks() for pool in pools]

    first = _request("first")
    first_blocks = manager.allocate_slots(first, num_new_tokens=1)
    assert first_blocks is not None
    first_ids = tuple(_ids(group) for group in first_blocks.blocks)
    assert first_ids == ([(0, 1)], [(1, 1)])

    manager.free(first)
    assert [pool.get_num_free_blocks() for pool in pools] == initial_free
    assert all(
        first.request_id not in single.req_to_blocks
        for single in manager.coordinator.single_type_managers
    )

    second = _request("second")
    second_blocks = manager.allocate_slots(second, num_new_tokens=1)
    assert second_blocks is not None
    second_ids = tuple(_ids(group) for group in second_blocks.blocks)
    assert all(
        pool_id == group_id
        for group_id, group in enumerate(second_ids)
        for pool_id, _ in group
    )
    # The single-block limiting pool must reuse its just-freed local ID. The
    # larger pool is free to preserve its LRU queue order and use its next ID.
    assert second_ids[1] == first_ids[1] == [(1, 1)]


def test_usage_reports_pressure_of_limiting_pool():
    manager = _make_manager(pool_capacities=(5, 2))
    assert manager.usage == 0.0

    request = _request("request")
    assert manager.allocate_slots(request, num_new_tokens=1) is not None

    # Pool usages are 1/4 and 1/1. The aggregate must reflect the exhausted
    # pool, not average unrelated block counts into a misleading 2/5.
    assert [pool.get_usage() for pool in manager.coordinator.block_pools] == [
        pytest.approx(0.25),
        pytest.approx(1.0),
    ]
    assert manager.usage == pytest.approx(1.0)


def test_mamba_zeroing_ids_remain_group_qualified():
    config = KVCacheConfig(
        num_blocks=5,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=64,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="none",
                ),
            ),
        ],
        kv_cache_pools=[
            KVCachePoolSpec(num_blocks=32, group_ids=[0]),
            KVCachePoolSpec(num_blocks=5, group_ids=[1]),
        ],
    )
    manager = KVCacheManager(
        kv_cache_config=config,
        max_model_len=64,
        scheduler_block_size=64,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )

    blocks = manager.allocate_slots(_request("request", 4), num_new_tokens=4)
    assert blocks is not None
    assert manager.take_new_block_ids()
    assert manager.take_new_mamba_block_ids() == [(1, [1])]
