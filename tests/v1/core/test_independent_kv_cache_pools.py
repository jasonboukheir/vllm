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


@pytest.fixture(autouse=True)
def _initialize_block_hash_seed():
    init_none_hash(sha256)


def _make_manager(
    pool_capacities: tuple[int, int] = (5, 2),
    *,
    enable_caching: bool = False,
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
        enable_caching=enable_caching,
    )


def _request(request_id: str, num_tokens: int = 1):
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


def test_prefix_cache_hits_are_resolved_in_each_independent_pool():
    """Identical local block IDs must resolve through their owning pool."""
    manager = _make_manager(pool_capacities=(5, 5), enable_caching=True)
    first = _request("first", num_tokens=8)

    assert manager.allocate_slots(first, num_new_tokens=8) is not None
    manager.free(first)

    second = _request("second", num_tokens=8)
    blocks, num_computed_tokens, _ = manager.get_computed_blocks(second)

    # vLLM recomputes the final token, so an eight-token prompt reuses the
    # first complete four-token block. Both physical pools legitimately use
    # the same local ID, distinguished by KVCacheBlock.pool_id.
    assert num_computed_tokens == BLOCK_SIZE
    assert [_ids(group) for group in blocks.blocks] == [
        [(0, 1)],
        [(1, 1)],
    ]

    allocated = manager.allocate_slots(
        second,
        num_new_tokens=4,
        num_new_computed_tokens=num_computed_tokens,
        new_computed_blocks=blocks,
    )
    assert allocated is not None
    assert all(block.ref_cnt == 1 for group in blocks.blocks for block in group)
    assert all(
        block.pool_id == group_id
        for group_id, group in enumerate(allocated.blocks)
        for block in group
    )


def test_independent_pool_kv_events_remain_fail_closed():
    config = _make_manager().kv_cache_config
    with pytest.raises(NotImplementedError, match="KV cache events"):
        KVCacheManager(
            kv_cache_config=config,
            max_model_len=64,
            scheduler_block_size=BLOCK_SIZE,
            hash_block_size=BLOCK_SIZE,
            enable_caching=True,
            enable_kv_cache_events=True,
        )


def _make_hybrid_spec_manager(num_spec_tokens: int = 2) -> KVCacheManager:
    """Independent attention/recurrent pools with configurable MTP gamma."""
    config = KVCacheConfig(
        num_blocks=5,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    # In mode=none this is capacity metadata: one recurrent
                    # state is resident per request, not one per 64 tokens.
                    block_size=64,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="none",
                    num_speculative_blocks=num_spec_tokens,
                ),
            ),
        ],
        kv_cache_pools=[
            KVCachePoolSpec(num_blocks=32, group_ids=[0]),
            KVCachePoolSpec(num_blocks=5, group_ids=[1]),
        ],
    )
    return KVCacheManager(
        kv_cache_config=config,
        max_model_len=64,
        scheduler_block_size=64,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
        use_eagle=True,
    )


@pytest.mark.parametrize("num_spec_tokens", [0, 1, 2])
@pytest.mark.parametrize("context_len", [1, 31, 63, 64])
def test_mamba_none_memory_is_context_invariant(
    num_spec_tokens: int, context_len: int
):
    """Mode=none costs one running state plus gamma scratch states.

    Unlike full attention, recurrent cache residency must not scale with the
    context length. ``block_size=max_model_len`` is scheduling metadata here.
    The exact max-length case also guards against lookahead spilling into an
    accidental fourth recurrent-state block.
    """
    manager = _make_hybrid_spec_manager(num_spec_tokens)
    request = _request("context", num_tokens=context_len)
    initial_mamba_free = manager.block_pools[1].get_num_free_blocks()

    assert manager.allocate_slots(
        request,
        num_new_tokens=context_len,
        num_lookahead_tokens=num_spec_tokens,
    ) is not None
    mamba_blocks = manager.get_blocks(request.request_id).blocks[1]

    assert len(mamba_blocks) == 1 + num_spec_tokens
    assert manager.block_pools[1].get_num_free_blocks() == (
        initial_mamba_free - 1 - num_spec_tokens
    )


def test_mtp2_hybrid_pools_keep_one_committed_and_two_draft_states():
    """Rejected MTP states are scratch slots, not sequence-length growth.

    Linear attention keeps s0 (the committed running state) and one private
    state for each of the two speculative positions.  The two draft paths
    share s0; they do not each clone the committed prefix.  Full attention, by
    contrast, reserves ordinary token slots for target + lookahead tokens.
    """
    manager = _make_hybrid_spec_manager()
    request = _request("mtp2", num_tokens=3)
    initial_free = [pool.get_num_free_blocks() for pool in manager.block_pools]

    allocated = manager.allocate_slots(
        request, num_new_tokens=3, num_lookahead_tokens=2
    )
    assert allocated is not None
    full_blocks, state_blocks = manager.get_blocks(request.request_id).blocks

    # Five token positions straddle two 4-token full-attention blocks. Mamba's
    # physical cost is exactly s0+s1+s2, independent of the context length.
    assert len(full_blocks) == 2
    assert len(state_blocks) == 3
    assert [pool.get_num_free_blocks() for pool in manager.block_pools] == [
        initial_free[0] - 2,
        initial_free[1] - 3,
    ]

    state_ids = _ids(state_blocks)
    full_ids = _ids(full_blocks)

    # Model a step in which both drafts were rejected: only the three prompt
    # tokens became computed. The next target+two-draft step reuses both the
    # full-attention lookahead page and all recurrent state slots. In
    # particular state_ids[0] remains the shared committed s0.
    request.num_computed_tokens = 3
    reused = manager.allocate_slots(
        request, num_new_tokens=1, num_lookahead_tokens=2
    )
    assert reused is not None
    assert all(not group for group in reused.blocks)
    full_after, states_after = manager.get_blocks(request.request_id).blocks
    assert _ids(full_after) == full_ids
    assert _ids(states_after) == state_ids


def test_mtp2_reject_scratch_is_reclaimed_when_request_finishes():
    """Lookahead pages persist for reuse, then all return to their own pool."""
    manager = _make_hybrid_spec_manager()
    request = _request("mtp2", num_tokens=3)
    initial_free = [pool.get_num_free_blocks() for pool in manager.block_pools]

    assert manager.allocate_slots(
        request, num_new_tokens=3, num_lookahead_tokens=2
    ) is not None
    request.num_computed_tokens = 3  # both speculative tokens rejected

    # Reject does not eagerly churn allocations: scratch is safe to overwrite
    # on the next step. Request teardown is the reclamation boundary.
    assert [len(group) for group in manager.get_blocks(request.request_id).blocks] == [
        2,
        3,
    ]
    manager.free(request)
    assert [pool.get_num_free_blocks() for pool in manager.block_pools] == initial_free
    assert all(
        request.request_id not in single.req_to_blocks
        for single in manager.coordinator.single_type_managers
    )
