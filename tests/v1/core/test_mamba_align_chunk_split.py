# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mamba "align" prefill chunk splitting (`_mamba_block_aligned_split`).

Invariant: slot `p` holds the state after exactly `(p + 1) * block_size` tokens.
State is written at chunk ends, so a chunk ending mid-block leaves its slot at
the wrong offset, and a later chunk crossing that boundary publishes it anyway.
Requests resuming from it then restore a truncated state (#43559).
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import OPTConfig

import vllm.platforms as platforms
from vllm.platforms.cpu import CpuPlatform
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.sched import scheduler as scheduler_module
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

# Mirrors the deployment where the poisoning was observed (Qwen3.6-27B): mamba
# block 1600, MTP with 3 draft tokens, prompts shorter than 2 mamba blocks.
ATTN_BLOCK_SIZE = 16
MAMBA_BLOCK_SIZE = 1600
NUM_SPEC = 3
PROMPT_LEN = 2002
MAMBA_GROUP_ID = 1


def _make_hybrid_kv_cache_manager(
    num_prefill_checkpoint_blocks: int = 0,
) -> KVCacheManager:
    config = KVCacheConfig(
        num_blocks=10000,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full_layer"],
                FullAttentionSpec(
                    block_size=ATTN_BLOCK_SIZE,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba_layer"],
                MambaSpec(
                    block_size=MAMBA_BLOCK_SIZE,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                    num_speculative_blocks=NUM_SPEC,
                    num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
                ),
            ),
        ],
    )
    return KVCacheManager(
        config,
        max_model_len=262144,
        scheduler_block_size=MAMBA_BLOCK_SIZE,
        hash_block_size=ATTN_BLOCK_SIZE,
        enable_caching=True,
        use_eagle=True,
    )


def _split(
    request: Request,
    num_new_tokens: int,
    use_eagle: bool = True,
    partial_hit: bool = False,
    num_prefill_checkpoint_blocks: int = 0,
) -> int:
    """Call the real `Scheduler._mamba_block_aligned_split` on a stub self."""
    stub = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=MAMBA_BLOCK_SIZE),
        mamba_prefill_alignment=MAMBA_BLOCK_SIZE,
        needs_mamba_cache_alignment=True,
        use_eagle=use_eagle,
        max_num_scheduled_tokens=16384,
        scheduler_config=SimpleNamespace(long_prefill_token_threshold=0),
        # `prefix_match_unit` finer than the block size (#46384).
        mamba_partial_cache_hit=partial_hit,
        hash_block_size=ATTN_BLOCK_SIZE,
        mamba_has_prefill_checkpoint_blocks=(
            num_prefill_checkpoint_blocks > 0 and not use_eagle
        ),
    )
    return Scheduler._mamba_block_aligned_split(stub, request, num_new_tokens)


XPU_GDN_CHUNK_SIZE = 64


def _make_xpu_gdn_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    is_xpu: bool = True,
    max_num_batched_tokens: int = 8192,
    long_prefill_token_threshold: int = 0,
    mamba_type: MambaAttentionBackendEnum = MambaAttentionBackendEnum.QWEN_GDN_ATTN,
    num_speculative_tokens: int | None = None,
    supports_multimodal_inputs: bool = False,
) -> Scheduler:
    monkeypatch.setattr(platforms, "current_platform", CpuPlatform())
    monkeypatch.setattr(
        scheduler_module,
        "current_platform",
        SimpleNamespace(is_xpu=lambda: is_xpu),
    )
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path / "cache"))
    if supports_multimodal_inputs:
        monkeypatch.setattr(
            scheduler_module.MultiModalRegistry,
            "supports_multimodal_inputs",
            lambda *_args, **_kwargs: True,
        )
        monkeypatch.setattr(
            scheduler_module,
            "MultiModalBudget",
            lambda *_args, **_kwargs: SimpleNamespace(
                encoder_compute_budget=0,
                encoder_cache_size=0,
            ),
        )
    model_path = tmp_path / "opt"
    OPTConfig(max_position_embeddings=65536).save_pretrained(model_path)
    return create_scheduler(
        model=str(model_path),
        max_num_seqs=4,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=65536,
        long_prefill_token_threshold=long_prefill_token_threshold,
        num_speculative_tokens=num_speculative_tokens,
        block_size=XPU_GDN_CHUNK_SIZE,
        kv_cache_spec=MambaSpec(
            block_size=XPU_GDN_CHUNK_SIZE,
            shapes=((1, 1),),
            dtypes=(torch.float32,),
            mamba_type=mamba_type,
            mamba_cache_mode="none",
        ),
    )


def _add_b4_ragged_requests(scheduler: Scheduler) -> list[Request]:
    requests = [
        create_requests(
            1,
            num_tokens=num_tokens,
            block_size=XPU_GDN_CHUNK_SIZE,
            req_ids=[request_id],
        )[0]
        for request_id, num_tokens in (
            ("short-a", 127),
            ("long-a", 4095),
            ("long-b", 4095),
            ("short-b", 127),
        )
    ]
    for request in requests:
        scheduler.add_request(request)
    return requests


def test_xpu_gdn_b4_budget_uses_absolute_64_token_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = _make_xpu_gdn_scheduler(monkeypatch, tmp_path)
    _add_b4_ragged_requests(scheduler)

    output = scheduler.schedule()

    assert scheduler.cache_config.mamba_cache_mode == "none"
    assert scheduler.mamba_prefill_alignment == XPU_GDN_CHUNK_SIZE
    assert output.num_scheduled_tokens == {
        "short-a": 127,
        "long-a": 4095,
        "long-b": 3968,
    }
    assert sum(output.num_scheduled_tokens.values()) == 8190
    assert output.num_scheduled_tokens["long-b"] % XPU_GDN_CHUNK_SIZE == 0


def test_xpu_gdn_final_tail_is_not_truncated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = _make_xpu_gdn_scheduler(monkeypatch, tmp_path)
    (request,) = create_requests(
        1,
        num_tokens=4095,
        block_size=XPU_GDN_CHUNK_SIZE,
    )
    request.num_computed_tokens = 3968

    assert scheduler._mamba_block_aligned_split(request, 127) == 127


def test_xpu_gdn_arithmetic_grid_survives_internal_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = _make_xpu_gdn_scheduler(monkeypatch, tmp_path)
    scheduler.mamba_has_prefill_checkpoint_blocks = True
    (request,) = create_requests(
        1,
        num_tokens=4095,
        block_size=XPU_GDN_CHUNK_SIZE,
    )

    assert scheduler._mamba_block_aligned_split(request, 3970) == 3968


@pytest.mark.parametrize(
    ("max_num_batched_tokens", "long_prefill_token_threshold", "match"),
    [
        (66, 0, "effective scheduler token budget"),
        (8192, 63, "long_prefill_token_threshold"),
    ],
)
def test_xpu_gdn_sub_64_config_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    max_num_batched_tokens: int,
    long_prefill_token_threshold: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _make_xpu_gdn_scheduler(
            monkeypatch,
            tmp_path,
            max_num_batched_tokens=max_num_batched_tokens,
            long_prefill_token_threshold=long_prefill_token_threshold,
        )


def test_xpu_gdn_budget_reserves_an_aligned_prefill_after_b4_decodes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = _make_xpu_gdn_scheduler(
        monkeypatch,
        tmp_path,
        max_num_batched_tokens=XPU_GDN_CHUNK_SIZE + 3,
    )

    decode_requests = create_requests(
        3,
        num_tokens=1,
        block_size=XPU_GDN_CHUNK_SIZE,
        req_ids=["decode-a", "decode-b", "decode-c"],
    )
    for request in decode_requests:
        scheduler.add_request(request)
    prompt_output = scheduler.schedule()
    scheduler.update_from_output(
        prompt_output,
        ModelRunnerOutput(
            req_ids=[request.request_id for request in decode_requests],
            req_id_to_index={
                request.request_id: index
                for index, request in enumerate(decode_requests)
            },
            sampled_token_ids=[[0], [0], [0]],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[],
        ),
    )

    (prefill_request,) = create_requests(
        1,
        num_tokens=4095,
        block_size=XPU_GDN_CHUNK_SIZE,
        req_ids=["prefill"],
    )
    scheduler.add_request(prefill_request)
    output = scheduler.schedule()

    assert output.num_scheduled_tokens == {
        "decode-a": 1,
        "decode-b": 1,
        "decode-c": 1,
        "prefill": XPU_GDN_CHUNK_SIZE,
    }


def test_non_xpu_gdn_default_scheduling_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = _make_xpu_gdn_scheduler(monkeypatch, tmp_path, is_xpu=False)
    _add_b4_ragged_requests(scheduler)

    output = scheduler.schedule()

    assert scheduler.mamba_prefill_alignment == 1
    assert not scheduler.need_mamba_block_aligned_split
    assert output.num_scheduled_tokens == {
        "short-a": 127,
        "long-a": 4095,
        "long-b": 3970,
    }
    assert sum(output.num_scheduled_tokens.values()) == 8192


def test_xpu_generic_gdn_default_scheduling_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    scheduler = _make_xpu_gdn_scheduler(
        monkeypatch,
        tmp_path,
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
    )

    assert scheduler.mamba_prefill_alignment == 1
    assert not scheduler.need_mamba_block_aligned_split


@pytest.mark.parametrize(
    "profile",
    ["speculative", "multimodal"],
)
def test_xpu_qwen_gdn_out_of_scope_profiles_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile: str,
) -> None:
    scheduler = _make_xpu_gdn_scheduler(
        monkeypatch,
        tmp_path,
        num_speculative_tokens=3 if profile == "speculative" else None,
        supports_multimodal_inputs=profile == "multimodal",
    )

    assert scheduler.mamba_prefill_alignment == 1
    assert not scheduler.need_mamba_block_aligned_split


@pytest.mark.parametrize(
    ("prompt_len", "num_new_tokens", "use_eagle", "expected"),
    [
        (2002, 2002, False, 2002),
        (3602, 2000, False, 2000),
        (3602, 3602, True, MAMBA_BLOCK_SIZE),
    ],
)
def test_internal_checkpoint_split(
    prompt_len: int, num_new_tokens: int, use_eagle: bool, expected: int
) -> None:
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    assert (
        _split(
            request,
            num_new_tokens,
            use_eagle=use_eagle,
            num_prefill_checkpoint_blocks=1,
        )
        == expected
    )
    if not use_eagle:
        manager = _make_hybrid_kv_cache_manager(num_prefill_checkpoint_blocks=1)
        assert manager.allocate_slots(request, expected) is not None
        mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
        blocks = mamba_manager.req_to_blocks[request.request_id]
        assert all(not block.is_null for block in blocks)  # checkpoint + running state


def _run_chunked_prefill(
    manager: KVCacheManager, request: Request, budgets: list[int]
) -> dict[int, int]:
    """Prefill `request`, one step per entry in `budgets`.

    A zero-token split means the budget cannot fund an aligned chunk; the
    scheduler defers the request to a later step, so this does too.

    Returns physical mamba block id -> the token offset of the state it holds,
    mirroring the GDN kernel: the running slot ends up at the chunk end.
    """
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    state_at: dict[int, int] = {}
    # `budgets` fragments the first steps; afterwards the request is alone and
    # gets as much as it can use, so the prefill always finishes.
    for step in range(len(budgets) + 64):
        computed = request.num_computed_tokens
        if computed >= request.num_tokens:
            break
        budget = budgets[step] if step < len(budgets) else request.num_tokens
        num_new = _split(request, min(request.num_tokens - computed, budget))
        if num_new == 0:
            continue
        assert (
            manager.allocate_slots(request, num_new, num_lookahead_tokens=NUM_SPEC)
            is not None
        )
        request.num_computed_tokens = computed + num_new
        blocks = mamba_manager.req_to_blocks[request.request_id]
        running = cdiv(request.num_computed_tokens, MAMBA_BLOCK_SIZE) - 1
        state_at[blocks[running].block_id] = request.num_computed_tokens
    return state_at


def _count_cached_boundary_states(
    manager: KVCacheManager, request: Request, state_at: dict[int, int]
) -> int:
    """Assert every hash-cached mamba slot holds the state its hash claims.

    Covers both full-block snapshots (`(p + 1) * block_size`) and the
    partial-tail entries align mode registers at an exact token count.

    Returns the number of cached slots checked.
    """
    mamba_manager = manager.coordinator.single_type_managers[MAMBA_GROUP_ID]
    checked = 0
    for pos, block in enumerate(mamba_manager.req_to_blocks[request.request_id]):
        if block.is_null or block.block_hash is None:
            continue
        claimed = block.block_hash_num_tokens
        assert state_at.get(block.block_id) == claimed, (
            f"mamba slot {pos} is hashed as state@{claimed} but holds "
            f"state@{state_at.get(block.block_id)}"
        )
        checked += 1
    return checked


def _prefill(prompt_len: int, budgets: list[int]) -> int:
    manager = _make_hybrid_kv_cache_manager()
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    state_at = _run_chunked_prefill(manager, request, budgets)
    assert request.num_computed_tokens == prompt_len, "prefill did not complete"
    return _count_cached_boundary_states(manager, request, state_at)


def test_fragmented_first_chunk_does_not_poison_mamba_prefix_cache() -> None:
    """EAGLE zeroes `last_cache_position` for prompts under two blocks.

    Past that point any chunk end used to be accepted, so a short first chunk
    (concurrent prefills sharing the budget) left slot 0 at state@364 while the
    next chunk crossed 1600 and published slot 0 as state@1600.
    """
    _prefill(PROMPT_LEN, budgets=[364, PROMPT_LEN])


def test_fragmented_tail_chunk_does_not_poison_mamba_prefix_cache() -> None:
    """Same poisoning one block in, where a hit is still cacheable.

    `last_cache_position` is 1600, so the chunk ending there is cached. The next
    chunk used to be free to stop mid-block (slot 1 at state@2600) and the one
    after it crossed 3200, publishing slot 1 as state@3200.
    """
    assert _prefill(3602, budgets=[1600, 1000, 3602]) > 0


@pytest.mark.parametrize("first_chunk", [800, 900, 1599, 1601, 2000])
def test_intermediate_chunk_ends_stay_block_aligned(first_chunk: int) -> None:
    """Every non-final prefill chunk must end on a mamba block boundary."""
    _prefill(PROMPT_LEN, budgets=[first_chunk, PROMPT_LEN, PROMPT_LEN])


@pytest.mark.parametrize(
    ("block_size", "prompt_len", "budgets"),
    [
        # Kimi-K3-scale mamba blocks: TP8 shards the recurrent state 8 ways
        # (~1.5k block), DEP16 keeps it whole (~12k block). The first budget
        # walks the request up to `last_cache_position`; the second is the
        # sub-block fragment that lands in the unguarded tail region.
        (1536, 30000, [27648, 1024, 30000]),
        (12288, 30000, [12288, 4000, 30000]),
        (12288, 41000, [24576, 8000, 41000]),
    ],
)
def test_poisoning_is_block_size_independent(
    monkeypatch: pytest.MonkeyPatch,
    block_size: int,
    prompt_len: int,
    budgets: list[int],
) -> None:
    """The invariant is per-block, so large mamba blocks are not safer."""
    import sys

    monkeypatch.setattr(sys.modules[__name__], "MAMBA_BLOCK_SIZE", block_size)
    assert _prefill(prompt_len, budgets=budgets) > 0


@pytest.mark.parametrize("partial_hit", [False, True])
@pytest.mark.parametrize("resume_at", [331, 1599, 1601, 2531, 3011])
@pytest.mark.parametrize(
    ("num_prefill_checkpoint_blocks", "use_eagle"),
    [(0, True), (1, False)],
)
def test_unaligned_resume_never_runs_past_its_block(
    partial_hit: bool,
    resume_at: int,
    num_prefill_checkpoint_blocks: int,
    use_eagle: bool,
) -> None:
    """A prefill resuming mid-block must re-align before crossing a boundary.

    Reachable with a finer `prefix_match_unit` (its partial-tail stop ends a
    chunk off-grid by design) and with unaligned external tokens from a KV
    connector.
    """
    prompt_len = 3602
    (request,) = create_requests(1, num_tokens=prompt_len, block_size=ATTN_BLOCK_SIZE)
    tail_boundary = prompt_len // ATTN_BLOCK_SIZE * ATTN_BLOCK_SIZE

    pos, ends = resume_at, []
    while pos < prompt_len:
        request.num_computed_tokens = pos
        num_new = _split(
            request,
            prompt_len - pos,
            use_eagle=use_eagle,
            partial_hit=partial_hit,
            num_prefill_checkpoint_blocks=num_prefill_checkpoint_blocks,
        )
        assert num_new > 0, f"no progress at {pos}"
        if pos % MAMBA_BLOCK_SIZE != 0:
            block_end = (pos // MAMBA_BLOCK_SIZE + 1) * MAMBA_BLOCK_SIZE
            assert pos + num_new <= block_end, (
                f"chunk [{pos}, {pos + num_new}) starts mid-block and runs past "
                f"{block_end}; the slot holding state@{pos} gets hashed as "
                f"state@{block_end}"
            )
        pos += num_new
        ends.append(pos)

    for end in ends[:-1]:
        aligned = end % MAMBA_BLOCK_SIZE == 0
        assert aligned or (partial_hit and end == tail_boundary), (
            f"intermediate chunk end {end} is neither block-aligned nor the "
            f"partial-tail boundary"
        )
