# SPDX-License-Identifier: Apache-2.0
"""B70 differential gates for the fused KVarN Sinkhorn record writer."""

import os
from types import SimpleNamespace

import pytest
import torch

RECORD_BYTES = 35_072
PACKED_BYTES = 256 * 128 // 2
K_S_COL = PACKED_BYTES
K_ZP = K_S_COL + 256 * 2
K_S_ROW = K_ZP + 256 * 2
V_PACKED = K_S_ROW + 128 * 2
V_S_COL = V_PACKED + PACKED_BYTES
V_S_ROW = V_S_COL + 256 * 2
V_ZP = V_S_ROW + 128 * 2


def _load_ops() -> None:
    library = os.environ.get("VLLM_XPU_KERNELS_LIBRARY")
    if library:
        torch.ops.load_library(library)
    else:
        import vllm_xpu_kernels._vllm_fa2_C  # noqa: F401


@pytest.fixture(scope="module", autouse=True)
def b70_runtime() -> None:
    if not torch.xpu.is_available():
        pytest.skip("an XPU is not available")
    assert torch.xpu.get_device_name(0) == "Intel(R) Arc(TM) Pro B70 Graphics"
    _load_ops()
    required = ("kvarn_pack_balanced_kv", "kvarn_sinkhorn_pack_kv")
    if not all(hasattr(torch.ops._vllm_fa2_C, name) for name in required):
        pytest.skip("the built extension does not contain both writer ABIs")


@pytest.fixture(autouse=True)
def exact_rtn(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KVARN_RTN_QUANTILE", raising=False)


def _tail_case(case: str, *, pool_size: int = 4) -> tuple[torch.Tensor, ...]:
    shape = (pool_size, 128, 4, 256)
    if case.startswith("random-"):
        generator = torch.Generator().manual_seed(int(case.removeprefix("random-")))
        key = torch.randn(shape, generator=generator)
        value = torch.randn(shape, generator=generator) * 1.25 - 0.125
    elif case == "constant":
        key = torch.full(shape, 0.75)
        value = torch.full(shape, -1.5)
    elif case == "extreme":
        generator = torch.Generator().manual_seed(1701)
        key = (torch.randn(shape, generator=generator) * 384).clamp(-2048, 2048)
        value = (torch.randn(shape, generator=generator) * 768).clamp(-4096, 4096)
        key[..., 0] = -2048
        key[..., -1] = 2048
        value[:, 0, :, :] = -4096
        value[:, -1, :, :] = 4096
    elif case == "near-tie":
        tokens = (torch.arange(128, dtype=torch.float32) % 16) + 0.5
        channels = (torch.arange(256, dtype=torch.float32) % 17) / 1024
        heads = torch.arange(4, dtype=torch.float32) / 2048
        key = (
            tokens[None, :, None, None]
            + channels[None, None, None, :]
            + heads[None, None, :, None]
        ).expand(shape)
        value = (
            channels[None, None, None, :]
            + tokens[None, :, None, None] / 16
            - heads[None, None, :, None]
        ).expand(shape)
    else:
        raise AssertionError(f"unknown case {case}")
    return key.to(torch.float16).contiguous(), value.to(torch.float16).contiguous()


def _write_production_control(
    tail_key: torch.Tensor,
    tail_value: torch.Tensor,
    pool_slots: torch.Tensor,
    block_ids: torch.Tensor,
    packed_cache: torch.Tensor,
    iterations: int,
) -> None:
    """Run the actual service Triton Sinkhorn + native balanced pack path."""
    from vllm.v1.attention.backends import kvarn_attn

    block_count = block_ids.numel()
    key = tail_key.index_select(0, pool_slots).float()
    value = tail_value.index_select(0, pool_slots).float()
    key_tiles = key.permute(0, 2, 3, 1).reshape(block_count * 4, 256, 128)
    value_tiles = value.permute(0, 2, 1, 3).reshape(block_count * 4, 128, 256)
    balanced = kvarn_attn._sinkhorn_balance_kv(
        key_tiles, value_tiles, SimpleNamespace(sinkhorn_iters=iterations)
    )
    kvarn_attn._launch_kvarn_native_balanced_writer(balanced, block_ids, packed_cache)


def _write_fused_candidate(
    tail_key: torch.Tensor,
    tail_value: torch.Tensor,
    pool_slots: torch.Tensor,
    block_ids: torch.Tensor,
    packed_cache: torch.Tensor,
    iterations: int,
) -> None:
    from vllm.v1.attention.backends import kvarn_attn

    kvarn_attn._launch_kvarn_native_sinkhorn_writer(
        tail_key,
        tail_value,
        pool_slots,
        block_ids,
        packed_cache,
        iterations,
        ownership_unique=True,
    )


def _unpack_q4(packed: torch.Tensor) -> torch.Tensor:
    unpacked = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.uint8)
    unpacked[..., 0::2] = packed & 0xF
    unpacked[..., 1::2] = (packed >> 4) & 0xF
    return unpacked


def _unpack_dpas_k4(packed: torch.Tensor) -> torch.Tensor:
    count = packed.shape[0]
    physical = _unpack_q4(packed.reshape(count, 2, 4, 4, 8, 2, 32, 1))
    return physical.permute(0, 2, 6, 5, 1, 3, 7, 4).reshape(count, 256, 128)


def _unpack_dpas_v4(packed: torch.Tensor) -> torch.Tensor:
    count = packed.shape[0]
    physical = _unpack_q4(packed.reshape(count, 2, 8, 4, 8, 2, 2, 8, 1))
    return physical.permute(0, 1, 3, 7, 5, 2, 6, 8, 4).reshape(count, 128, 256)


def _dequant_records(records: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    flat = records.reshape(-1, RECORD_BYTES)
    count = flat.shape[0]
    key_q = _unpack_dpas_k4(flat[:, :K_S_COL].reshape(count, 256, 64)).float()
    key_scale = flat[:, K_S_COL:K_ZP].contiguous().view(torch.float16).float()
    key_zero = flat[:, K_ZP:K_S_ROW].contiguous().view(torch.float16).float()
    key_row = flat[:, K_S_ROW:V_PACKED].contiguous().view(torch.float16).float()
    key = (key_q * key_scale[..., None] + key_zero[..., None]) * key_row[:, None, :]

    value_q = _unpack_dpas_v4(
        flat[:, V_PACKED:V_S_COL].reshape(count, 128, 128)
    ).float()
    value_col = flat[:, V_S_COL:V_S_ROW].contiguous().view(torch.float16).float()
    value_scale = flat[:, V_S_ROW:V_ZP].contiguous().view(torch.float16).float()
    value_zero = flat[:, V_ZP:RECORD_BYTES].contiguous().view(torch.float16).float()
    value = (value_q * value_scale[..., None] + value_zero[..., None]) * value_col[
        :, None, :
    ]
    return key, value


def _mismatch_evidence(actual: torch.Tensor, expected: torch.Tensor) -> str:
    byte_mismatches = int(torch.count_nonzero(actual != expected))
    actual_key, actual_value = _dequant_records(actual)
    expected_key, expected_value = _dequant_records(expected)
    key_error = torch.nan_to_num((actual_key - expected_key).abs(), nan=float("inf"))
    value_error = torch.nan_to_num(
        (actual_value - expected_value).abs(), nan=float("inf")
    )
    return (
        f"byte_mismatches={byte_mismatches}; "
        f"dequant_key_max_abs={float(key_error.max())}; "
        f"dequant_key_mean_abs={float(key_error.mean())}; "
        f"dequant_value_max_abs={float(value_error.max())}; "
        f"dequant_value_mean_abs={float(value_error.mean())}"
    )


def _run_differential(
    tail_key_cpu: torch.Tensor,
    tail_value_cpu: torch.Tensor,
    pool_slots_cpu: torch.Tensor,
    block_ids_cpu: torch.Tensor,
    *,
    iterations: int = 16,
) -> None:
    tail_key = tail_key_cpu.xpu()
    tail_value = tail_value_cpu.xpu()
    original_key = tail_key.clone()
    original_value = tail_value.clone()
    pool_slots = pool_slots_cpu.xpu()
    block_ids = block_ids_cpu.xpu()
    cache_shape = (int(block_ids_cpu.max()) + 2, 4, RECORD_BYTES)
    control = torch.full(cache_shape, 0xA5, dtype=torch.uint8, device="xpu")
    candidate = torch.full_like(control, 0xA5)

    _write_production_control(
        tail_key, tail_value, pool_slots, block_ids, control, iterations
    )
    _write_fused_candidate(
        tail_key, tail_value, pool_slots, block_ids, candidate, iterations
    )
    torch.xpu.synchronize()

    actual = candidate.cpu()
    expected = control.cpu()
    selected_actual = actual.index_select(0, block_ids_cpu)
    selected_expected = expected.index_select(0, block_ids_cpu)
    assert torch.equal(actual, expected), _mismatch_evidence(
        selected_actual, selected_expected
    )
    assert torch.equal(tail_key, original_key)
    assert torch.equal(tail_value, original_value)


@pytest.mark.parametrize(
    "case",
    ["random-1", "random-17", "random-20260905", "constant", "extreme", "near-tie"],
)
def test_fused_writer_is_byte_identical_to_production_path(case: str) -> None:
    tail_key, tail_value = _tail_case(case)
    _run_differential(
        tail_key,
        tail_value,
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([5], dtype=torch.int64),
    )


def test_fused_writer_multi_block_iteration16_matches_production_path() -> None:
    tail_key, tail_value = _tail_case("random-41", pool_size=5)
    _run_differential(
        tail_key,
        tail_value,
        torch.tensor([4, 1, 3], dtype=torch.int64),
        torch.tensor([9, 2, 7], dtype=torch.int64),
    )


def test_fused_writer_empty_and_negative_iteration_contracts() -> None:
    tail_key_cpu, tail_value_cpu = _tail_case("random-3", pool_size=1)
    tail_key = tail_key_cpu.xpu()
    tail_value = tail_value_cpu.xpu()
    empty = torch.empty(0, dtype=torch.int64, device="xpu")
    zero = torch.zeros(1, dtype=torch.int64, device="xpu")
    cache = torch.full((1, 4, RECORD_BYTES), 0xA5, dtype=torch.uint8, device="xpu")

    _write_fused_candidate(tail_key, tail_value, empty, empty, cache, 16)
    with pytest.raises(RuntimeError, match="between 0 and 64"):
        _write_fused_candidate(tail_key, tail_value, zero, zero, cache, -1)
    torch.xpu.synchronize()
    assert torch.all(cache == 0xA5)
    assert torch.equal(tail_key.cpu(), tail_key_cpu)
    assert torch.equal(tail_value.cpu(), tail_value_cpu)


def test_fused_writer_records_temporary_allocator_lifetimes() -> None:
    tail_key_cpu, tail_value_cpu = _tail_case("random-73", pool_size=2)
    slots_cpu = torch.tensor([1], dtype=torch.int64)
    blocks_cpu = torch.tensor([3], dtype=torch.int64)
    control = torch.full((5, 4, RECORD_BYTES), 0xA5, dtype=torch.uint8, device="xpu")
    control_key = tail_key_cpu.xpu()
    control_value = tail_value_cpu.xpu()
    _write_production_control(
        control_key,
        control_value,
        slots_cpu.xpu(),
        blocks_cpu.xpu(),
        control,
        16,
    )
    torch.xpu.synchronize()

    candidate = torch.full_like(control, 0xA5)

    def enqueue_from_temporaries() -> None:
        _write_fused_candidate(
            tail_key_cpu.xpu(),
            tail_value_cpu.xpu(),
            slots_cpu.xpu(),
            blocks_cpu.xpu(),
            candidate,
            16,
        )

    enqueue_from_temporaries()
    # Encourage reuse immediately after every source/index tensor loses its
    # Python owner. recordStream must keep all allocations alive until the
    # custom queue work completes.
    churn = [
        torch.full(tail_key_cpu.shape, index, dtype=torch.float16, device="xpu")
        for index in range(8)
    ]
    torch.xpu.synchronize()
    assert len(churn) == 8
    actual = candidate.cpu()
    expected = control.cpu()
    assert torch.equal(actual, expected), _mismatch_evidence(actual[3:4], expected[3:4])
