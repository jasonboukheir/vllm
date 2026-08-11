"""XPU writer-to-Triton-reader contracts for KVarN record strides."""

from __future__ import annotations

import os

import pytest
import torch

from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
from vllm.v1.attention.backends.kvarn_attn import _sinkhorn_pack_kv
from vllm.v1.attention.ops.triton_kvarn_decode import (
    _kvarn_build_packed_kv_kernel,
    _kvarn_scatter_store_kernel,
)
from vllm.v1.attention.ops.triton_kvarn_flush import (
    _kvarn_flush_record_kernel,
)

D = 256
G = 128
HK = 4
SEQUENCE_BOUNDARIES = (127, 128, 129, 255, 256, 257)
XPU_TRITON_AVAILABLE = torch.xpu.is_available() and hasattr(
    _kvarn_scatter_store_kernel, "__getitem__"
)


@pytest.mark.skipif(not XPU_TRITON_AVAILABLE, reason="XPU Triton is required")
def test_scatter_store_rejects_stale_high_pool_slot() -> None:
    """A stale lookup must not write beyond the sparse tail pool."""
    guard = 2
    pool_size = 3
    backing_k = torch.full(
        (pool_size + 2 * guard, G, HK, D),
        123,
        dtype=torch.float16,
        device="xpu",
    )
    backing_v = backing_k.clone()
    pool_k = backing_k[guard : guard + pool_size]
    pool_v = backing_v[guard : guard + pool_size]
    k = torch.ones((1, HK, D), dtype=torch.float16, device="xpu")
    v = torch.full_like(k, 2)
    slot_mapping = torch.tensor([0], dtype=torch.int64, device="xpu")
    # Deliberately points one element past the tail pool.
    block_to_slot = torch.tensor([pool_size], dtype=torch.int32, device="xpu")

    _kvarn_scatter_store_kernel[(1, HK)](
        k,
        v,
        slot_mapping,
        block_to_slot,
        pool_k,
        pool_v,
        k.stride(0),
        k.stride(1),
        pool_k.stride(0),
        pool_k.stride(1),
        pool_k.stride(2),
        GROUP=G,
        D=D,
        NUM_BLOCKS_LOOKUP=1,
        POOL_SIZE=pool_size,
        num_warps=2,
        num_stages=2,
    )
    torch.xpu.synchronize()
    assert bool((backing_k == 123).all().item())
    assert bool((backing_v == 123).all().item())


def _packed_low_nibble_first(values: torch.Tensor) -> torch.Tensor:
    return values[..., 0::2] | (values[..., 1::2] << 4)


def _record_fields(num_blocks: int) -> tuple[torch.Tensor, ...]:
    """Create coordinate-coded logical K/V records with exact unit scales."""
    m = num_blocks * HK
    block = torch.arange(num_blocks)[:, None, None, None]
    head = torch.arange(HK)[None, :, None, None]
    dim = torch.arange(D)[None, None, :, None]
    token = torch.arange(G)[None, None, None, :]
    qk = ((11 * block + 7 * head + 3 * dim + token) & 15).byte()
    qv = ((5 * block + 13 * head + dim + 9 * token) & 15).byte()
    qk = qk.expand(num_blocks, HK, D, G).reshape(m, D, G)
    qv = qv.permute(0, 1, 3, 2).expand(num_blocks, HK, G, D)
    qv = qv.reshape(m, G, D)

    ones_d = torch.ones((m, D), dtype=torch.float16)
    zeros_d = torch.zeros_like(ones_d)
    ones_g = torch.ones((m, G), dtype=torch.float16)
    zeros_g = torch.zeros_like(ones_g)
    return (
        _packed_low_nibble_first(qk).contiguous().xpu(),
        ones_d.contiguous().view(torch.uint8).xpu(),
        zeros_d.contiguous().view(torch.uint8).xpu(),
        ones_g.contiguous().view(torch.uint8).xpu(),
        _packed_low_nibble_first(qv).contiguous().xpu(),
        ones_d.contiguous().view(torch.uint8).xpu(),
        ones_g.contiguous().view(torch.uint8).xpu(),
        zeros_g.contiguous().view(torch.uint8).xpu(),
        qk,
        qv,
    )


def _native_reference(
    query: torch.Tensor, qk: torch.Tensor, qv: torch.Tensor, seq_len: int
) -> torch.Tensor:
    keys = (
        qk.view(-1, HK, D, G).permute(0, 3, 1, 2).reshape(-1, HK, D)[:seq_len].float()
    )
    values = (
        qv.view(-1, HK, G, D).permute(0, 2, 1, 3).reshape(-1, HK, D)[:seq_len].float()
    )
    kv_heads = torch.arange(24) // 6
    scores = torch.einsum("hd,thd->ht", query[0].float(), keys[:, kv_heads]) / 16.0
    return torch.einsum(
        "ht,thd->hd", torch.softmax(scores, dim=-1), values[:, kv_heads]
    )


def _native_decode(
    query: torch.Tensor,
    cache: torch.Tensor,
    block_table: torch.Tensor,
    lengths: torch.Tensor,
    lookup: torch.Tensor,
    tail_k: torch.Tensor,
    tail_v: torch.Tensor,
    output: torch.Tensor,
    max_seq_len: int,
) -> None:
    """Call the native API using the split mode selected by the test."""
    splits = int(os.environ.get("KVARN_NATIVE_XPU_SPLITS", "1"))
    batch = query.shape[0]
    # Use enough surrounding storage to catch writes whose bad batch/head
    # stride skips well beyond one output frame.  The one-split native path
    # writes its mainloop result directly and therefore needs its own canary;
    # guarding only the split scratch tensors misses that class of corruption.
    guard = 1 << 20

    def guarded(shape: tuple[int, ...], dtype: torch.dtype):
        elements = torch.Size(shape).numel()
        storage = torch.full((elements + 2 * guard,), 123.0, dtype=dtype, device="xpu")
        return storage[guard : guard + elements].view(shape), storage

    guarded_output, output_storage = guarded(tuple(output.shape), output.dtype)
    temp, temp_storage = guarded((batch, 24 * splits, D), torch.float16)
    sums, sums_storage = guarded((batch, 24, splits), torch.float32)
    maxima, maxima_storage = guarded((batch, 24, splits), torch.float32)
    torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
        query,
        cache,
        block_table,
        lengths,
        lookup,
        tail_k,
        tail_v,
        temp,
        sums,
        maxima,
        guarded_output,
        max_seq_len,
        1.0 / 16.0,
    )
    torch.xpu.synchronize()
    output.copy_(guarded_output)
    for storage in (output_storage, temp_storage, sums_storage, maxima_storage):
        assert bool((storage[:guard] == 123).all().item())
        assert bool((storage[-guard:] == 123).all().item())


@pytest.mark.skipif(not XPU_TRITON_AVAILABLE, reason="XPU Triton is required")
@pytest.mark.parametrize(
    ("cache_dtype", "stride"),
    [
        ("kvarn_k4v4_g128", 65536),
        ("kvarn_k4v4_g128_compact", 35072),
    ],
)
@pytest.mark.parametrize("dpas_layout", [False, True])
def test_flush_writer_to_materialize_reader_at_page_boundaries(
    cache_dtype: str, stride: int, dpas_layout: bool
) -> None:
    cfg = KVarNConfig.from_cache_dtype(cache_dtype, head_dim=D)
    assert cfg.record_bytes == stride
    physical_blocks = 5
    logical_blocks = 3
    block_ids_cpu = torch.tensor([3, 0, 4], dtype=torch.int64)
    fields = _record_fields(logical_blocks)
    *device_fields, qk, qv = fields
    cache = torch.full(
        (physical_blocks, HK, stride), 0xA7, dtype=torch.uint8, device="xpu"
    )
    block_ids = block_ids_cpu.xpu()
    m = logical_blocks * HK
    flush_block = 256
    num_chunks = (stride + flush_block - 1) // flush_block
    _kvarn_flush_record_kernel[(m, num_chunks)](
        *device_fields,
        block_ids,
        cache,
        cache.stride(0),
        cache.stride(1),
        K_PACKED_BYTES=cfg.k_packed_bytes,
        V_PACKED_OFFSET=cfg.v_packed_offset,
        V_PACKED_BYTES=cfg.v_packed_bytes,
        K_SCOL_OFFSET=cfg.k_s_col_offset,
        K_ZP_OFFSET=cfg.k_zp_offset,
        K_SROW_OFFSET=cfg.k_s_row_offset,
        V_SCOL_OFFSET=cfg.v_s_col_offset,
        V_SROW_OFFSET=cfg.v_s_row_offset,
        V_ZP_OFFSET=cfg.v_zp_offset,
        RECORD_BYTES=cfg.tile_bytes,
        TILE_BYTES=cfg.record_bytes,
        NUM_HEADS=HK,
        NATIVE_DPAS_LAYOUT=dpas_layout,
        BLOCK=flush_block,
        num_warps=4,
    )

    # Writer must not cross an adjacent physical block or head record.
    torch.xpu.synchronize()
    untouched = cache[[1, 2]].cpu()
    assert bool((untouched == 0xA7).all())
    if not dpas_layout:
        for logical_block, physical_block in enumerate(block_ids_cpu.tolist()):
            for head in range(HK):
                record = cache[physical_block, head].cpu()
                field = logical_block * HK + head
                torch.testing.assert_close(
                    record[: cfg.k_packed_bytes],
                    device_fields[0][field].flatten().cpu(),
                )
                torch.testing.assert_close(
                    record[cfg.k_s_col_offset : cfg.k_zp_offset],
                    device_fields[1][field].flatten().cpu(),
                )

    block_table = block_ids_cpu.to(torch.int32).view(1, logical_blocks).xpu()
    block_to_slot = torch.full((physical_blocks,), -1, dtype=torch.int32, device="xpu")
    tail_k = torch.zeros((1, G, HK, D), dtype=torch.float16, device="xpu")
    tail_v = torch.zeros_like(tail_k)
    for seq_len in SEQUENCE_BOUNDARIES:
        seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="xpu")
        cu = torch.tensor([0, seq_len], dtype=torch.int32, device="xpu")
        out_k = torch.full((seq_len, HK, D), -1, dtype=torch.float16, device="xpu")
        out_v = torch.full_like(out_k, -1)
        _kvarn_build_packed_kv_kernel[(logical_blocks, HK)](
            block_table,
            seq_lens,
            cu,
            block_to_slot,
            cache,
            tail_k,
            tail_v,
            out_k,
            out_v,
            block_table.stride(0),
            cache.stride(0),
            cache.stride(1),
            tail_k.stride(0),
            tail_k.stride(1),
            tail_k.stride(2),
            out_k.stride(0),
            out_k.stride(1),
            MAX_BLOCKS_PER_REQ=logical_blocks,
            D=D,
            GROUP=G,
            K_BITS=4,
            V_BITS=4,
            NUM_BLOCKS_LOOKUP=physical_blocks,
            K_PACKED_OFFSET=cfg.k_packed_offset,
            K_S_COL_OFFSET=cfg.k_s_col_offset,
            K_ZP_OFFSET=cfg.k_zp_offset,
            K_S_ROW_OFFSET=cfg.k_s_row_offset,
            V_PACKED_OFFSET=cfg.v_packed_offset,
            V_S_COL_OFFSET=cfg.v_s_col_offset,
            V_S_ROW_OFFSET=cfg.v_s_row_offset,
            V_ZP_OFFSET=cfg.v_zp_offset,
            DPAS_LAYOUT=dpas_layout,
            num_warps=4,
            num_stages=2,
        )
        expected_k = (
            qk.view(logical_blocks, HK, D, G)
            .permute(0, 3, 1, 2)
            .reshape(logical_blocks * G, HK, D)[:seq_len]
        )
        expected_v = (
            qv.view(logical_blocks, HK, G, D)
            .permute(0, 2, 1, 3)
            .reshape(logical_blocks * G, HK, D)[:seq_len]
        )
        torch.testing.assert_close(out_k.cpu(), expected_k.half(), rtol=0, atol=0)
        torch.testing.assert_close(out_v.cpu(), expected_v.half(), rtol=0, atol=0)


@pytest.mark.skipif(not XPU_TRITON_AVAILABLE, reason="XPU Triton is required")
@pytest.mark.parametrize(
    ("cache_dtype", "stride"),
    [
        ("kvarn_k4v4_g128_compact", 35072),
        ("kvarn_k4v4_g128", 65536),
    ],
)
@pytest.mark.parametrize("seq_len", [127, 128, 129, 255, 256, 257])
def test_dpas_flush_writer_to_native_reader_matches_fp32_oracle(
    monkeypatch: pytest.MonkeyPatch,
    cache_dtype: str,
    stride: int,
    seq_len: int,
) -> None:
    """Exercise the production DPAS writer directly through the Xe2 reader."""
    if not hasattr(torch.ops._vllm_fa2_C, "kvarn_decode"):
        pytest.skip("the native KVarN decoder is not loaded")
    monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "1")
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "1")
    cfg = KVarNConfig.from_cache_dtype(cache_dtype, head_dim=D)
    assert cfg.record_bytes == stride
    physical_blocks = 5
    logical_blocks = 3
    block_ids_cpu = torch.tensor([3, 0, 4], dtype=torch.int64)
    *device_fields, qk, qv = _record_fields(logical_blocks)
    cache = torch.full(
        (physical_blocks, HK, stride), 0xA7, dtype=torch.uint8, device="xpu"
    )
    m = logical_blocks * HK
    flush_block = 256
    _kvarn_flush_record_kernel[(m, (stride + flush_block - 1) // flush_block)](
        *device_fields,
        block_ids_cpu.xpu(),
        cache,
        cache.stride(0),
        cache.stride(1),
        K_PACKED_BYTES=cfg.k_packed_bytes,
        V_PACKED_OFFSET=cfg.v_packed_offset,
        V_PACKED_BYTES=cfg.v_packed_bytes,
        K_SCOL_OFFSET=cfg.k_s_col_offset,
        K_ZP_OFFSET=cfg.k_zp_offset,
        K_SROW_OFFSET=cfg.k_s_row_offset,
        V_SCOL_OFFSET=cfg.v_s_col_offset,
        V_SROW_OFFSET=cfg.v_s_row_offset,
        V_ZP_OFFSET=cfg.v_zp_offset,
        RECORD_BYTES=cfg.tile_bytes,
        TILE_BYTES=cfg.record_bytes,
        NUM_HEADS=HK,
        NATIVE_DPAS_LAYOUT=True,
        BLOCK=flush_block,
        num_warps=4,
    )
    # Triton and the out-of-tree SYCL extension currently use distinct queue
    # wrappers.  Make the producer/consumer boundary explicit in this direct
    # integration test; production must replace this with an event dependency.
    torch.xpu.synchronize()
    generator = torch.Generator().manual_seed(20260809 + seq_len)
    query_cpu = torch.randn((1, 24, D), generator=generator).half()
    output = torch.empty_like(query_cpu, device="xpu")
    block_table = block_ids_cpu.to(torch.int32).view(1, -1).xpu()
    tail_k = torch.zeros((1, G, HK, D), dtype=torch.float16, device="xpu")
    tail_v = torch.zeros_like(tail_k)
    _native_decode(
        query_cpu.xpu(),
        cache,
        block_table,
        torch.tensor([seq_len], dtype=torch.int32, device="xpu"),
        torch.full((physical_blocks,), -1, dtype=torch.int32, device="xpu"),
        tail_k,
        tail_v,
        output,
        seq_len,
    )
    torch.xpu.synchronize()
    reference = _native_reference(query_cpu, qk, qv, seq_len)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.cpu()[0].float(), reference, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(not XPU_TRITON_AVAILABLE, reason="XPU Triton is required")
@pytest.mark.parametrize("stride", [35072, 65536])
@pytest.mark.parametrize("zero_query", [True, False])
@pytest.mark.parametrize("hybrid_tail", [False, True])
def test_sinkhorn_quantized_dpas_writer_to_native_reader_matches_materialized(
    monkeypatch: pytest.MonkeyPatch,
    stride: int,
    zero_query: bool,
    hybrid_tail: bool,
) -> None:
    """Cover the real factor metadata that unit-scale writer tests omit."""
    if not hasattr(torch.ops._vllm_fa2_C, "kvarn_decode"):
        pytest.skip("the native KVarN decoder is not loaded")
    monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "1")
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "1")
    cache_dtype = "kvarn_k4v4_g128_compact" if stride == 35072 else "kvarn_k4v4_g128"
    cfg = KVarNConfig.from_cache_dtype(cache_dtype, head_dim=D)
    blocks = 2 if hybrid_tail else 3
    m = blocks * HK
    generator = torch.Generator().manual_seed(46613)
    key = torch.randn((m, D, G), generator=generator).xpu()
    value = torch.randn((m, G, D), generator=generator).xpu()
    k_out, v_out = _sinkhorn_pack_kv(key, value, cfg)
    fields = (
        k_out["q_packed_uint8"],
        k_out["s_col_K"].contiguous().view(torch.uint8),
        k_out["zp_K"].contiguous().view(torch.uint8),
        k_out["s_row_K"].contiguous().view(torch.uint8),
        v_out["q_packed_uint8"],
        v_out["s_col_V"].contiguous().view(torch.uint8),
        v_out["s_row_V"].contiguous().view(torch.uint8),
        v_out["zp_V"].contiguous().view(torch.uint8),
    )
    cache = torch.empty((blocks, HK, stride), dtype=torch.uint8, device="xpu")
    flush_block = 1024
    _kvarn_flush_record_kernel[(m, (stride + flush_block - 1) // flush_block)](
        *fields,
        torch.arange(blocks, device="xpu"),
        cache,
        cache.stride(0),
        cache.stride(1),
        K_PACKED_BYTES=cfg.k_packed_bytes,
        V_PACKED_OFFSET=cfg.v_packed_offset,
        V_PACKED_BYTES=cfg.v_packed_bytes,
        K_SCOL_OFFSET=cfg.k_s_col_offset,
        K_ZP_OFFSET=cfg.k_zp_offset,
        K_SROW_OFFSET=cfg.k_s_row_offset,
        V_SCOL_OFFSET=cfg.v_s_col_offset,
        V_SROW_OFFSET=cfg.v_s_row_offset,
        V_ZP_OFFSET=cfg.v_zp_offset,
        RECORD_BYTES=cfg.tile_bytes,
        TILE_BYTES=cfg.record_bytes,
        NUM_HEADS=HK,
        NATIVE_DPAS_LAYOUT=True,
        BLOCK=flush_block,
        num_warps=8,
    )
    torch.xpu.synchronize()
    seq_len = 129 if hybrid_tail else 257
    table = torch.arange(blocks, dtype=torch.int32, device="xpu").view(1, -1)
    lengths = torch.tensor([seq_len], dtype=torch.int32, device="xpu")
    lookup = torch.full((blocks,), -1, dtype=torch.int32, device="xpu")
    tail_k = torch.zeros((1, G, HK, D), dtype=torch.float16, device="xpu")
    tail_v = torch.zeros_like(tail_k)
    if hybrid_tail:
        lookup[-1] = 0
        tail_k.normal_()
        tail_v.normal_()
    query_cpu = (
        torch.zeros((1, 24, D), dtype=torch.float16)
        if zero_query
        else torch.randn((1, 24, D), generator=generator).half()
    )
    native = torch.empty_like(query_cpu, device="xpu")
    query = query_cpu.xpu()
    readonly = {
        "query": (query, query.clone()),
        "cache": (cache, cache.clone()),
        "table": (table, table.clone()),
        "lengths": (lengths, lengths.clone()),
        "lookup": (lookup, lookup.clone()),
        "tail_k": (tail_k, tail_k.clone()),
        "tail_v": (tail_v, tail_v.clone()),
    }
    _native_decode(
        query,
        cache,
        table,
        lengths,
        lookup,
        tail_k,
        tail_v,
        native,
        seq_len,
    )
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="xpu")
    materialized_k = torch.empty((seq_len, HK, D), dtype=torch.float16, device="xpu")
    materialized_v = torch.empty_like(materialized_k)
    _kvarn_build_packed_kv_kernel[(blocks, HK)](
        table,
        lengths,
        cu,
        lookup,
        cache,
        tail_k,
        tail_v,
        materialized_k,
        materialized_v,
        table.stride(0),
        cache.stride(0),
        cache.stride(1),
        tail_k.stride(0),
        tail_k.stride(1),
        tail_k.stride(2),
        materialized_k.stride(0),
        materialized_k.stride(1),
        MAX_BLOCKS_PER_REQ=blocks,
        D=D,
        GROUP=G,
        K_BITS=4,
        V_BITS=4,
        NUM_BLOCKS_LOOKUP=blocks,
        K_PACKED_OFFSET=cfg.k_packed_offset,
        K_S_COL_OFFSET=cfg.k_s_col_offset,
        K_ZP_OFFSET=cfg.k_zp_offset,
        K_S_ROW_OFFSET=cfg.k_s_row_offset,
        V_PACKED_OFFSET=cfg.v_packed_offset,
        V_S_COL_OFFSET=cfg.v_s_col_offset,
        V_S_ROW_OFFSET=cfg.v_s_row_offset,
        V_ZP_OFFSET=cfg.v_zp_offset,
        DPAS_LAYOUT=True,
        num_warps=4,
        num_stages=2,
    )
    kv_heads = torch.arange(24, device="xpu") // 6
    scores = (
        torch.einsum(
            "hd,thd->ht",
            query_cpu[0].xpu().float(),
            materialized_k[:, kv_heads].float(),
        )
        / 16.0
    )
    reference = torch.einsum(
        "ht,thd->hd",
        torch.softmax(scores, dim=-1),
        materialized_v[:, kv_heads].float(),
    )
    torch.xpu.synchronize()
    assert torch.isfinite(native).all()
    torch.testing.assert_close(
        native.cpu()[0].float(), reference.cpu(), atol=3e-2, rtol=3e-2
    )
    for name, (actual, expected) in readonly.items():
        torch.testing.assert_close(actual.cpu(), expected.cpu(), msg=name)


@pytest.mark.skipif(not XPU_TRITON_AVAILABLE, reason="XPU Triton is required")
@pytest.mark.parametrize("stride", [35072, 65536])
@pytest.mark.parametrize("seq_len", [127, 128, 129, 257])
def test_native_reader_matches_nonzero_hybrid_tail_pool(
    monkeypatch: pytest.MonkeyPatch, stride: int, seq_len: int
) -> None:
    """Cover positive block-to-slot entries used by live hybrid pages."""
    if not hasattr(torch.ops._vllm_fa2_C, "kvarn_decode"):
        pytest.skip("the native KVarN decoder is not loaded")
    monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "1")
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "1")
    blocks = (seq_len + G - 1) // G
    generator = torch.Generator().manual_seed(46812 + seq_len)
    tail_k_cpu = torch.randn((blocks, G, HK, D), generator=generator).half()
    tail_v_cpu = torch.randn((blocks, G, HK, D), generator=generator).half()
    tail_k = tail_k_cpu.xpu()
    tail_v = tail_v_cpu.xpu()
    query_cpu = torch.randn((1, 24, D), generator=generator).half()
    guard = 4096
    output_storage = torch.full(
        (query_cpu.numel() + 2 * guard,), 123.0, dtype=torch.float16, device="xpu"
    )
    output = output_storage[guard : guard + query_cpu.numel()].view_as(query_cpu)
    cache = torch.zeros((blocks, HK, stride), dtype=torch.uint8, device="xpu")
    table = torch.arange(blocks, dtype=torch.int32, device="xpu").view(1, -1)
    lookup = torch.arange(blocks, dtype=torch.int32, device="xpu")
    query = query_cpu.xpu()
    lengths = torch.tensor([seq_len], dtype=torch.int32, device="xpu")
    readonly = {
        "query": (query, query.clone()),
        "cache": (cache, cache.clone()),
        "table": (table, table.clone()),
        "lengths": (lengths, lengths.clone()),
        "lookup": (lookup, lookup.clone()),
        "tail_k": (tail_k, tail_k.clone()),
        "tail_v": (tail_v, tail_v.clone()),
    }
    _native_decode(
        query,
        cache,
        table,
        lengths,
        lookup,
        tail_k,
        tail_v,
        output,
        seq_len,
    )
    torch.xpu.synchronize()
    keys = tail_k_cpu.reshape(-1, HK, D)[:seq_len].float()
    values = tail_v_cpu.reshape(-1, HK, D)[:seq_len].float()
    kv_heads = torch.arange(24) // 6
    scores = torch.einsum("hd,thd->ht", query_cpu[0].float(), keys[:, kv_heads]) / 16
    reference = torch.einsum(
        "ht,thd->hd", torch.softmax(scores, dim=-1), values[:, kv_heads]
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.cpu()[0].float(), reference, atol=3e-2, rtol=3e-2)
    assert bool((output_storage[:guard] == 123).all().item())
    assert bool((output_storage[-guard:] == 123).all().item())
    for name, (actual, expected) in readonly.items():
        torch.testing.assert_close(actual.cpu(), expected.cpu(), msg=name)
