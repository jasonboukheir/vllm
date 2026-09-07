# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused log-domain iterative variance-normalization for KVarN.

Matches the PyTorch reference in
``vllm/model_executor/layers/quantization/kvarn/sinkhorn.py`` semantically —
same 16 alternating col/row std-normalization passes, same best-so-far
tracking via the imbalance metric, same clamps. One Triton program per
``[R, C]`` tile; the grid dim is the number of tiles in the batch.

For ``R = C = 128`` the full tile is 64 KB fp32 — fits in a single Triton
block's register/SMEM budget on current GPUs.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

_CLIP_STD_MIN = 1e-3
_CLIP_STD_MAX = 1e3
_LOG_S_MIN = -0.3
_LOG_S_MAX = 10.0


@triton.jit
def _sinkhorn_log_kernel(
    Tile_ptr,  # [N, R, C] fp32 input — rotated tile
    Balanced_ptr,  # [N, R, C] fp32 output
    SCol_ptr,  # [N, C] fp32 output (s_col, per-column)
    SRow_ptr,  # [N, R] fp32 output (s_row, per-row)
    # Strides
    stride_tn,
    stride_tr,
    stride_bn,
    stride_br,
    stride_sc_n,
    stride_sr_n,
    # Dims
    R: tl.constexpr,
    C: tl.constexpr,
    ITERATIONS: tl.constexpr,
    # Algorithm params (kept as tl.constexpr for the compiler)
    CLIP_STD_MIN: tl.constexpr,
    CLIP_STD_MAX: tl.constexpr,
    LOG_S_MIN: tl.constexpr,
    LOG_S_MAX: tl.constexpr,
    FP16_POOL_INPUT: tl.constexpr = False,
):
    """One program per tile. Loads a R x C tile into registers, does
    ``ITERATIONS`` alternating col/row log-domain normalizations, tracks
    the best-so-far (lowest-imbalance) scales, and writes (balanced, s_col,
    s_row).
    """
    pid = tl.program_id(0)

    r_offs = tl.arange(0, R)
    c_offs = tl.arange(0, C)

    # Load tile [R, C] into registers
    tile_base = pid * stride_tn
    tile_ptrs = Tile_ptr + tile_base + r_offs[:, None] * stride_tr + c_offs[None, :]
    tile = tl.load(tile_ptrs).to(tl.float32)
    if FP16_POOL_INPUT:
        # Only the validated fused pool path can assert FP16 provenance.
        # Keep the immutable input narrow; all normalization remains FP32.
        tile = tile.to(tl.float16)

    # log_s_col [C], log_s_row [R]; initialised at zero (exp = 1)
    log_s_col = tl.zeros([C], dtype=tl.float32)
    log_s_row = tl.zeros([R], dtype=tl.float32)

    # cur = tile / s_col / s_row = tile (with mu = 1 initially)
    cur = tile.to(tl.float32)

    # ── initial imbalance + best snapshot ─────────────────────────────────
    col_mean0 = tl.sum(cur, axis=0) / R
    col_var0 = tl.sum(cur * cur, axis=0) / R - col_mean0 * col_mean0
    col_std0 = tl.sqrt(tl.maximum(col_var0 * R / (R - 1), 0.0))
    row_mean0 = tl.sum(cur, axis=1) / C
    row_var0 = tl.sum(cur * cur, axis=1) / C - row_mean0 * row_mean0
    row_std0 = tl.sqrt(tl.maximum(row_var0 * C / (C - 1), 0.0))

    col_max0 = tl.max(col_std0)
    col_min0 = tl.maximum(tl.min(col_std0), 1e-8)
    row_max0 = tl.max(row_std0)
    row_min0 = tl.maximum(tl.min(row_std0), 1e-8)
    imb_best = col_max0 / col_min0 + row_max0 / row_min0

    sc_best = tl.exp(log_s_col)  # ones[C]
    sr_best = tl.exp(log_s_row)  # ones[R]

    # ── iterations ────────────────────────────────────────────────────────
    for _ in tl.static_range(ITERATIONS):
        # Update column scales from cur's per-column std
        col_mean = tl.sum(cur, axis=0) / R
        col_var = tl.sum(cur * cur, axis=0) / R - col_mean * col_mean
        col_std = tl.sqrt(tl.maximum(col_var * R / (R - 1), 0.0))
        col_std_clipped = tl.maximum(tl.minimum(col_std, CLIP_STD_MAX), CLIP_STD_MIN)
        log_s_col = log_s_col + tl.log(col_std_clipped)
        log_s_col = tl.maximum(tl.minimum(log_s_col, LOG_S_MAX), LOG_S_MIN)
        s_col_lin = tl.exp(log_s_col)
        s_row_lin = tl.exp(log_s_row)
        if FP16_POOL_INPUT:
            # Explicit reloads shorten the live FP32 input lifetime.
            cur = (
                tl.load(tile_ptrs, volatile=True).to(tl.float32)
                / s_col_lin[None, :]
                / s_row_lin[:, None]
            )
        else:
            cur = tile / s_col_lin[None, :] / s_row_lin[:, None]

        # Update row scales from new cur's per-row std
        row_mean = tl.sum(cur, axis=1) / C
        row_var = tl.sum(cur * cur, axis=1) / C - row_mean * row_mean
        row_std = tl.sqrt(tl.maximum(row_var * C / (C - 1), 0.0))
        row_std_clipped = tl.maximum(tl.minimum(row_std, CLIP_STD_MAX), CLIP_STD_MIN)
        log_s_row = log_s_row + tl.log(row_std_clipped)
        log_s_row = tl.maximum(tl.minimum(log_s_row, LOG_S_MAX), LOG_S_MIN)
        s_col_lin = tl.exp(log_s_col)
        s_row_lin = tl.exp(log_s_row)
        if FP16_POOL_INPUT:
            # Explicit reloads shorten the live FP32 input lifetime.
            cur = (
                tl.load(tile_ptrs, volatile=True).to(tl.float32)
                / s_col_lin[None, :]
                / s_row_lin[:, None]
            )
        else:
            cur = tile / s_col_lin[None, :] / s_row_lin[:, None]

        # Imbalance + best-so-far update
        col_mean_n = tl.sum(cur, axis=0) / R
        col_var_n = tl.sum(cur * cur, axis=0) / R - col_mean_n * col_mean_n
        col_std_n = tl.sqrt(tl.maximum(col_var_n * R / (R - 1), 0.0))
        row_mean_n = tl.sum(cur, axis=1) / C
        row_var_n = tl.sum(cur * cur, axis=1) / C - row_mean_n * row_mean_n
        row_std_n = tl.sqrt(tl.maximum(row_var_n * C / (C - 1), 0.0))
        col_max_n = tl.max(col_std_n)
        col_min_n = tl.maximum(tl.min(col_std_n), 1e-8)
        row_max_n = tl.max(row_std_n)
        row_min_n = tl.maximum(tl.min(row_std_n), 1e-8)
        imb = col_max_n / col_min_n + row_max_n / row_min_n

        better = imb <= imb_best
        sc_best = tl.where(better, s_col_lin, sc_best)
        sr_best = tl.where(better, s_row_lin, sr_best)
        imb_best = tl.where(better, imb, imb_best)

    # ── final: balanced = tile / sc_best / sr_best, write outputs ─────────
    balanced = tile.to(tl.float32) / sc_best[None, :] / sr_best[:, None]
    bal_ptrs = (
        Balanced_ptr + pid * stride_bn + r_offs[:, None] * stride_br + c_offs[None, :]
    )
    tl.store(bal_ptrs, balanced)
    tl.store(SCol_ptr + pid * stride_sc_n + c_offs, sc_best)
    tl.store(SRow_ptr + pid * stride_sr_n + r_offs, sr_best)


def kvarn_sinkhorn_triton(
    tiles: torch.Tensor,
    iterations: int = 16,
    *,
    _fp16_pool_input: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton driver for ``_sinkhorn_log_kernel``.

    Args:
        tiles: ``[N, R, C]`` fp32 (or any real dtype, cast inside). Both R
            and C must be compile-time-constant power-of-2 values; we hard-
            code R = C = 128 for the first PR.
        iterations: number of alternating col/row passes (default 16).

    Returns:
        balanced: ``[N, R, C]`` fp32.
        s_col:    ``[N, C]`` fp32.
        s_row:    ``[N, R]`` fp32.
    """
    assert tiles.ndim == 3
    N, R, C = tiles.shape
    tiles = tiles.contiguous().to(torch.float32)
    device = tiles.device

    # The Triton kernel loads the WHOLE [R, C] tile into one program's registers
    # and unrolls the iteration loop. At large head_dim that tile is huge (e.g.
    # head_dim 512 -> [512, 128] = 256 KB) and the Triton compiler hangs/explodes
    # (128/256 compile fine). Route large tiles to the batched PyTorch Sinkhorn
    # (identical algorithm). Flush is infrequent + off the decode hot path, so the
    # cost is fine; head_dim<=256 keeps the fast kernel.
    if max(R, C) > 256:
        from vllm.model_executor.layers.quantization.kvarn.sinkhorn import (
            variance_normalize_batched,
        )

        bal, s_col_b, s_row_b = variance_normalize_batched(tiles, iterations=iterations)
        return (
            bal.contiguous(),
            s_col_b.reshape(N, C).contiguous(),
            s_row_b.reshape(N, R).contiguous(),
        )

    balanced = torch.empty(N, R, C, dtype=torch.float32, device=device)
    s_col = torch.empty(N, C, dtype=torch.float32, device=device)
    s_row = torch.empty(N, R, dtype=torch.float32, device=device)

    # The public/general FP32 path must never narrow arbitrary input values.
    # Enable only for the eight-iteration XPU beta shapes, with provenance
    # supplied by the validated fused-pool entry point below.
    fp16_pool_input = (
        _fp16_pool_input
        and device.type == "xpu"
        and iterations == 8
        and (R, C) in ((256, 128), (128, 256))
    )
    _sinkhorn_log_kernel[(N,)](
        tiles,
        balanced,
        s_col,
        s_row,
        tiles.stride(0),
        tiles.stride(1),
        balanced.stride(0),
        balanced.stride(1),
        s_col.stride(0),
        s_row.stride(0),
        R=R,
        C=C,
        ITERATIONS=iterations,
        CLIP_STD_MIN=_CLIP_STD_MIN,
        CLIP_STD_MAX=_CLIP_STD_MAX,
        LOG_S_MIN=_LOG_S_MIN,
        LOG_S_MAX=_LOG_S_MAX,
        FP16_POOL_INPUT=fp16_pool_input,
        # num_warps=8, not 4: the program keeps the whole [R, C] fp32 tile (plus
        # a working copy) live, so at 4 warps the per-thread footprint is several
        # KB of registers -> the compiler spills to CUDA local memory, and the
        # driver permanently reserves local_bytes x max_threads x num_SMs of
        # device memory for the context (~2 GiB on a 188-SM part for the
        # [256, 128] tile; a missing-KV-capacity component). 8 warps
        # halves the per-thread footprint: ~70% less reserved local memory AND
        # ~4x faster flush (the spills were also the kernel's bottleneck).
        # Balanced-tile output is unchanged within fp32 reduction noise (~5e-7
        # rel); 16 warps saves a bit more memory but is 2x slower than 8.
        num_warps=8,
        num_stages=2,
    )
    return balanced, s_col, s_row


@triton.jit
def _sinkhorn_pool_materialize_kernel(
    TailKey_ptr,
    TailValue_ptr,
    PoolSlots_ptr,
    KeyTiles_ptr,
    ValueTiles_ptr,
    stride_pp,
    stride_pg,
    stride_ph,
    stride_pd,
    NUM_KV_HEADS: tl.constexpr,
    GROUP: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_G: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Gather, cast, and orient both K/V tile batches in one launch."""
    tile_index = tl.program_id(0)
    group_block = tl.program_id(1)
    dim_block = tl.program_id(2)

    block_index = (tile_index // NUM_KV_HEADS).to(tl.int64)
    head_index = (tile_index - block_index * NUM_KV_HEADS).to(tl.int64)
    pool_slot = tl.load(PoolSlots_ptr + block_index).to(tl.int64)
    group_offsets = group_block * BLOCK_G + tl.arange(0, BLOCK_G)
    dim_offsets = dim_block * BLOCK_D + tl.arange(0, BLOCK_D)

    source = (
        pool_slot * stride_pp
        + group_offsets[:, None] * stride_pg
        + head_index * stride_ph
        + dim_offsets[None, :] * stride_pd
    )
    key = tl.load(TailKey_ptr + source).to(tl.float32)
    value = tl.load(TailValue_ptr + source).to(tl.float32)

    key_output = (
        tile_index * HEAD_DIM * GROUP
        + dim_offsets[:, None] * GROUP
        + group_offsets[None, :]
    )
    value_output = (
        tile_index * GROUP * HEAD_DIM
        + group_offsets[:, None] * HEAD_DIM
        + dim_offsets[None, :]
    )
    tl.store(KeyTiles_ptr + key_output, tl.trans(key))
    tl.store(ValueTiles_ptr + value_output, value)


def _materialize_sinkhorn_pool_kv(
    tail_key: torch.Tensor,
    tail_value: torch.Tensor,
    pool_slots: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create production-layout fp32 K/V tiles with one fused XPU launch."""
    block_count = pool_slots.numel()
    _, group, num_kv_heads, head_dim = tail_key.shape
    tile_count = block_count * num_kv_heads
    key_tiles = torch.empty(
        tile_count,
        head_dim,
        group,
        dtype=torch.float32,
        device=tail_key.device,
    )
    value_tiles = torch.empty(
        tile_count,
        group,
        head_dim,
        dtype=torch.float32,
        device=tail_key.device,
    )
    if tile_count == 0:
        return key_tiles, value_tiles

    block_g = 64
    block_d = 128
    grid = (
        tile_count,
        triton.cdiv(group, block_g),
        triton.cdiv(head_dim, block_d),
    )
    _sinkhorn_pool_materialize_kernel[grid](
        tail_key,
        tail_value,
        pool_slots,
        key_tiles,
        value_tiles,
        tail_key.stride(0),
        tail_key.stride(1),
        tail_key.stride(2),
        tail_key.stride(3),
        NUM_KV_HEADS=num_kv_heads,
        GROUP=group,
        HEAD_DIM=head_dim,
        BLOCK_G=block_g,
        BLOCK_D=block_d,
        num_warps=8,
        num_stages=2,
    )
    return key_tiles, value_tiles


def kvarn_sinkhorn_fused_pool_kv_triton(
    tail_key: torch.Tensor,
    tail_value: torch.Tensor,
    pool_slots: torch.Tensor,
    *,
    iterations: int,
) -> tuple[torch.Tensor, ...]:
    """Fused-materialize and balance pages from the Xe2 beta tail-pool ABI."""
    expected_shape = (128, 4, 256)
    if (
        tail_key.device.type != "xpu"
        or tail_key.dtype != torch.float16
        or not tail_key.is_contiguous()
        or tail_key.shape[1:] != expected_shape
        or tail_value.shape != tail_key.shape
        or tail_value.device != tail_key.device
        or tail_value.dtype != tail_key.dtype
        or not tail_value.is_contiguous()
        or pool_slots.device != tail_key.device
        or pool_slots.dtype != torch.int64
        or pool_slots.ndim != 1
        or not pool_slots.is_contiguous()
        or not 0 <= iterations <= 64
    ):
        raise ValueError(
            "pool-indexed Sinkhorn requires contiguous XPU FP16 K/V pools "
            "with shape [P,128,4,256], contiguous same-device int64 slots, "
            "and 0..64 iterations"
        )
    key_tiles, value_tiles = _materialize_sinkhorn_pool_kv(
        tail_key, tail_value, pool_slots
    )
    key = kvarn_sinkhorn_triton(
        key_tiles, iterations=iterations, _fp16_pool_input=True
    )
    value = kvarn_sinkhorn_triton(
        value_tiles, iterations=iterations, _fp16_pool_input=True
    )
    return (*key, *value)
