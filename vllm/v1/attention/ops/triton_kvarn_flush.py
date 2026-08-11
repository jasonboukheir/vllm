# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton record writer for batched KVarN flushes."""

from __future__ import annotations

from vllm.triton_utils import tl, triton


def kvarn_dpas_k_coord(lane: int, slot: int) -> tuple[int, int]:
    """Return the token and dimension within one Xe2 K fragment."""
    return lane // 2 + 8 * (slot % 2), 2 * (slot // 2) + lane % 2


def kvarn_dpas_v_coord(lane: int, slot: int) -> tuple[int, int]:
    """Return the dimension and token within one Xe2 V fragment."""
    inner = slot % 16
    return (
        lane // 2 + 8 * (inner % 2) + 16 * (slot // 16),
        2 * (inner // 2) + lane % 2,
    )


@triton.jit
def _kvarn_flush_record_kernel(
    KPacked,
    KSCol,
    KZp,
    KSRow,
    VPacked,
    VSCol,
    VSRow,
    VZp,
    BlockIds,
    Cache,
    stride_cache_block,
    stride_cache_head,
    K_PACKED_BYTES: tl.constexpr,
    V_PACKED_OFFSET: tl.constexpr,
    V_PACKED_BYTES: tl.constexpr,
    K_SCOL_OFFSET: tl.constexpr,
    K_ZP_OFFSET: tl.constexpr,
    K_SROW_OFFSET: tl.constexpr,
    V_SCOL_OFFSET: tl.constexpr,
    V_SROW_OFFSET: tl.constexpr,
    V_ZP_OFFSET: tl.constexpr,
    RECORD_BYTES: tl.constexpr,
    TILE_BYTES: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    NATIVE_DPAS_LAYOUT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Write one cache record, optionally in retained Xe2 B-fragment order."""
    pid = tl.program_id(0)
    offs = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = offs < TILE_BYTES
    value = tl.zeros([BLOCK], tl.uint8)

    k_payload = offs < K_PACKED_BYTES
    k_src = offs
    if NATIVE_DPAS_LAYOUT:
        k_byte = offs % 32
        k_lane = (offs // 32) % 16
        k_sg = (offs // (32 * 16)) % 4
        k_tile = (offs // (32 * 16 * 4)) % 4
        k_half = offs // (32 * 16 * 4 * 4)
        k_slot0 = 2 * k_byte
        k_token0 = k_lane // 2 + 8 * (k_slot0 % 2)
        k_dim0 = 2 * (k_slot0 // 2) + k_lane % 2
        k_token1 = k_lane // 2 + 8 * ((k_slot0 + 1) % 2)
        k_dim1 = 2 * ((k_slot0 + 1) // 2) + k_lane % 2
        k_row0 = k_tile * 64 + k_dim0
        k_row1 = k_tile * 64 + k_dim1
        k_col0 = k_half * 64 + k_sg * 16 + k_token0
        k_col1 = k_half * 64 + k_sg * 16 + k_token1
        k_src0 = k_row0 * 64 + k_col0 // 2
        k_src1 = k_row1 * 64 + k_col1 // 2
        kb0 = tl.load(KPacked + pid * K_PACKED_BYTES + k_src0, mask=k_payload)
        kb1 = tl.load(KPacked + pid * K_PACKED_BYTES + k_src1, mask=k_payload)
        kn0 = (kb0 >> (4 * (k_col0 % 2))) & 15
        kn1 = (kb1 >> (4 * (k_col1 % 2))) & 15
        value = tl.where(k_payload, kn0 | (kn1 << 4), value)
    else:
        value = tl.where(
            k_payload,
            tl.load(KPacked + pid * K_PACKED_BYTES + k_src, mask=k_payload),
            value,
        )

    v_payload = (offs >= V_PACKED_OFFSET) & (
        offs < V_PACKED_OFFSET + V_PACKED_BYTES
    )
    v_offs = offs - V_PACKED_OFFSET
    v_src = v_offs
    if NATIVE_DPAS_LAYOUT:
        v_byte = v_offs % 16
        v_lane = (v_offs // 16) % 16
        v_sg = (v_offs // (16 * 16)) % 4
        v_tile = (v_offs // (16 * 16 * 4)) % 8
        v_half = v_offs // (16 * 16 * 4 * 8)
        v_slot0 = 2 * v_byte
        v_inner0 = v_slot0 % 16
        v_dim0 = v_lane // 2 + 8 * (v_inner0 % 2) + 16 * (v_slot0 // 16)
        v_token0 = 2 * (v_inner0 // 2) + v_lane % 2
        v_slot1 = v_slot0 + 1
        v_inner1 = v_slot1 % 16
        v_dim1 = v_lane // 2 + 8 * (v_inner1 % 2) + 16 * (v_slot1 // 16)
        v_token1 = 2 * (v_inner1 // 2) + v_lane % 2
        v_row0 = v_half * 64 + v_sg * 16 + v_token0
        v_row1 = v_half * 64 + v_sg * 16 + v_token1
        v_col0 = v_tile * 32 + v_dim0
        v_col1 = v_tile * 32 + v_dim1
        v_src0 = v_row0 * 128 + v_col0 // 2
        v_src1 = v_row1 * 128 + v_col1 // 2
        vb0 = tl.load(VPacked + pid * V_PACKED_BYTES + v_src0, mask=v_payload)
        vb1 = tl.load(VPacked + pid * V_PACKED_BYTES + v_src1, mask=v_payload)
        vn0 = (vb0 >> (4 * (v_col0 % 2))) & 15
        vn1 = (vb1 >> (4 * (v_col1 % 2))) & 15
        value = tl.where(v_payload, vn0 | (vn1 << 4), value)
    else:
        value = tl.where(
            v_payload,
            tl.load(VPacked + pid * V_PACKED_BYTES + v_src, mask=v_payload),
            value,
        )

    mask = (offs >= K_SCOL_OFFSET) & (offs < K_ZP_OFFSET)
    ptrs = KSCol + pid * (K_ZP_OFFSET - K_SCOL_OFFSET) + offs - K_SCOL_OFFSET
    value = tl.where(mask, tl.load(ptrs, mask=mask), value)
    mask = (offs >= K_ZP_OFFSET) & (offs < K_SROW_OFFSET)
    ptrs = KZp + pid * (K_SROW_OFFSET - K_ZP_OFFSET) + offs - K_ZP_OFFSET
    value = tl.where(mask, tl.load(ptrs, mask=mask), value)
    mask = (offs >= K_SROW_OFFSET) & (offs < V_PACKED_OFFSET)
    ptrs = (
        KSRow
        + pid * (V_PACKED_OFFSET - K_SROW_OFFSET)
        + offs
        - K_SROW_OFFSET
    )
    value = tl.where(mask, tl.load(ptrs, mask=mask), value)
    mask = (offs >= V_SCOL_OFFSET) & (offs < V_SROW_OFFSET)
    ptrs = VSCol + pid * (V_SROW_OFFSET - V_SCOL_OFFSET) + offs - V_SCOL_OFFSET
    value = tl.where(mask, tl.load(ptrs, mask=mask), value)
    mask = (offs >= V_SROW_OFFSET) & (offs < V_ZP_OFFSET)
    ptrs = VSRow + pid * (V_ZP_OFFSET - V_SROW_OFFSET) + offs - V_SROW_OFFSET
    value = tl.where(mask, tl.load(ptrs, mask=mask), value)
    mask = (offs >= V_ZP_OFFSET) & (offs < RECORD_BYTES)
    ptrs = VZp + pid * (RECORD_BYTES - V_ZP_OFFSET) + offs - V_ZP_OFFSET
    value = tl.where(mask, tl.load(ptrs, mask=mask), value)

    block = tl.load(BlockIds + pid // NUM_HEADS)
    head = pid % NUM_HEADS
    dst = Cache + block * stride_cache_block + head * stride_cache_head + offs
    tl.store(dst, value, mask=valid)
