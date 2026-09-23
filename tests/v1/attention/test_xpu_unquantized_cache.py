# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit unquantized cache settings must reach XPU's input-matching writer."""

import pytest
import torch

from vllm import _custom_ops as ops


def _cache_pair(blocks, block_size, heads, dim, dtype, padding, strided):
    if strided:
        # FlashAttentionImpl splits the last dimension after transposing H/N.
        storage = torch.empty(
            (blocks, heads, block_size, 2 * dim), device="xpu", dtype=dtype
        )
        key, value = storage.transpose(1, 2).split(dim, dim=-1)
        assert key.stride(-2) == block_size * 2 * dim
    else:
        key = torch.empty((blocks, block_size, heads, dim), device="xpu", dtype=dtype)
        value = torch.empty_like(key)
    key.fill_(padding)
    value.fill_(-padding)
    return key, value


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
@pytest.mark.parametrize("strided", [False, True])
def test_explicit_unquantized_cache_preserves_values_and_sparse_slots(
    dtype_name, strided
):
    if not torch.xpu.is_available():
        pytest.skip("requires XPU")
    dtype = getattr(torch, dtype_name)
    key = torch.arange(6 * 4 * 256, device="xpu").reshape(6, 4, 256).to(dtype)
    value = -key
    cache_k, cache_v = _cache_pair(3, 128, 4, 256, dtype, float("nan"), strided)
    slots_cpu = torch.tensor([0, 127, 128, 383, 256, -1], dtype=torch.int64)
    slots = slots_cpu.xpu()
    scale = torch.ones(1, device="xpu")
    ops.reshape_and_cache_flash(
        key, value, cache_k, cache_v, slots, dtype_name, scale, scale
    )
    expected_k = torch.full(cache_k.shape, float("nan"), dtype=dtype)
    expected_v = torch.full_like(expected_k, float("nan"))
    expected_k.view(-1, 4, 256)[slots_cpu[:-1]] = key.cpu()[:-1]
    expected_v.view(-1, 4, 256)[slots_cpu[:-1]] = value.cpu()[:-1]
    torch.testing.assert_close(
        cache_k.cpu(), expected_k, atol=0, rtol=0, equal_nan=True
    )
    torch.testing.assert_close(
        cache_v.cpu(), expected_v, atol=0, rtol=0, equal_nan=True
    )
    wrong = cache_v.to(torch.float32)
    with pytest.raises(ValueError, match="matching input and cache dtypes"):
        ops.reshape_and_cache_flash(
            key, value, cache_k, wrong, slots, dtype_name, scale, scale
        )


@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("padding", [0.0, 123.0])
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("query_len", [1, 3, 129])
@torch.inference_mode()
def test_bf16_draft_paged_attention_matches_independent_causal_reference(
    batch_size, query_len, padding, strided, monkeypatch
):
    """MTP and cached prefill must respect sparse pages and rejected future rows."""
    if not torch.xpu.is_available():
        pytest.skip("requires XPU")
    import torch.nn.functional as F
    from vllm_xpu_kernels import flash_attn_interface

    from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func

    def reject_fallback(*args, **kwargs):
        pytest.fail("causal reference requires native XPU attention")

    monkeypatch.setattr(flash_attn_interface, "_fallback_varlen_attn", reject_fallback)

    torch.manual_seed(43)
    device, dtype = "xpu", torch.bfloat16
    block_size, heads, kv_heads, dim = 128, 24, 4, 256
    lengths = [257 + i * 128 for i in range(batch_size)]
    max_blocks = (max(lengths) + block_size - 1) // block_size
    pages = torch.randperm(batch_size * max_blocks).reshape(batch_size, max_blocks)
    # Serving starts with zeroed pools; reused slots contain finite past K/V.
    cache_k, cache_v = _cache_pair(
        batch_size * max_blocks, block_size, kv_heads, dim, dtype, padding, strided
    )
    keys, values = [], []
    for i, length in enumerate(lengths):
        key = torch.randn((length, kv_heads, dim), device=device, dtype=dtype)
        value = torch.randn_like(key)
        keys.append(key)
        values.append(value)
        slots = (
            pages[i, torch.arange(length) // block_size] * block_size
            + torch.arange(length) % block_size
        ).to(device)
        scale = torch.ones(1, device=device)
        ops.reshape_and_cache_flash(
            key, value, cache_k, cache_v, slots, "bfloat16", scale, scale
        )

    query = torch.randn(
        (batch_size * query_len, heads, dim), device=device, dtype=dtype
    )
    output = torch.full_like(query, float("nan"))
    block_table = pages.to(device=device, dtype=torch.int32)
    cu_query = (
        torch.arange(batch_size + 1, device=device, dtype=torch.int32) * query_len
    )
    seq_lens = torch.tensor(lengths, device=device, dtype=torch.int32)

    def execute():
        flash_attn_varlen_func(
            q=query,
            k=cache_k,
            v=cache_v,
            out=output,
            cu_seqlens_q=cu_query,
            seqused_k=seq_lens,
            max_seqlen_q=query_len,
            max_seqlen_k=max(lengths),
            softmax_scale=dim**-0.5,
            causal=True,
            block_table=block_table,
        )
        assert torch.isfinite(output).all()
        return output.clone()

    expected = []
    for i, length in enumerate(lengths):
        causal = torch.arange(length, device=device)[None, :] <= (
            torch.arange(query_len, device=device)[:, None] + length - query_len
        )
        expected.append(
            F.scaled_dot_product_attention(
                query[i * query_len : (i + 1) * query_len]
                .float()
                .transpose(0, 1)[None],
                keys[i].float().transpose(0, 1)[None],
                values[i].float().transpose(0, 1)[None],
                attn_mask=causal,
                enable_gqa=True,
                scale=dim**-0.5,
            )[0].transpose(0, 1)
        )
    original = execute()
    torch.testing.assert_close(
        original.float(), torch.cat(expected), atol=0.01, rtol=0.02
    )
    if query_len > 1:
        saved = []
        for i, length in enumerate(lengths):
            page, offset = (
                pages[i, (length - 1) // block_size].item(),
                (length - 1) % block_size,
            )
            saved.append(cache_v[page, offset].clone())
            cache_v[page, offset].fill_(64)
        poisoned = execute().view(batch_size, query_len, heads, dim)
        torch.testing.assert_close(
            poisoned[:, :-1], original.view_as(poisoned)[:, :-1], atol=0, rtol=0
        )
        assert (poisoned[:, -1] - original.view_as(poisoned)[:, -1]).abs().max() > 0.01
        for i, length in enumerate(lengths):
            page, offset = (
                pages[i, (length - 1) // block_size].item(),
                (length - 1) % block_size,
            )
            cache_v[page, offset].copy_(saved[i])
    torch.testing.assert_close(execute(), original, atol=0, rtol=0)
