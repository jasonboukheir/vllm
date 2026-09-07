# SPDX-License-Identifier: Apache-2.0
"""Real B70 verification: DPAS history, causal queries and rejected-tail overwrite."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
from vllm.v1.attention.backends.kvarn_attn import (
    KVarNAttentionImpl,
    KVarNMetadata,
    _build_hadamard,
    _native_verify_view,
)
from vllm.v1.attention.ops.kvarn_store import _pack_dpas_k4, _pack_dpas_v4


@pytest.mark.parametrize("seq_len", [383, 384, 385])
@pytest.mark.parametrize("query_len", [2, 3])
@pytest.mark.parametrize("history_pages", [1, 1022])
@torch.inference_mode()
def test_dpas_mtp_verify_masks_future_and_overwrites_rejected_tail(
    seq_len, query_len, history_pages
):
    if not torch.xpu.is_available():
        pytest.skip("requires real XPU")
    cfg = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128_compact", 256)
    torch.manual_seed(37)
    device = torch.device("xpu")
    seq_len += (history_pages - 1) * 128
    # Logical pages: resident sink, packed history, two resident tail pages.
    cache = torch.zeros((5, 4, cfg.record_bytes), dtype=torch.uint8)
    qk = torch.randint(0, 16, (4, 256, 128), dtype=torch.uint8)
    qv = torch.randint(0, 16, (4, 128, 256), dtype=torch.uint8)
    cache[1, :, : cfg.k_packed_bytes] = _pack_dpas_k4(qk).reshape(4, -1)
    cache[1, :, cfg.v_packed_offset : cfg.v_packed_offset + cfg.v_packed_bytes] = (
        _pack_dpas_v4(qv).reshape(4, -1)
    )
    for offset, length, value in (
        (cfg.k_s_col_offset, 256, 0.125),
        (cfg.k_zp_offset, 256, -1),
        (cfg.k_s_row_offset, 128, 1),
        (cfg.v_s_col_offset, 256, 1),
        (cfg.v_s_row_offset, 128, 0.125),
        (cfg.v_zp_offset, 128, -1),
    ):
        cache[1, :, offset : offset + length * 2] = torch.full(
            (4, length), value, dtype=torch.float16
        ).view(torch.uint8)
    packed_k = (qk.float().permute(2, 0, 1) * 0.125 - 1).half().to(device)
    packed_v = (qv.float().permute(1, 0, 2) * 0.125 - 1).half().to(device)
    cache = cache.to(device)
    impl = object.__new__(KVarNAttentionImpl)
    impl.kvarn_config = cfg
    impl.num_heads, impl.num_kv_heads, impl.head_size = 24, 4, 256
    impl.scale, impl.fa_version = 256**-0.5, None
    impl._kvarn_dpas_layout = True
    impl._kvarn_cache_layout = "xe2_dpas"
    impl._kvarn_cached_prefill_materializer = "native_xe2"
    impl._block_lookup_size = 5
    impl._block_to_slot_t = torch.tensor(
        [2, -1, -1, 0, 1], dtype=torch.int32, device=device
    )
    impl._tail_K_pool = torch.randn((3, 128, 4, 256), device=device).half()
    impl._tail_V_pool = torch.randn_like(impl._tail_K_pool)
    impl._fa_K_buf = torch.empty((512, 4, 256), dtype=torch.float16, device=device)
    impl._fa_V_buf = torch.empty_like(impl._fa_K_buf)
    impl._H_fp16 = _build_hadamard(256, device).half()
    impl._kvarn_native_kernel_variant = 18
    impl._kvarn_native_max_splits = 32
    impl._kvarn_native_split_policy = "b70_q6_id18_v1"
    impl._q_rot_fp16_buf = torch.empty(
        (query_len * 24, 256), device=device, dtype=torch.float16
    )
    impl._fused_out_buf = torch.empty_like(impl._q_rot_fp16_buf)
    impl._native_output_fp16_buf = torch.empty_like(impl._q_rot_fp16_buf)
    impl._native_decode_scratch = (
        torch.empty((query_len, 24 * 32, 256), device=device, dtype=torch.float16),
        torch.empty((query_len, 24, 32), device=device, dtype=torch.float32),
        torch.empty((query_len, 24, 32), device=device, dtype=torch.float32),
    )

    def no_materialization(*args, **kwargs):
        raise AssertionError("native verify must not materialize history")

    impl._materialize_cached_prefill_kv = no_materialization
    md = KVarNMetadata(
        seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device),
        slot_mapping=torch.zeros(query_len, dtype=torch.int64, device=device),
        block_table=torch.tensor(
            [[3, *([1] * history_pages), 4, 0]], dtype=torch.int32, device=device
        ),
        query_start_loc=torch.tensor([0, query_len], dtype=torch.int32, device=device),
        num_actual_tokens=query_len,
        max_query_len=query_len,
        max_seq_len=seq_len,
        is_prefill=True,
        num_decodes=1,
        num_decode_tokens=query_len,
        has_cached_multiquery=True,
    )
    q = torch.randn((query_len, 24, 256), device=device).half()
    md.native_verify_metadata = _native_verify_view(md)
    assert md.native_verify_metadata is not None

    def verify():
        key = torch.cat(
            [
                impl._tail_K_pool[0],
                packed_k.repeat(history_pages, 1, 1),
                *impl._tail_K_pool[1:],
            ]
        )[:seq_len]
        value = torch.cat(
            [
                impl._tail_V_pool[0],
                packed_v.repeat(history_pages, 1, 1),
                *impl._tail_V_pool[1:],
            ]
        )[:seq_len]
        qr = (q.reshape(-1, 256) @ impl._H_fp16).reshape_as(q).float()
        mask = torch.arange(seq_len, device=device)[None, :] <= (
            torch.arange(query_len, device=device)[:, None] + seq_len - query_len
        )
        expected = F.scaled_dot_product_attention(
            qr.transpose(0, 1)[None],
            key.float().transpose(0, 1)[None],
            value.float().transpose(0, 1)[None],
            attn_mask=mask,
            scale=impl.scale,
            enable_gqa=True,
        )[0].transpose(0, 1)
        expected = (expected.reshape(-1, 256) @ impl._H_fp16.float()).reshape_as(q)
        result = impl._verify_decode_path(q, cache, md).clone()
        assert torch.isfinite(result).all()
        torch.testing.assert_close(result.float(), expected, atol=0.01, rtol=0.02)
        return result

    original = verify()
    saved = impl._tail_V_pool.clone()
    for rejected_from in range(1, query_len):
        # No draft accepted, or a partially accepted draft prefix.
        for position in range(seq_len - query_len + rejected_from, seq_len):
            slot, offset = position // 128 - history_pages, position % 128
            # Keep a visible future-token signal despite long-context dilution.
            impl._tail_V_pool[slot, offset].fill_(64 * (seq_len / 384))
        poisoned = verify()
        torch.testing.assert_close(
            original[:rejected_from], poisoned[:rejected_from], atol=0, rtol=0
        )
        assert (original[-1] - poisoned[-1]).abs().max() > 0.01
        impl._tail_V_pool.copy_(saved)
        torch.testing.assert_close(verify(), original, atol=0, rtol=0)
    torch.xpu.synchronize()
