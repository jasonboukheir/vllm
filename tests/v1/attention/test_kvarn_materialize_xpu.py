# SPDX-License-Identifier: Apache-2.0
"""Check the Triton cached-prefill fallback against independently decoded codes."""

import pytest
import torch

from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
from vllm.v1.attention.ops.kvarn_store import _pack_dpas_k4, _pack_dpas_v
from vllm.v1.attention.ops.triton_kvarn_decode import _kvarn_build_packed_kv_kernel


@pytest.mark.parametrize("value_bits", [2, 4])
@pytest.mark.parametrize("dpas", [False, True])
@torch.inference_mode()
def test_materializer_ragged_packed_and_resident_pages(value_bits, dpas):
    if not torch.xpu.is_available():
        pytest.skip("requires real XPU")
    cfg = KVarNConfig.from_cache_dtype(f"kvarn_k4v{value_bits}_g128_compact", 256)
    torch.manual_seed(91)
    qk = torch.randint(0, 16, (4, 256, 128), dtype=torch.uint8)
    qv = torch.randint(0, 1 << value_bits, (4, 128, 256), dtype=torch.uint8)
    if dpas:
        pk, pv = _pack_dpas_k4(qk), _pack_dpas_v(qv, value_bits)
    else:
        pk = qk[..., ::2] | (qk[..., 1::2] << 4)
        pv = torch.zeros((4, 128, 256 * value_bits // 8), dtype=torch.uint8)
        for lane in range(8 // value_bits):
            pv |= qv[..., lane :: 8 // value_bits] << (lane * value_bits)
    # Extra stride padding proves that consumers use the actual record stride.
    cache = torch.full((3, 4, cfg.record_bytes + 64), 0xA7, dtype=torch.uint8)
    cache[1, :, : cfg.k_packed_bytes] = pk.reshape(4, -1)
    cache[1, :, cfg.v_packed_offset : cfg.v_packed_offset + cfg.v_packed_bytes] = (
        pv.reshape(4, -1)
    )
    for offset, length, value in (
        (cfg.k_s_col_offset, 256, 0.125),
        (cfg.k_zp_offset, 256, -1),
        (cfg.k_s_row_offset, 128, 0.5),
        (cfg.v_s_col_offset, 256, 2),
        (cfg.v_s_row_offset, 128, 0.25),
        (cfg.v_zp_offset, 128, -0.5),
    ):
        cache[1, :, offset : offset + length * 2] = torch.full(
            (4, length), value, dtype=torch.float16
        ).view(torch.uint8)
    k = ((qk.float().permute(2, 0, 1) * 0.125 - 1) * 0.5).half()
    v = ((qv.float().permute(1, 0, 2) * 0.25 - 0.5) * 2).half()
    kp = torch.randn((2, 128, 4, 256)).half()
    vp = torch.randn_like(kp)
    expected_k = torch.cat((kp[0], k, kp[1, :1], k))
    expected_v = torch.cat((vp[0], v, vp[1, :1], v))
    cache, kp, vp = [x.to("xpu") for x in (cache, kp, vp)]
    ko = torch.full((392, 4, 256), 37, dtype=torch.float16, device="xpu")
    vo = torch.full_like(ko, -37)
    bt = torch.tensor([[2, 1, 0], [1, 0, 0]], dtype=torch.int32, device="xpu")
    lengths = torch.tensor([257, 128], dtype=torch.int32, device="xpu")
    cu = torch.tensor([0, 257, 385], dtype=torch.int32, device="xpu")
    lookup = torch.tensor([1, -1, 0], dtype=torch.int32, device="xpu")
    before = cache.clone()
    compiled_hashes = set()
    # Page-count changes must reuse code, including padded grids beyond the
    # block-table width. Invalid programs return before loading a page index.
    for max_blocks in [3, 4, 16]:
        ko.fill_(37)
        vo.fill_(-37)
        compiled = _kvarn_build_packed_kv_kernel[(max_blocks, 2, 4)](
            bt,
            lengths,
            cu,
            lookup,
            cache,
            kp,
            vp,
            ko,
            vo,
            bt.stride(0),
            cache.stride(0),
            cache.stride(1),
            kp.stride(0),
            kp.stride(1),
            kp.stride(2),
            ko.stride(0),
            ko.stride(1),
            D=256,
            GROUP=128,
            K_BITS=4,
            V_BITS=value_bits,
            NUM_BLOCKS_LOOKUP=3,
            K_PACKED_OFFSET=cfg.k_packed_offset,
            K_S_COL_OFFSET=cfg.k_s_col_offset,
            K_ZP_OFFSET=cfg.k_zp_offset,
            K_S_ROW_OFFSET=cfg.k_s_row_offset,
            V_PACKED_OFFSET=cfg.v_packed_offset,
            V_S_COL_OFFSET=cfg.v_s_col_offset,
            V_S_ROW_OFFSET=cfg.v_s_row_offset,
            V_ZP_OFFSET=cfg.v_zp_offset,
            DPAS_LAYOUT=dpas,
        )
        torch.testing.assert_close(ko[:385].cpu(), expected_k, atol=0, rtol=0)
        torch.testing.assert_close(vo[:385].cpu(), expected_v, atol=0, rtol=0)
        assert (ko[385:] == 37).all() and (vo[385:] == -37).all()
        assert torch.equal(cache, before)
        compiled_hashes.add(compiled.hash)
    assert len(compiled_hashes) == 1
