# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for KVarN configuration and cache accounting."""

import os
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.kvarn.config import (
    KVARN_PRESETS,
    KVarNConfig,
    kvarn_prefill_fp16_window_blocks,
)
from vllm.platforms.xpu import XPUPlatform
from vllm.v1.attention.backends.kvarn_attn import (
    KVarNAttentionBackend,
    KVarNAttentionImpl,
    KVarNMetadataBuilder,
    _can_elide_fa_cu_seqlens,
    _cast_kvarn_activations,
    _defer_kvarn_prefill_history_blocks,
    _is_pure_kvarn_decode_step,
    _is_pure_qlen1_batch,
    _kvarn_block_table_numpy,
    _kvarn_dpas_layout_for_config,
    _kvarn_prefill_fp16_window_blocks,
    _kvarn_reclaimable_block_ids,
    _kvarn_walk_back_flush_blocks,
    _KVarNMetadataStageRing,
    _protect_kvarn_prefill_window_blocks,
    _reconcile_kvarn_sink_ownership,
    _rotate_kvarn_kv_into_scratch,
    _use_kvarn_fused_verify,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.backends.turboquant_attn import TurboQuantAttentionBackend
from vllm.v1.attention.ops.kvarn_store import (
    _pack_dpas_k4,
    _pack_dpas_v4,
    kvarn_store_tile_k_batch_from_sinkhorn,
    kvarn_store_tile_v_batch_from_sinkhorn,
)
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheLayout,
    KVCacheTensor,
    compute_layer_kv_cache_shape_bytes,
    compute_layout_strides,
    create_kv_cache_views,
    get_kv_quant_mode,
    is_quantized_kv_cache,
)


def _shared_q_output_maps():
    return (
        KVarNAttentionImpl._shared_q_fp32_buf,
        KVarNAttentionImpl._shared_q_rot_fp32_buf,
        KVarNAttentionImpl._shared_q_rot_fp16_buf,
        KVarNAttentionImpl._shared_out_rot_fp32_buf,
        KVarNAttentionImpl._shared_output_fp32_buf,
        KVarNAttentionImpl._shared_fused_out_buf,
        KVarNAttentionImpl._shared_native_output_fp16_buf,
    )


def test_pure_decode_preserves_native_transform_inputs():
    query, key, value = (torch.randn(2, 3, dtype=torch.bfloat16) for _ in range(3))

    cast_query, cast_key, cast_value = _cast_kvarn_activations(
        query, key, value, query_only=True
    )

    assert cast_query is query
    assert cast_key is key
    assert cast_value is value


def test_kvarn_activation_cast_preserves_fp16_tensor_identity():
    tensors = tuple(torch.randn(2, 3, dtype=torch.float16) for _ in range(3))

    cast = _cast_kvarn_activations(*tensors, query_only=False)

    assert all(result is original for result, original in zip(cast, tensors))


def test_non_decode_kvarn_casts_every_activation():
    tensors = tuple(torch.randn(2, 3, dtype=torch.bfloat16) for _ in range(3))

    cast = _cast_kvarn_activations(*tensors, query_only=False)

    assert all(tensor.dtype == torch.float16 for tensor in cast)


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({}, True),
        ({"is_prefill": True}, False),
        ({"max_query_len": 2}, False),
        ({"num_decodes": 0}, False),
        ({"num_decode_tokens": 3}, False),
        ({"num_actual_tokens": 3}, False),
    ],
)
def test_native_store_requires_a_pure_qlen1_decode(override, expected):
    values = dict(
        is_prefill=False,
        max_query_len=1,
        num_decodes=4,
        num_decode_tokens=4,
        num_actual_tokens=4,
    )
    values.update(override)

    assert _is_pure_kvarn_decode_step(SimpleNamespace(**values), 4) is expected


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({}, True),
        ({"num_decodes": 0}, False),
        ({"num_prefills": 1}, False),
        ({"num_decode_tokens": 3}, False),
        ({"num_actual_tokens": 3}, False),
        ({"max_query_len": 2}, False),
    ],
)
def test_pure_qlen1_batch_requires_only_single_token_decodes(override, expected):
    values = dict(
        num_decodes=4,
        num_prefills=0,
        num_decode_tokens=4,
        num_actual_tokens=4,
        max_query_len=1,
    )
    values.update(override)
    assert _is_pure_qlen1_batch(**values) is expected


def test_kvarn_block_table_prefers_runner_cpu_mirror():
    class DeviceTable:
        def cpu(self):
            raise AssertionError("device block table must not be copied to CPU")

    mirror = torch.tensor([[3, 5], [7, 11]], dtype=torch.int32)
    result = _kvarn_block_table_numpy(
        SimpleNamespace(block_table_cpu=mirror, block_table_tensor=DeviceTable())
    )

    assert result.tolist() == [[3, 5], [7, 11]]


def test_runner_cpu_block_table_view_excludes_stale_padded_rows():
    from vllm.v1.worker.gpu_model_runner import _block_table_cpu_view

    table = SimpleNamespace(
        get_numpy_array=lambda: np.array(
            [[3, 5], [7, 11], [101, 103], [107, 109]], dtype=np.int32
        )
    )

    result = _block_table_cpu_view(table, num_reqs=2)

    assert result.tolist() == [[3, 5], [7, 11]]


@pytest.mark.parametrize("proposer_name", ["step3p5", "gemma4"])
def test_per_group_block_table_swap_clears_primary_cpu_mirror(proposer_name):
    class MetadataBuilder:
        def build_for_drafting(self, common_attn_metadata, draft_index):
            return common_attn_metadata

    group = SimpleNamespace(
        kv_cache_group_id=1,
        layer_names=["model.layers.0.self_attn"],
        get_metadata_builder=lambda: MetadataBuilder(),
    )
    primary_mirror = np.array([[3], [5]], dtype=np.int32)
    common = SimpleNamespace(
        num_reqs=2,
        num_actual_tokens=2,
        block_table_tensor=torch.tensor([[3], [5]], dtype=torch.int32),
        block_table_cpu=primary_mirror,
        batch_size=lambda: 2,
    )

    if proposer_name == "step3p5":
        from vllm.v1.spec_decode.step3p5 import Step3p5MTPProposer

        proposer = object.__new__(Step3p5MTPProposer)
        proposer._per_group_slot_mappings = {}
        method = Step3p5MTPProposer.build_per_group_and_layer_attn_metadata
    else:
        from vllm.v1.spec_decode.gemma4 import Gemma4Proposer

        proposer = object.__new__(Gemma4Proposer)
        method = Gemma4Proposer.build_per_group_and_layer_attn_metadata
    proposer.draft_attn_groups = [group]
    proposer._per_group_block_tables = {
        1: torch.tensor([[11], [13]], dtype=torch.int32)
    }

    per_group, _ = method(proposer, common)

    assert per_group[0].block_table_cpu is None
    assert common.block_table_cpu is primary_mirror


def test_static_sink_table_replacement_clears_cpu_mirror():
    from vllm.model_executor.layers.attention.static_sink_attention import (
        create_static_sink_attention_backend,
    )

    class MetadataBuilder:
        def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
            pass

        def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
            return common_attn_metadata

    class AttentionBackend:
        @staticmethod
        def get_builder_cls():
            return MetadataBuilder

    backend = create_static_sink_attention_backend(AttentionBackend, sink_len=16)
    builder = backend.get_builder_cls()(
        None,
        [],
        SimpleNamespace(
            model_config=SimpleNamespace(max_model_len=32),
            scheduler_config=SimpleNamespace(max_num_seqs=2),
            cache_config=SimpleNamespace(block_size=16),
        ),
        torch.device("cpu"),
    )
    primary_mirror = np.array([[3, 5], [7, 11]], dtype=np.int32)
    common = SimpleNamespace(
        seq_lens=torch.tensor([16, 16], dtype=torch.int32),
        max_seq_len=16,
        num_reqs=2,
        block_table_tensor=torch.tensor([[3, 5], [7, 11]], dtype=torch.int32),
        block_table_cpu=primary_mirror,
    )

    result = builder.build(0, common)

    assert result.block_table_cpu is None


def test_fa_metadata_elision_requires_fused_eager_without_capture_history():
    kwargs = dict(
        pure_qlen1=True,
        has_built_cudagraph_metadata=False,
    )
    with patch.dict(os.environ, {"KVARN_FUSED_DECODE": "0"}):
        # Capture stages the persistent buffers, and the later ordinary build
        # must keep refreshing them for materializer graph replay.
        assert not _can_elide_fa_cu_seqlens(for_cudagraph_capture=True, **kwargs)
        assert not _can_elide_fa_cu_seqlens(for_cudagraph_capture=False, **kwargs)
    with patch.dict(os.environ, {"KVARN_FUSED_DECODE": "1"}):
        assert _can_elide_fa_cu_seqlens(for_cudagraph_capture=False, **kwargs)
        assert not _can_elide_fa_cu_seqlens(
            pure_qlen1=False,
            for_cudagraph_capture=False,
            has_built_cudagraph_metadata=False,
        )
        assert not _can_elide_fa_cu_seqlens(
            pure_qlen1=True,
            for_cudagraph_capture=False,
            has_built_cudagraph_metadata=True,
        )


def test_cudagraph_builder_forces_persistent_fa_metadata_staging():
    builder = object.__new__(KVarNMetadataBuilder)
    builder._has_built_cudagraph_metadata = False
    common = object()
    sentinel = object()

    with patch.object(builder, "build", return_value=sentinel) as build:
        assert builder.build_for_cudagraph_capture(common) is sentinel

    build.assert_called_once_with(0, common, _for_cudagraph_capture=True)
    assert builder._has_built_cudagraph_metadata
    with patch.dict(os.environ, {"KVARN_FUSED_DECODE": "1"}):
        assert not _can_elide_fa_cu_seqlens(
            pure_qlen1=True,
            for_cudagraph_capture=False,
            has_built_cudagraph_metadata=builder._has_built_cudagraph_metadata,
        )


def test_pure_qlen1_builder_skips_fa_staging_and_slot_mapping_d2h():
    class NoStageRing:
        def acquire(self):
            raise AssertionError("pure qlen=1 must not acquire a metadata stage")

    class SlotMapping(torch.Tensor):
        def tolist(self):
            raise AssertionError("slot mapping must not be copied to CPU")

    KVarNAttentionImpl.reset_process_state()
    try:
        builder = object.__new__(KVarNMetadataBuilder)
        builder.reorder_batch_threshold = 1
        builder._group = 128
        builder._group_key = ("model.layers.0.self_attn",)
        builder._layer_names_set = set(builder._group_key)
        builder._retired_sinks = {}
        builder._block_fill = {}
        builder._max_model_len = 65536
        builder._metadata_stages = NoStageRing()
        builder._cu_seqlens_q_buf = None
        builder._cu_seqlens_k_buf = None
        builder._cu_seqlens_q_host = None
        builder._cu_seqlens_k_host = None
        builder._vq_req_buf = None
        builder._vq_seqlen_buf = None
        builder._vq_req_host = None
        builder._vq_seqlen_host = None

        seq_lens_cpu = torch.tensor([4096, 4097], dtype=torch.int32)
        query_start_loc_cpu = torch.tensor([0, 1, 2], dtype=torch.int32)
        block_table_cpu = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
        slot_mapping = torch.tensor([128, 385], dtype=torch.int64).as_subclass(
            SlotMapping
        )
        cam = SimpleNamespace(
            seq_lens=seq_lens_cpu,
            seq_lens_cpu=seq_lens_cpu,
            query_start_loc=query_start_loc_cpu,
            query_start_loc_cpu=query_start_loc_cpu,
            block_table_tensor=block_table_cpu,
            block_table_cpu=block_table_cpu,
            slot_mapping=slot_mapping,
            num_actual_tokens=2,
            max_query_len=1,
            max_seq_len=4097,
            causal=True,
        )

        with (
            patch(
                "vllm.v1.attention.backends.kvarn_attn.split_decodes_and_prefills",
                return_value=(2, 0, 2, 0),
            ),
            patch(
                "vllm.v1.attention.backends.kvarn_attn._can_elide_fa_cu_seqlens",
                return_value=True,
            ),
        ):
            metadata = builder.build(0, cam)

        assert metadata.fa_cu_seqlens_q is None
        assert metadata.fa_cu_seqlens_k is None
        assert metadata.slot_mapping_cpu is None
    finally:
        KVarNAttentionImpl.reset_process_state()


def test_dpas_pack_matches_frozen_xe2_fragment_coordinates():
    dims = torch.arange(256, dtype=torch.int32)[:, None]
    tokens = torch.arange(128, dtype=torch.int32)[None, :]
    q_k = ((3 * dims + 5 * tokens + dims // 16 + tokens // 16) & 15).unsqueeze(0)
    q_v = (7 * tokens.T + 11 * dims.T + tokens.T // 8 + dims.T // 32) & 15
    q_v = q_v.unsqueeze(0)

    expected_k = torch.empty((2, 4, 4, 16, 32), dtype=torch.uint8)
    expected_v = torch.empty((2, 8, 4, 16, 16), dtype=torch.uint8)
    for half in range(2):
        for tile in range(4):
            for subgroup in range(4):
                for lane in range(16):
                    for byte in range(32):
                        values = []
                        for nibble in range(2):
                            slot = 2 * byte + nibble
                            token = lane // 2 + 8 * (slot % 2)
                            dim = 2 * (slot // 2) + lane % 2
                            values.append(
                                q_k[
                                    0,
                                    tile * 64 + dim,
                                    half * 64 + subgroup * 16 + token,
                                ]
                            )
                        expected_k[half, tile, subgroup, lane, byte] = values[0] | (
                            values[1] << 4
                        )
        for tile in range(8):
            for subgroup in range(4):
                for lane in range(16):
                    for byte in range(16):
                        values = []
                        for nibble in range(2):
                            slot = 2 * byte + nibble
                            inner = slot % 16
                            dim = lane // 2 + 8 * (inner % 2) + 16 * (slot // 16)
                            token = 2 * (inner // 2) + lane % 2
                            values.append(
                                q_v[
                                    0,
                                    half * 64 + subgroup * 16 + token,
                                    tile * 32 + dim,
                                ]
                            )
                        expected_v[half, tile, subgroup, lane, byte] = values[0] | (
                            values[1] << 4
                        )

    torch.testing.assert_close(
        _pack_dpas_k4(q_k).flatten(), expected_k.flatten(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        _pack_dpas_v4(q_v).flatten(), expected_v.flatten(), rtol=0, atol=0
    )


def test_dpas_store_preserves_metadata_and_fails_closed_on_wrong_shape():
    balanced_k = torch.randn(2, 256, 128)
    balanced_v = torch.randn(2, 128, 256)
    k_s_col = torch.rand(2, 128)
    k_s_row = torch.rand(2, 256)
    v_s_col = torch.rand(2, 256)
    v_s_row = torch.rand(2, 128)

    natural_k = kvarn_store_tile_k_batch_from_sinkhorn(balanced_k, k_s_col, k_s_row, 4)
    dpas_k = kvarn_store_tile_k_batch_from_sinkhorn(
        balanced_k, k_s_col, k_s_row, 4, dpas_layout=True
    )
    natural_v = kvarn_store_tile_v_batch_from_sinkhorn(balanced_v, v_s_col, v_s_row, 4)
    dpas_v = kvarn_store_tile_v_batch_from_sinkhorn(
        balanced_v, v_s_col, v_s_row, 4, dpas_layout=True
    )

    for field in ("s_col_K", "zp_K", "s_row_K"):
        torch.testing.assert_close(dpas_k[field], natural_k[field], rtol=0, atol=0)
    for field in ("s_col_V", "s_row_V", "zp_V"):
        torch.testing.assert_close(dpas_v[field], natural_v[field], rtol=0, atol=0)
    assert dpas_k["q_packed_uint8"].shape == natural_k["q_packed_uint8"].shape
    assert dpas_v["q_packed_uint8"].shape == natural_v["q_packed_uint8"].shape
    assert not torch.equal(dpas_k["q_packed_uint8"], natural_k["q_packed_uint8"])
    assert not torch.equal(dpas_v["q_packed_uint8"], natural_v["q_packed_uint8"])

    with pytest.raises(ValueError, match="requires 4-bit"):
        kvarn_store_tile_k_batch_from_sinkhorn(
            balanced_k, k_s_col, k_s_row, 2, dpas_layout=True
        )
    with pytest.raises(ValueError, match=r"requires \[N, 128, 256\]"):
        _pack_dpas_v4(torch.zeros(1, 64, 256, dtype=torch.int32))


def test_dpas_layout_dispatch_requires_exact_cache_config(
    monkeypatch: pytest.MonkeyPatch,
):
    cfg = SimpleNamespace(head_dim=256, group=128, key_bits=4, value_bits=4)
    monkeypatch.delenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", raising=False)
    assert not _kvarn_dpas_layout_for_config(cfg)

    monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "1")
    assert _kvarn_dpas_layout_for_config(cfg)
    cfg.value_bits = 2
    with pytest.raises(RuntimeError, match="requires D256/G128/K4V4"):
        _kvarn_dpas_layout_for_config(cfg)


def test_dpas_layout_bypasses_natural_fused_verify_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KVARN_FUSED_VERIFY", raising=False)
    assert _use_kvarn_fused_verify(
        max_query_len=1,
        max_seq_len=8192,
        group=128,
        batch_size=4,
        dpas_layout=False,
    )
    assert not _use_kvarn_fused_verify(
        max_query_len=1,
        max_seq_len=8192,
        group=128,
        batch_size=4,
        dpas_layout=True,
    )


def test_reset_process_state_releases_previous_model_generation():
    KVarNAttentionImpl.reset_process_state()
    try:
        device = torch.device("cpu")
        group_key = ("model.layers.3.self_attn",)
        mirror_key = (device, group_key)
        scratch_key = (device, 8, 2)

        KVarNAttentionImpl._all_impls.append(object())  # type: ignore[arg-type]
        KVarNAttentionImpl._block_to_slot_dict[group_key] = {7: 1}
        KVarNAttentionImpl._global_sink_blocks[group_key] = {7}
        KVarNAttentionImpl._free_slots[group_key] = [0]
        KVarNAttentionImpl._allocator_pool_size[group_key] = 2
        KVarNAttentionImpl._max_known_block_id[group_key] = 7
        KVarNAttentionImpl._block_to_slot_t_per_device[mirror_key] = torch.ones(1)
        KVarNAttentionImpl._is_sink_t_per_device[mirror_key] = torch.ones(1)
        KVarNAttentionImpl._kernel_warmed.add(("decode", device))
        KVarNAttentionImpl._tiles_dumped = True
        for mapping in _shared_q_output_maps():
            mapping[scratch_key] = torch.ones(1)
        KVarNAttentionImpl._shared_native_decode_scratch[scratch_key] = (
            torch.ones(1),
            torch.ones(1),
            torch.ones(1),
        )
        KVarNAttentionImpl._shared_mid_o_buf[scratch_key] = torch.ones(1)
        KVarNAttentionImpl._shared_mid_lse_buf[scratch_key] = torch.ones(1)
        KVarNAttentionImpl._shared_fa_K_buf[scratch_key] = torch.ones(1)
        KVarNAttentionImpl._shared_fa_V_buf[scratch_key] = torch.ones(1)

        KVarNAttentionImpl.reset_process_state()

        mappings = (
            *_shared_q_output_maps(),
            KVarNAttentionImpl._shared_native_decode_scratch,
            KVarNAttentionImpl._shared_mid_o_buf,
            KVarNAttentionImpl._shared_mid_lse_buf,
            KVarNAttentionImpl._shared_fa_K_buf,
            KVarNAttentionImpl._shared_fa_V_buf,
            KVarNAttentionImpl._block_to_slot_dict,
            KVarNAttentionImpl._global_sink_blocks,
            KVarNAttentionImpl._free_slots,
            KVarNAttentionImpl._allocator_pool_size,
            KVarNAttentionImpl._block_to_slot_t_per_device,
            KVarNAttentionImpl._is_sink_t_per_device,
            KVarNAttentionImpl._max_known_block_id,
        )
        assert all(not mapping for mapping in mappings)
        assert not KVarNAttentionImpl._all_impls
        assert not KVarNAttentionImpl._kernel_warmed
        assert KVarNAttentionImpl._tiles_dumped is False
    finally:
        KVarNAttentionImpl.reset_process_state()


def test_shared_q_output_scratch_grows_as_one_complete_set():
    KVarNAttentionImpl.reset_process_state()
    try:
        device = torch.device("cpu")
        bkey = (device, 8, 2)
        KVarNAttentionImpl._ensure_shared_q_output_scratch(bkey, 4, 8, device)
        first = tuple(mapping[bkey] for mapping in _shared_q_output_maps())

        KVarNAttentionImpl._ensure_shared_q_output_scratch(bkey, 8, 8, device)
        grown = tuple(mapping[bkey] for mapping in _shared_q_output_maps())
        assert all(buffer.shape == (8, 8) for buffer in grown)
        assert tuple(buffer.dtype for buffer in grown) == (
            torch.float32,
            torch.float32,
            torch.float16,
            torch.float32,
            torch.float32,
            torch.float16,
            torch.float16,
        )
        assert all(new is not old for new, old in zip(grown, first, strict=True))

        real_empty = torch.empty
        allocation_count = 0

        def fail_partway_through_growth(*args, **kwargs):
            nonlocal allocation_count
            allocation_count += 1
            if allocation_count == 3:
                raise RuntimeError("synthetic allocation failure")
            return real_empty(*args, **kwargs)

        with (
            patch.object(torch, "empty", side_effect=fail_partway_through_growth),
            pytest.raises(RuntimeError, match="synthetic allocation failure"),
        ):
            KVarNAttentionImpl._ensure_shared_q_output_scratch(bkey, 16, 8, device)
        after_failure = tuple(mapping[bkey] for mapping in _shared_q_output_maps())
        assert all(
            current is previous
            for current, previous in zip(after_failure, grown, strict=True)
        )

        KVarNAttentionImpl._ensure_shared_q_output_scratch(bkey, 2, 8, device)
        retained = tuple(mapping[bkey] for mapping in _shared_q_output_maps())
        assert all(
            current is previous
            for current, previous in zip(retained, grown, strict=True)
        )
    finally:
        KVarNAttentionImpl.reset_process_state()


def test_pool_initialized_fast_path_is_group_capacity_and_binding_safe():
    KVarNAttentionImpl.reset_process_state()
    try:
        impl = object.__new__(KVarNAttentionImpl)
        device = torch.device("cpu")
        group_key = ("model.layers.0.self_attn",)
        mirror_key = (device, group_key)
        scratch_key = (device, 8, 2)
        impl._group_key = group_key
        impl.kvarn_config = SimpleNamespace(head_dim=8)
        impl.num_kv_heads = 2
        impl._tail_K_pool = torch.empty(1)
        impl._tail_V_pool = torch.empty(1)
        impl._H_fp16 = torch.empty(1)
        impl._k_rot_scratch = torch.empty(1)
        impl._v_rot_scratch = torch.empty(1)
        impl._pool_ready_key = mirror_key
        impl._pool_ready_native_key = None
        impl._pool_ready_env = (
            os.environ.get("KVARN_NATIVE_XPU"),
            os.environ.get("KVARN_NATIVE_XPU_PERSISTENT_SCRATCH"),
            os.environ.get("KVARN_NATIVE_XPU_SPLITS"),
        )
        impl._native_decode_scratch = None

        block_to_slot = torch.empty(2048)
        is_sink = torch.empty(2048)
        KVarNAttentionImpl._block_to_slot_t_per_device[mirror_key] = block_to_slot
        KVarNAttentionImpl._is_sink_t_per_device[mirror_key] = is_sink
        impl._block_to_slot_t = block_to_slot
        impl._is_sink_t = is_sink
        impl._block_lookup_size = 2048

        attr_maps = (
            ("_q_fp32_buf", KVarNAttentionImpl._shared_q_fp32_buf),
            ("_q_rot_fp32_buf", KVarNAttentionImpl._shared_q_rot_fp32_buf),
            ("_q_rot_fp16_buf", KVarNAttentionImpl._shared_q_rot_fp16_buf),
            ("_out_rot_fp32_buf", KVarNAttentionImpl._shared_out_rot_fp32_buf),
            ("_output_fp32_buf", KVarNAttentionImpl._shared_output_fp32_buf),
            ("_fused_out_buf", KVarNAttentionImpl._shared_fused_out_buf),
            (
                "_native_output_fp16_buf",
                KVarNAttentionImpl._shared_native_output_fp16_buf,
            ),
            ("_mid_o_buf", KVarNAttentionImpl._shared_mid_o_buf),
            ("_mid_lse_buf", KVarNAttentionImpl._shared_mid_lse_buf),
            ("_fa_K_buf", KVarNAttentionImpl._shared_fa_K_buf),
            ("_fa_V_buf", KVarNAttentionImpl._shared_fa_V_buf),
        )
        for attr, mapping in attr_maps:
            tensor = torch.empty(1)
            mapping[scratch_key] = tensor
            setattr(impl, attr, tensor)

        assert impl._pool_is_initialized_for(device, 2048)
        assert not impl._pool_is_initialized_for(device, 2049)

        impl._group_key = ("model.layers.1.self_attn",)
        assert not impl._pool_is_initialized_for(device, 1)
        impl._group_key = group_key

        KVarNAttentionImpl._shared_fused_out_buf[scratch_key] = torch.empty(1)
        assert not impl._pool_is_initialized_for(device, 1)
    finally:
        KVarNAttentionImpl.reset_process_state()


@pytest.mark.parametrize(
    ("name", "key_bits", "value_bits", "group"),
    [
        ("kvarn_k4v2_g128", 4, 2, 128),
        ("kvarn_k4v4_g128", 4, 4, 128),
        ("kvarn_k4v2_g64", 4, 2, 64),
        ("kvarn_k4v4_g64", 4, 4, 64),
    ],
)
def test_presets_are_an_auditable_fixed_contract(name, key_bits, value_bits, group):
    assert KVARN_PRESETS[name] == {
        "key_bits": key_bits,
        "value_bits": value_bits,
        "group": group,
    }
    config = KVarNConfig.from_cache_dtype(name, head_dim=256)
    assert (config.key_bits, config.value_bits, config.group) == (
        key_bits,
        value_bits,
        group,
    )


def test_unknown_preset_fails_closed():
    with pytest.raises(ValueError, match="Unknown KVarN cache dtype"):
        KVarNConfig.from_cache_dtype("kvarn_k3v2_g128", head_dim=256)


@pytest.mark.parametrize("cache_dtype", KVARN_PRESETS)
def test_kvarn_presets_are_registered_as_quantized_cache_modes(cache_dtype):
    assert get_kv_quant_mode(cache_dtype).is_kvarn
    assert is_quantized_kv_cache(cache_dtype)


@pytest.mark.parametrize(
    ("name", "raw_tile_bytes", "aligned_tile_bytes", "slot_bytes"),
    [
        ("kvarn_k4v2_g128", 26880, 32768, 256),
        ("kvarn_k4v4_g128", 35072, 65536, 512),
        ("kvarn_k4v4_g128_compact", 35072, 35072, 274),
        ("kvarn_k4v2_g64", 14208, 16384, 256),
        ("kvarn_k4v4_g64", 18304, 32768, 512),
    ],
)
def test_head_dim_256_layout_is_contiguous_and_page_aligned(
    name, raw_tile_bytes, aligned_tile_bytes, slot_bytes
):
    config = KVarNConfig.from_cache_dtype(name, head_dim=256)
    assert config.tile_bytes == raw_tile_bytes
    assert config.tile_bytes_aligned == aligned_tile_bytes
    assert config.tile_bytes_aligned // config.group == slot_bytes
    assert config.tile_bytes_aligned >= config.tile_bytes

    regions = [
        (config.k_packed_offset, config.k_packed_bytes),
        (config.k_s_col_offset, config.head_dim * 2),
        (config.k_zp_offset, config.head_dim * 2),
        (config.k_s_row_offset, config.group * 2),
        (config.v_packed_offset, config.v_packed_bytes),
        (config.v_s_col_offset, config.head_dim * 2),
        (config.v_s_row_offset, config.group * 2),
        (config.v_zp_offset, config.group * 2),
    ]
    assert regions[0][0] == 0
    for (offset, size), (next_offset, _) in zip(regions, regions[1:]):
        assert offset + size == next_offset
    assert regions[-1][0] + regions[-1][1] == config.tile_bytes


def test_compact_d256_k4v4_record_has_no_power_of_two_padding():
    config = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128_compact", head_dim=256)
    assert config.compact_records
    assert config.record_bytes == config.tile_bytes == 35_072
    assert 4 * config.record_bytes == 140_288
    spec = KVarNAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=128,
            num_kv_heads=4,
            head_size=256,
            dtype=torch.bfloat16,
            kv_quant_mode=get_kv_quant_mode("kvarn_k4v4_g128_compact"),
        )
    )
    assert spec.page_size_bytes == 140_288
    assert spec.state_content_size_bytes == 35_072


def test_pool_accounts_for_default_bounded_prefill_window(monkeypatch):
    monkeypatch.delenv("KVARN_PREFILL_FP16_WINDOW_BLOCKS", raising=False)
    config = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128_compact", head_dim=256)

    # B1/MNBT=2048: sink + tail + 16 recent blocks + 16 current blocks +
    # eight slots of allocator headroom.
    assert kvarn_prefill_fp16_window_blocks() == 16
    assert config.pool_slots(1, 2048) == 42


def test_pool_and_concurrency_use_the_same_prefill_window(monkeypatch):
    monkeypatch.setenv("KVARN_PREFILL_FP16_WINDOW_BLOCKS", "4")
    config = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128_compact", head_dim=256)
    slot_bytes = config._slot_bytes_per_layer(num_kv_heads=4)
    num_layers = 16
    max_slots = 100

    assert config.pool_slots(4, 2048) == 48
    assert (
        config.max_supported_seqs(
            total_gpu_bytes=max_slots * slot_bytes * num_layers,
            num_kv_heads=4,
            num_layers=num_layers,
            max_num_batched_tokens=2048,
            frac=1.0,
        )
        == 12
    )


class _ModelConfig:
    def __init__(self, attention_layers, total_layers=64):
        self.attention_layers = attention_layers
        self.total_layers = total_layers

    def get_num_layers_by_block_type(self, parallel_config, block_type):
        assert block_type == "attention"
        if self.attention_layers is None:
            raise RuntimeError("block-type metadata unavailable")
        return self.attention_layers

    def get_num_layers(self, parallel_config):
        return self.total_layers


def test_hybrid_pool_counts_only_full_attention_layers():
    model = _ModelConfig(attention_layers=16, total_layers=64)
    assert KVarNConfig.num_kvarn_layers(model, SimpleNamespace()) == 16


def test_layer_count_falls_back_when_block_metadata_is_unavailable():
    model = _ModelConfig(attention_layers=None, total_layers=64)
    assert KVarNConfig.num_kvarn_layers(model, SimpleNamespace()) == 64


def test_attention_free_pipeline_rank_does_not_allocate_a_pool():
    model = _ModelConfig(attention_layers=0, total_layers=16)
    assert KVarNConfig.num_kvarn_layers(model, SimpleNamespace()) == 0


@pytest.mark.parametrize("cache_dtype", KVARN_PRESETS)
def test_xpu_routes_every_kvarn_preset_to_kvarn_backend(cache_dtype):
    selector = AttentionSelectorConfig(
        head_size=256,
        dtype=torch.bfloat16,
        kv_cache_dtype=cache_dtype,
        block_size=128,
    )
    assert (
        XPUPlatform.get_attn_backend_cls(AttentionBackendEnum.FLASH_ATTN, selector)
        == AttentionBackendEnum.KVARN.get_path()
    )


def test_kvarn_current_layout_uses_one_packed_record_per_tile():
    base = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.uint8,
        kv_quant_mode=get_kv_quant_mode("kvarn_k4v4_g128"),
    )
    kvarn = KVarNAttentionBackend.customize_spec(base)
    config = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128", head_dim=256)
    assert kvarn.tokens_per_state == config.group
    assert kvarn.state_content_size_bytes == config.tile_bytes_aligned
    assert compute_layer_kv_cache_shape_bytes(kvarn, 3) == (
        3,
        1,
        1,
        config.tile_bytes_aligned,
    )

    layer_stride, block_stride, _, _, _ = compute_layout_strides(
        kvarn, 3, 1, KVCacheLayout.LBHNC
    )
    tensor = KVCacheTensor(
        size=kvarn.page_size_bytes * 3,
        layers=["kvarn"],
        layer_stride=layer_stride,
        block_stride=block_stride,
    )
    raw = torch.empty(tensor.size, dtype=torch.int8)
    (view,) = create_kv_cache_views(raw, kvarn, 3, KVCacheLayout.LBHNC, tensor)
    assert view.dtype == torch.uint8
    assert view.shape == (3, 1, 1, config.tile_bytes_aligned)
    records = KVarNAttentionImpl._record_cache_view(view)
    assert records.shape == (3, 1, config.tile_bytes_aligned)
    assert records.stride() == view.squeeze(2).stride()


def test_kvarn_forward_profiles_with_empty_cache_sentinel():
    impl = object.__new__(KVarNAttentionImpl)
    impl.kvarn_config = SimpleNamespace(record_bytes=35072)
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.head_size = 4
    pool_calls = []
    impl._ensure_pool = lambda device, num_blocks_hint: pool_calls.append(
        (device, num_blocks_hint)
    )
    impl._prefill_first_chunk = lambda q, k, v, metadata, cache: torch.full_like(q, 2)
    metadata = SimpleNamespace(
        num_actual_tokens=3,
        is_prefill=True,
        num_decodes=0,
        num_decode_tokens=0,
        has_cached_multiquery=False,
        vq_seqlen=None,
    )
    query = torch.zeros(3, 8, dtype=torch.float16)
    key = torch.zeros(3, 4, dtype=torch.float16)
    value = torch.zeros_like(key)
    empty_cache = torch.empty(0, dtype=torch.uint8)

    output = impl.forward(None, query, key, value, empty_cache, metadata)

    assert pool_calls == [(query.device, 0)]
    assert output.shape == query.shape
    assert torch.equal(output, torch.full_like(query, 2))
    assert not hasattr(impl, "_kv_cache_ref")


@pytest.mark.parametrize("output_ndim", [2, 3])
def test_kvarn_forward_copies_into_provided_output(output_ndim):
    impl = object.__new__(KVarNAttentionImpl)
    impl.kvarn_config = SimpleNamespace(record_bytes=35072)
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.head_size = 4
    impl._ensure_pool = lambda device, num_blocks_hint: None
    attention = torch.arange(16, dtype=torch.float16).view(2, 2, 4) / 8
    impl._prefill_first_chunk = lambda q, k, v, metadata, cache: attention
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        is_prefill=True,
        num_decodes=0,
        num_decode_tokens=0,
        has_cached_multiquery=False,
        vq_seqlen=None,
    )
    query = torch.zeros(3, 8, dtype=torch.bfloat16)
    key = torch.zeros(3, 4, dtype=torch.bfloat16)
    value = torch.zeros_like(key)
    output_shape = (3, 2, 4) if output_ndim == 3 else (3, 8)
    output = torch.full(output_shape, -123, dtype=torch.bfloat16)

    result = impl.forward(
        None,
        query,
        key,
        value,
        torch.empty(0, dtype=torch.uint8),
        metadata,
        output=output,
    )

    expected = attention if output_ndim == 3 else attention.reshape(2, 8)
    assert result is output
    assert torch.equal(output[:2], expected.to(torch.bfloat16))
    assert torch.equal(output[2], torch.full_like(output[2], -123))


def test_immediately_recycled_sink_is_delabeled_outside_new_row_zero():
    """LIFO reuse must not carry a finished request's sink role into history."""
    old_sink = 5
    new_sink = 7
    sinks = {old_sink, new_sink}
    is_sink_t = torch.zeros(16, dtype=torch.bool)
    # The failure requires immediate reuse, before an absence step can retire it.
    retired_sinks = {}
    is_sink_t[list(sinks)] = True

    _reconcile_kvarn_sink_ownership(
        sinks=sinks,
        retired_sinks=retired_sinks,
        blocks_needed={new_sink, old_sink},
        row0_set={new_sink},
        is_sink_t=is_sink_t,
    )

    assert sinks == {new_sink}
    assert not is_sink_t[old_sink]
    assert is_sink_t[new_sink]


def test_deferred_prefill_flush_retains_committed_history(monkeypatch):
    monkeypatch.setenv("VLLM_KVARN_DEFER_PREFILL_FLUSH", "1")
    monkeypatch.setenv("KVARN_PREFILL_FP16_WINDOW_BLOCKS", "1")
    row = [10, 11, 12, 13]
    resident = {10: 0, 11: 1, 12: 2, 99: 3}
    blocks_needed = {13}
    deferred_blocks: set[int] = set()

    defer = _defer_kvarn_prefill_history_blocks(
        row,
        q_len=256,
        committed_len=384,
        group=128,
        bt_cols=len(row),
        resident_blocks=resident,
        blocks_needed=blocks_needed,
        deferred_blocks=deferred_blocks,
    )
    flush_seen: set[int] = set()
    walk_back = _kvarn_walk_back_flush_blocks(
        row,
        committed_len=384,
        group=128,
        bt_cols=len(row),
        resident_blocks=resident,
        sinks={10},
        flush_seen=flush_seen,
        defer=defer,
        deferred_blocks=deferred_blocks,
    )
    shared_owner_walk_back = _kvarn_walk_back_flush_blocks(
        row,
        committed_len=384,
        group=128,
        bt_cols=len(row),
        resident_blocks=resident,
        sinks={10},
        flush_seen=flush_seen,
        defer=False,
        deferred_blocks=deferred_blocks,
    )
    reclaim = _kvarn_reclaimable_block_ids(resident, blocks_needed, flush_seen)

    assert defer
    assert blocks_needed == {10, 11, 12, 13}
    assert deferred_blocks == {10, 11, 12}
    assert walk_back == []
    assert shared_owner_walk_back == []
    assert flush_seen == set()
    assert reclaim == [99]


@pytest.mark.parametrize(
    ("flag", "q_len", "committed_len"),
    [
        (None, 256, 384),  # default continuation behavior
        ("1", 1, 384),  # decode remains on the production policy
        ("1", 256, 0),  # fresh prefill has no committed history
    ],
)
def test_deferred_prefill_flush_does_not_change_other_steps(
    monkeypatch, flag, q_len, committed_len
):
    if flag is None:
        monkeypatch.delenv("VLLM_KVARN_DEFER_PREFILL_FLUSH", raising=False)
    else:
        monkeypatch.setenv("VLLM_KVARN_DEFER_PREFILL_FLUSH", flag)
    row = [10, 11, 12, 13]
    resident = {10: 0, 11: 1, 12: 2, 99: 3}
    blocks_needed = {13}

    defer = _defer_kvarn_prefill_history_blocks(
        row,
        q_len=q_len,
        committed_len=committed_len,
        group=128,
        bt_cols=len(row),
        resident_blocks=resident,
        blocks_needed=blocks_needed,
    )
    flush_seen: set[int] = set()
    walk_back = _kvarn_walk_back_flush_blocks(
        row,
        committed_len=committed_len,
        group=128,
        bt_cols=len(row),
        resident_blocks=resident,
        sinks={10},
        flush_seen=flush_seen,
        defer=defer,
    )
    reclaim = _kvarn_reclaimable_block_ids(resident, blocks_needed, flush_seen)

    assert not defer
    assert blocks_needed == {13}
    if committed_len:
        assert walk_back == [12, 11]
        assert flush_seen == {11, 12}
        assert reclaim == [10, 99]
    else:
        assert walk_back == []
        assert flush_seen == set()
        assert reclaim == [10, 11, 12, 99]


def test_prefill_fp16_window_chunk2_retains_available_recent_history(monkeypatch):
    monkeypatch.delenv("VLLM_KVARN_DEFER_PREFILL_FLUSH", raising=False)
    monkeypatch.setenv("KVARN_PREFILL_FP16_WINDOW_BLOCKS", "16")
    row = list(range(32))
    resident = {bid: bid for bid in range(16)}
    blocks_needed = {0, *range(16, 32)}
    protected: set[int] = set()

    active = _protect_kvarn_prefill_window_blocks(
        row,
        q_len=2048,
        committed_len=2048,
        group=128,
        bt_cols=len(row),
        window=_kvarn_prefill_fp16_window_blocks(),
        resident_blocks=resident,
        blocks_needed=blocks_needed,
        protected_blocks=protected,
    )
    flush_seen: set[int] = set()
    walk_back = _kvarn_walk_back_flush_blocks(
        row,
        committed_len=2048,
        group=128,
        bt_cols=len(row),
        resident_blocks=resident,
        sinks={0},
        flush_seen=flush_seen,
        defer=False,
        deferred_blocks=protected,
    )
    reclaim = _kvarn_reclaimable_block_ids(resident, blocks_needed, flush_seen)

    assert active
    # Chunk 1 contains only 15 non-sink history blocks, so W=16 retains all
    # available history without counting the independently protected sink.
    assert protected == set(range(1, 16))
    assert walk_back == []
    assert reclaim == []
    assert len(blocks_needed) == 32


def test_prefill_fp16_window_chunk3_flushes_older_than_bounded_suffix(monkeypatch):
    monkeypatch.delenv("VLLM_KVARN_DEFER_PREFILL_FLUSH", raising=False)
    monkeypatch.setenv("KVARN_PREFILL_FP16_WINDOW_BLOCKS", "16")
    row = list(range(48))
    resident = {bid: bid for bid in range(32)}
    blocks_needed = {0, *range(32, 48)}
    protected: set[int] = set()

    active = _protect_kvarn_prefill_window_blocks(
        row,
        q_len=2048,
        committed_len=4096,
        group=128,
        bt_cols=len(row),
        window=_kvarn_prefill_fp16_window_blocks(),
        resident_blocks=resident,
        blocks_needed=blocks_needed,
        protected_blocks=protected,
    )
    flush_seen: set[int] = set()
    walk_back = _kvarn_walk_back_flush_blocks(
        row,
        committed_len=4096,
        group=128,
        bt_cols=len(row),
        resident_blocks=resident,
        sinks={0},
        flush_seen=flush_seen,
        defer=False,
        deferred_blocks=protected,
    )
    reclaim = _kvarn_reclaimable_block_ids(resident, blocks_needed, flush_seen)

    assert active
    assert protected == set(range(16, 32))
    # Walk-back skips the protected suffix and continues through all older
    # resident non-sink history instead of stopping at block 31.
    assert walk_back == list(range(15, 0, -1))
    assert flush_seen == set(range(1, 16))
    assert reclaim == []
    remaining_after_flush = (set(resident) - flush_seen) | set(range(32, 48))
    assert remaining_after_flush == {0, *range(16, 48)}
    assert len(remaining_after_flush) == 33


@pytest.mark.parametrize(
    ("window", "q_len", "committed_len"),
    [
        (0, 256, 384),  # default continuation behavior
        (16, 1, 384),  # decode remains on the production policy
        (16, 256, 0),  # fresh prefill has no committed history
    ],
)
def test_prefill_fp16_window_does_not_change_other_steps(
    monkeypatch, window, q_len, committed_len
):
    monkeypatch.delenv("VLLM_KVARN_DEFER_PREFILL_FLUSH", raising=False)
    monkeypatch.setenv("KVARN_PREFILL_FP16_WINDOW_BLOCKS", str(window))
    row = [10, 11, 12, 13]
    resident = {10: 0, 11: 1, 12: 2, 99: 3}
    blocks_needed = {13}
    protected: set[int] = set()

    active = _protect_kvarn_prefill_window_blocks(
        row,
        q_len=q_len,
        committed_len=committed_len,
        group=128,
        bt_cols=len(row),
        window=_kvarn_prefill_fp16_window_blocks(),
        resident_blocks=resident,
        blocks_needed=blocks_needed,
        protected_blocks=protected,
    )
    flush_seen: set[int] = set()
    walk_back = _kvarn_walk_back_flush_blocks(
        row,
        committed_len=committed_len,
        group=128,
        bt_cols=len(row),
        resident_blocks=resident,
        sinks={10},
        flush_seen=flush_seen,
        defer=False,
        deferred_blocks=protected,
    )

    assert not active
    assert protected == set()
    assert blocks_needed == {13}
    assert walk_back == ([12, 11] if committed_len else [])


@pytest.mark.parametrize("value", ["-1", "not-an-integer"])
def test_prefill_fp16_window_rejects_invalid_values(monkeypatch, value):
    monkeypatch.setenv("KVARN_PREFILL_FP16_WINDOW_BLOCKS", value)

    with pytest.raises(ValueError, match="must be a non-negative integer"):
        _kvarn_prefill_fp16_window_blocks()


def test_prefill_fp16_window_defaults_to_beta_guardrail(monkeypatch):
    monkeypatch.delenv("KVARN_PREFILL_FP16_WINDOW_BLOCKS", raising=False)

    assert _kvarn_prefill_fp16_window_blocks() == 16


def test_kvarn_rotation_uses_2d_gemm_and_preserves_scratch_canary(monkeypatch):
    """The XPU batched matmul writer must not return to the K/V store path."""
    num_tokens, num_heads, head_dim = 3, 2, 4
    generator = torch.Generator().manual_seed(0)
    key = torch.randn(
        num_tokens, num_heads, head_dim, generator=generator, dtype=torch.float32
    )
    value = torch.randn(
        num_tokens, num_heads, head_dim, generator=generator, dtype=torch.float32
    )
    hadamard = torch.randn(head_dim, head_dim, generator=generator, dtype=torch.float32)

    k_storage = torch.empty(num_tokens + 1, num_heads, head_dim, dtype=torch.float32)
    v_storage = torch.empty_like(k_storage)
    k_canary = torch.arange(num_heads * head_dim, dtype=torch.float32).view(
        num_heads, head_dim
    )
    v_canary = -k_canary - 1
    k_storage[num_tokens].copy_(k_canary)
    v_storage[num_tokens].copy_(v_canary)

    original_mm = torch.mm
    expected_k = original_mm(
        key.reshape(num_tokens * num_heads, head_dim), hadamard
    ).view_as(key)
    expected_v = original_mm(
        value.reshape(num_tokens * num_heads, head_dim), hadamard
    ).view_as(value)
    gemm_shapes = []

    def checked_mm(left, right, *, out=None):
        gemm_shapes.append(
            (left.shape, right.shape, out.shape if out is not None else None)
        )
        assert left.ndim == right.ndim == 2
        assert out is not None and out.ndim == 2
        return original_mm(left, right, out=out)

    monkeypatch.setattr(torch, "mm", checked_mm)
    _rotate_kvarn_kv_into_scratch(
        key,
        value,
        hadamard,
        k_storage[:num_tokens],
        v_storage[:num_tokens],
    )

    flat_shape = torch.Size((num_tokens * num_heads, head_dim))
    assert gemm_shapes == [
        (flat_shape, hadamard.shape, flat_shape),
        (flat_shape, hadamard.shape, flat_shape),
    ]
    torch.testing.assert_close(k_storage[:num_tokens], expected_k)
    torch.testing.assert_close(v_storage[:num_tokens], expected_v)
    assert torch.equal(k_storage[num_tokens], k_canary)
    assert torch.equal(v_storage[num_tokens], v_canary)


def test_kvarn_metadata_stage_waits_before_reuse_and_uses_fresh_events():
    class FakeEvent:
        def __init__(self, generation):
            self.generation = generation
            self.records = 0
            self.synchronizes = 0

        def record(self):
            self.records += 1

        def synchronize(self):
            self.synchronizes += 1

    events = []

    def new_event():
        event = FakeEvent(len(events))
        events.append(event)
        return event

    ring = _KVarNMetadataStageRing(depth=2)
    assert ring.acquire() == 0
    first = ring.release(0, new_event)
    assert ring.acquire() == 1
    second = ring.release(1, new_event)

    # Wrapping to slot zero must wait for its previous DMA generation before
    # the caller is allowed to mutate that pinned host row.
    assert ring.acquire() == 0
    assert first.synchronizes == 1
    replacement = ring.release(0, new_event)

    # An event is never re-recorded for a later generation.
    assert replacement is not first
    assert first.records == replacement.records == 1
    assert second.synchronizes == 0

    ring.drain()
    assert second.synchronizes == 1
    assert replacement.synchronizes == 1


def test_kvarn_page_unification_scales_by_whole_quantization_tiles():
    kvarn = KVarNAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=256,
            head_size_v=256,
            dtype=torch.bfloat16,
            kv_quant_mode=get_kv_quant_mode("kvarn_k4v4_g128"),
        )
    )
    native = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    unified = unify_kv_cache_spec_page_size({"kvarn": kvarn, "native": native})
    assert unified["kvarn"].block_size % kvarn.tokens_per_state == 0
    assert unified["kvarn"].page_size_bytes == unified["native"].page_size_bytes


def test_turboquant_page_unification_retains_scaling_behavior():
    turboquant = TurboQuantAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=128,
            num_kv_heads=1,
            head_size=256,
            head_size_v=256,
            dtype=torch.bfloat16,
            kv_quant_mode=get_kv_quant_mode("turboquant_4bit_nc"),
        )
    )
    native = FullAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=256,
        head_size_v=256,
        dtype=torch.bfloat16,
    )
    unified = unify_kv_cache_spec_page_size(
        {"turboquant": turboquant, "native": native}
    )
    assert unified["turboquant"].block_size >= turboquant.block_size
    assert unified["turboquant"].page_size_bytes == unified["native"].page_size_bytes
