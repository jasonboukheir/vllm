# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for KVarN configuration and cache accounting."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.kvarn.config import (
    KVARN_PRESETS,
    KVarNConfig,
    kvarn_decode_fp16_low_water_blocks,
    kvarn_decode_fp16_window_blocks,
    kvarn_prefill_fp16_window_blocks,
)
from vllm.platforms.xpu import XPUPlatform
from vllm.v1.attention.backends.kvarn_attn import (
    KVarNAttentionBackend,
    KVarNAttentionImpl,
    KVarNMetadata,
    KVarNMetadataBuilder,
    _can_elide_fa_cu_seqlens,
    _cast_kvarn_activations,
    _coordinate_kvarn_decode_window_blocks,
    _is_pure_kvarn_cached_prefill_step,
    _is_pure_kvarn_decode_step,
    _is_pure_kvarn_prefill_step,
    _is_pure_qlen1_batch,
    _kvarn_block_table_numpy,
    _kvarn_decode_flush_scope,
    _kvarn_decode_fp16_low_water_blocks,
    _kvarn_decode_fp16_window_blocks,
    _kvarn_decode_resident_suffix,
    _kvarn_native_balanced_writer_supported,
    _kvarn_prefill_fp16_window_blocks,
    _kvarn_reclaimable_block_ids,
    _kvarn_walk_back_flush_blocks,
    _KVarNMetadataStageRing,
    _KVarNQlen1MetadataKind,
    _launch_kvarn_native_balanced_writer,
    _protect_kvarn_decode_window_blocks,
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
from vllm.v1.attention.ops.triton_kvarn_decode import (
    KVARN_CACHE_LAYOUT_XE2_DPAS,
    KVARN_FRONTEND_QKV_SCATTER_INLINE_CURRENT_STREAM,
    KVARN_NATIVE_KERNEL_Q6_PREFETCH_RECORD_CURSOR,
    KVARN_NATIVE_SPLIT_POLICY_B70_Q6_ID18_V1,
    KVARN_PREFILL_STORE_HADAMARD_SCATTER,
    kvarn_cache_layout_requested,
    kvarn_frontend_variant_requested,
    kvarn_native_kernel_variant_requested,
    kvarn_native_split_count,
    kvarn_native_split_policy_requested,
    kvarn_prefill_store_variant_requested,
)
from vllm.v1.attention.selector import AttentionSelectorConfig
from vllm.v1.core.kv_cache_utils import unify_kv_cache_spec_page_size
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheLayout,
    KVCacheTensor,
    KVQuantMode,
    compute_layer_kv_cache_shape_bytes,
    compute_layout_strides,
    create_kv_cache_views,
    get_kv_quant_mode,
    is_quantized_kv_cache,
)


def test_kvarn_xpu_decode_uses_only_qualified_native_configuration():
    assert kvarn_cache_layout_requested(KVARN_CACHE_LAYOUT_XE2_DPAS) == "xe2_dpas"
    assert (
        kvarn_frontend_variant_requested(
            KVARN_FRONTEND_QKV_SCATTER_INLINE_CURRENT_STREAM
        )
        == "qkv_scatter_inline_current_stream"
    )
    assert (
        kvarn_prefill_store_variant_requested(KVARN_PREFILL_STORE_HADAMARD_SCATTER)
        == "hadamard_scatter"
    )
    assert kvarn_native_kernel_variant_requested("q6_prefetch_record_cursor") == (
        "q6_prefetch_record_cursor",
        KVARN_NATIVE_KERNEL_Q6_PREFETCH_RECORD_CURSOR,
    )
    assert kvarn_native_split_policy_requested(
        KVARN_NATIVE_SPLIT_POLICY_B70_Q6_ID18_V1
    ) == ("b70_q6_id18_v1", 32)
    assert [
        kvarn_native_split_count(
            65_023,
            32,
            batch_size=batch_size,
            split_policy=KVARN_NATIVE_SPLIT_POLICY_B70_Q6_ID18_V1,
            kernel_variant=KVARN_NATIVE_KERNEL_Q6_PREFETCH_RECORD_CURSOR,
        )
        for batch_size in (1, 2, 3, 4, 8, 12)
    ] == [32, 16, 8, 24, 4, 2]


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


def test_cached_prefill_cast_skips_unused_key_and_value():
    query, key, value = (torch.randn(2, 3, dtype=torch.bfloat16) for _ in range(3))

    cast_query, cast_key, cast_value = _cast_kvarn_activations(
        query,
        key,
        value,
        query_only=False,
        key_value_unused=True,
    )

    assert cast_query.dtype == torch.float16
    assert cast_key is key
    assert cast_value is value


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
        ({"is_prefill": False}, False),
        ({"max_query_len": 1}, False),
        ({"num_decodes": 1}, False),
        ({"num_decode_tokens": 1}, False),
        ({"num_actual_tokens": 3}, False),
    ],
)
def test_native_prefill_store_requires_a_pure_multi_token_prefill(override, expected):
    values = dict(
        is_prefill=True,
        max_query_len=4,
        num_decodes=0,
        num_decode_tokens=0,
        num_actual_tokens=4,
    )
    values.update(override)

    assert _is_pure_kvarn_prefill_step(SimpleNamespace(**values), 4) is expected
    if not override:
        assert not _is_pure_kvarn_prefill_step(SimpleNamespace(**values), 1)


def _fused_frontend_impl() -> KVarNAttentionImpl:
    impl = object.__new__(KVarNAttentionImpl)
    impl._kvarn_frontend_variant = "qkv_scatter"
    impl._kvarn_qkv_scatter_op_name = "kvarn_hadamard_qkv_scatter"
    impl._kvarn_qkv_scatter_op = Mock()
    impl._kvarn_frontend_bound = True
    impl._native_qkv_scatter_active_logged = False
    impl.use_fused_qkv_cache_update = True
    impl.use_inline_qkv_cache_update = False
    impl._pending_fused_qkv_signature = None
    impl._forward_pool_elision_active_logged = False
    impl._kvarn_qlen1_inline_plan = "reference"
    impl.use_trusted_qlen1_inline_plan = False
    impl._trusted_qlen1_inline_execution_logged = False
    impl._trusted_qlen1_inline_bound = None
    impl.use_bound_qlen1_inline_plan_v2 = False
    impl._bound_qlen1_inline_v2_execution_logged = False
    impl._bound_qlen1_inline_v2_binding_epoch = 0
    impl._bound_qlen1_inline_v2_plan = None
    impl.num_heads = 24
    impl.num_kv_heads = 4
    impl.head_size = 256
    impl.sliding_window = 0
    impl._group_key = (256, 4, 0)
    impl._pool_ready_key = (torch.device("cpu"), impl._group_key)
    impl._kv_cache_ref = None
    impl._kvarn_forward_pool_ensure = "always"
    impl.layer_name = "model.layers.0.self_attn"
    impl._kvarn_dpas_layout = True
    impl._kvarn_cache_layout = "xe2_dpas"
    impl._kvarn_native_split_policy = "fixed"
    impl._kvarn_native_max_splits = 16
    impl._kvarn_native_kernel_variant = 18
    impl._max_num_seqs = 12
    impl.kvarn_config = SimpleNamespace(
        group=128,
        key_bits=4,
        value_bits=4,
        record_bytes=35_072,
    )
    impl._block_to_slot_t = torch.zeros(2, dtype=torch.int32)
    impl._tail_K_pool = torch.empty(2, 128, 4, 256, dtype=torch.float16)
    impl._tail_V_pool = torch.empty_like(impl._tail_K_pool)
    impl._q_rot_fp16_buf = torch.empty(48, 256, dtype=torch.float16)
    impl._fused_out_buf = torch.empty_like(impl._q_rot_fp16_buf)
    impl._native_output_fp16_buf = torch.empty_like(impl._q_rot_fp16_buf)
    impl._native_decode_scratch = (
        torch.empty(12, 24 * 16, 256, dtype=torch.float16),
        torch.empty(12, 24, 16, dtype=torch.float32),
        torch.empty(12, 24, 16, dtype=torch.float32),
    )
    return impl


def _prefill_scatter_impl() -> KVarNAttentionImpl:
    impl = _fused_frontend_impl()
    impl._kvarn_prefill_store_variant = "hadamard_scatter"
    impl.use_fused_qkv_cache_update = False
    return impl


def _pure_prefill_metadata(tokens: int = 4) -> SimpleNamespace:
    return SimpleNamespace(
        is_prefill=True,
        max_query_len=tokens,
        num_decodes=0,
        num_decode_tokens=0,
        num_actual_tokens=tokens,
    )


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({}, True),
        ({"has_cached_multiquery": False}, False),
        ({"is_prefill": False}, False),
        ({"max_query_len": 1}, False),
        ({"num_decodes": 1}, False),
        ({"num_actual_tokens": 16}, False),
    ],
)
def test_native_materializer_requires_pure_cached_prefill(
    override: dict, expected: bool
) -> None:
    values = vars(_pure_prefill_metadata(tokens=17)) | {
        "max_query_len": 11,
        "has_cached_multiquery": True,
    }
    values.update(override)

    assert _is_pure_kvarn_cached_prefill_step(SimpleNamespace(**values), 17) is expected


def _cached_prefill_impl() -> KVarNAttentionImpl:
    impl = object.__new__(KVarNAttentionImpl)
    impl.layer_name = "model.layers.0.self_attn"
    impl.num_kv_heads = 4
    impl.head_size = 256
    impl.kvarn_config = KVarNConfig(
        head_dim=256,
        key_bits=4,
        value_bits=4,
        group=128,
        compact_records=True,
    )
    impl._kvarn_cache_layout = "xe2_dpas"
    impl._kvarn_dpas_layout = True
    impl._kvarn_cached_prefill_materializer = "native_xe2"
    impl._block_lookup_size = 16
    impl._block_to_slot_t = torch.arange(16, dtype=torch.int32)
    impl._tail_K_pool = torch.empty(16, 128, 4, 256, dtype=torch.float16)
    impl._tail_V_pool = torch.empty_like(impl._tail_K_pool)
    impl._fa_K_buf = torch.empty(2048, 4, 256, dtype=torch.float16)
    impl._fa_V_buf = torch.empty_like(impl._fa_K_buf)
    return impl


def _cached_prefill_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        is_prefill=True,
        max_query_len=11,
        num_decodes=0,
        num_decode_tokens=0,
        num_actual_tokens=17,
        has_cached_multiquery=True,
        seq_lens=torch.tensor([257, 1535], dtype=torch.int32),
        seq_lens_cpu=[257, 1535],
        fa_cu_seqlens_k=torch.tensor([0, 257, 1792], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 6, 17], dtype=torch.int32),
        block_table=torch.zeros(2, 12, dtype=torch.int32),
        max_seq_len=1535,
        causal=True,
    )


def test_cached_prefill_materializer_selection_is_frozen_and_logged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    KVarNAttentionImpl.reset_process_state()
    try:
        enabled = Mock(return_value=True)
        marker = Mock()
        monkeypatch.setattr(kvarn_attn, "kvarn_native_feature_enabled", enabled)
        monkeypatch.setattr(kvarn_attn.logger, "info_once", marker)

        assert KVarNAttentionImpl._select_cached_prefill_materializer() == (
            "native_xe2"
        )
        enabled.return_value = False
        assert KVarNAttentionImpl._select_cached_prefill_materializer() == (
            "native_xe2"
        )

        enabled.assert_called_once_with("MATERIALIZE")
        marker.assert_called_once_with(
            "[KVARN_FACTORY] selected_cached_prefill_materializer=%s; "
            "selectors=KVARN_NATIVE_XPU,KVARN_NATIVE_XPU_MATERIALIZE; "
            "eligibility_fallback=reference; immutable for engine lifetime",
            "native_xe2",
        )
    finally:
        KVarNAttentionImpl.reset_process_state()


def test_native_cached_prefill_materializer_preserves_exact_op_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _cached_prefill_impl()
    cache = torch.empty(16, 4, 35_072, dtype=torch.uint8)
    metadata = _cached_prefill_metadata()
    key_output = impl._fa_K_buf
    value_output = impl._fa_V_buf
    call = Mock()
    fake_ops = SimpleNamespace(kvarn_materialize_packed_kv=call)
    monkeypatch.setattr(kvarn_attn.torch.ops, "_vllm_fa2_C", fake_ops)

    impl._launch_native_cached_prefill_materializer(
        cache,
        metadata.block_table,
        metadata.seq_lens,
        metadata.fa_cu_seqlens_k,
        key_output,
        value_output,
        metadata.max_seq_len,
    )

    call.assert_called_once_with(
        cache,
        metadata.block_table,
        metadata.seq_lens,
        metadata.fa_cu_seqlens_k,
        impl._block_to_slot_t,
        impl._tail_K_pool,
        impl._tail_V_pool,
        key_output,
        value_output,
        1535,
        True,
    )


def test_cached_prefill_native_dispatch_reuses_builder_metadata_without_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _cached_prefill_impl()
    metadata = _cached_prefill_metadata()
    cache = torch.empty(16, 4, 35_072, dtype=torch.uint8)
    query = torch.empty(17, 24, 256, dtype=torch.float16)
    launch = Mock()

    KVarNAttentionImpl._cached_prefill_materializer_counters.clear()
    monkeypatch.setattr(
        impl,
        "_native_cached_prefill_materializer_eligibility",
        lambda *args, **kwargs: (True, "eligible"),
    )
    monkeypatch.setattr(impl, "_launch_native_cached_prefill_materializer", launch)
    monkeypatch.setattr(
        impl,
        "_launch_reference_cached_prefill_materializer",
        Mock(side_effect=AssertionError("reference materializer executed")),
    )
    monkeypatch.setattr(
        torch.Tensor,
        "item",
        Mock(side_effect=AssertionError("device scalar synchronized")),
    )
    monkeypatch.setattr(
        torch,
        "cumsum",
        Mock(side_effect=AssertionError("cu-seqlens rebuilt")),
    )
    marker = Mock()
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    monkeypatch.setattr(kvarn_attn.logger, "info_once", marker)
    monkeypatch.setattr(
        kvarn_attn.F,
        "pad",
        Mock(side_effect=AssertionError("cu-seqlens padded")),
    )

    key_output, value_output, cu_k, total_k, max_seq_len = (
        impl._materialize_cached_prefill_kv(query, cache, metadata)
    )

    assert key_output is impl._fa_K_buf
    assert value_output is impl._fa_V_buf
    assert cu_k.data_ptr() == metadata.fa_cu_seqlens_k.data_ptr()
    assert (total_k, max_seq_len) == (1792, 1535)
    launch.assert_called_once()
    launch_args = launch.call_args.args
    assert launch_args[0] is cache
    assert launch_args[1].data_ptr() == metadata.block_table.data_ptr()
    assert launch_args[2].data_ptr() == metadata.seq_lens.data_ptr()
    assert launch_args[3].data_ptr() == metadata.fa_cu_seqlens_k.data_ptr()
    assert launch_args[4] is impl._fa_K_buf
    assert launch_args[5] is impl._fa_V_buf
    assert launch_args[6] == 1535
    assert KVarNAttentionImpl.cached_prefill_materializer_counters() == {
        "calls": 1,
        "native_launches": 1,
        "reference_launches": 0,
        "selected_native_fallbacks": 0,
    }
    marker.assert_called_once_with(
        "[KVARN_PREFILL_MATERIALIZER] active=native_xe2; layer=%s; "
        "native_op=kvarn_materialize_packed_kv; cache_layout=%s; "
        "eligibility_fallback=reference",
        "model.layers.0.self_attn",
        "xe2_dpas",
    )


def test_native_cached_prefill_materializer_eligibility_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _cached_prefill_impl()
    metadata = _cached_prefill_metadata()
    cache = torch.empty(16, 4, 35_072, dtype=torch.uint8)
    args = (
        cache,
        metadata.block_table,
        metadata.seq_lens,
        metadata.fa_cu_seqlens_k,
        impl._fa_K_buf,
        impl._fa_V_buf,
        metadata,
    )
    kwargs = {
        "num_query_tokens": 17,
        "total_k": 1792,
        "max_seq_len": 1535,
    }

    impl._kvarn_cached_prefill_materializer = "reference"
    assert impl._native_cached_prefill_materializer_eligibility(*args, **kwargs) == (
        False,
        "selected_reference",
    )
    impl._kvarn_cached_prefill_materializer = "native_xe2"
    metadata.has_cached_multiquery = False
    assert impl._native_cached_prefill_materializer_eligibility(*args, **kwargs) == (
        False,
        "not_pure_cached_prefill",
    )
    metadata.has_cached_multiquery = True
    impl._block_to_slot_t = None
    assert impl._native_cached_prefill_materializer_eligibility(*args, **kwargs) == (
        False,
        "missing_tensor",
    )
    impl._block_to_slot_t = torch.arange(16, dtype=torch.int32)
    impl._kvarn_cache_layout = "natural"
    assert impl._native_cached_prefill_materializer_eligibility(*args, **kwargs) == (
        False,
        "unsupported_cache_abi",
    )
    impl._kvarn_cache_layout = "xe2_dpas"
    monkeypatch.setattr(
        kvarn_attn, "kvarn_native_layout_abi_supported", lambda *_: False
    )
    assert impl._native_cached_prefill_materializer_eligibility(*args, **kwargs) == (
        False,
        "native_op_unavailable",
    )


def test_final_ragged_cached_prefill_routes_materialized_history_to_flash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _cached_prefill_impl()
    impl.num_heads = 24
    impl._H_fp16 = torch.eye(256, dtype=torch.float16)
    metadata = _cached_prefill_metadata()
    cache = torch.empty(16, 4, 35_072, dtype=torch.uint8)
    query = torch.zeros(17, 24, 256, dtype=torch.float16)
    materialize = Mock(
        return_value=(
            impl._fa_K_buf,
            impl._fa_V_buf,
            metadata.fa_cu_seqlens_k,
            1792,
            1535,
        )
    )
    flash = Mock(side_effect=lambda query, *args, **kwargs: query)

    monkeypatch.setattr(kvarn_attn, "_HAS_FLASH_ATTN", True)
    monkeypatch.setattr(kvarn_attn, "_use_kvarn_fused_verify", lambda **_: False)
    monkeypatch.setattr(impl, "_materialize_cached_prefill_kv", materialize)
    monkeypatch.setattr(impl, "_flash_varlen", flash)

    output = impl._cached_multiquery_path(query, cache, metadata)

    materialize.assert_called_once_with(query, cache, metadata)
    assert output.shape == query.shape
    flash.assert_called_once()
    call = flash.call_args
    assert call.args[0].shape == query.shape
    assert call.args[1].shape == (1792, 4, 256)
    assert call.args[2].shape == (1792, 4, 256)
    assert call.kwargs["cu_q"].data_ptr() == metadata.query_start_loc.data_ptr()
    assert call.kwargs["cu_k"].data_ptr() == metadata.fa_cu_seqlens_k.data_ptr()
    assert call.kwargs["max_q"] == 11
    assert call.kwargs["max_k"] == 1535
    assert call.kwargs["causal"] is True


def test_cached_prefill_native_selection_falls_back_and_counts_reason(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _cached_prefill_impl()
    metadata = _cached_prefill_metadata()
    cache = torch.empty(16, 4, 35_072, dtype=torch.uint8)
    query = torch.empty(17, 24, 256, dtype=torch.float16)
    reference = Mock()
    marker = Mock()

    KVarNAttentionImpl._cached_prefill_materializer_counters.clear()
    monkeypatch.setattr(
        impl,
        "_native_cached_prefill_materializer_eligibility",
        lambda *args, **kwargs: (False, "unsupported_cache_abi"),
    )
    monkeypatch.setattr(
        impl,
        "_launch_native_cached_prefill_materializer",
        Mock(side_effect=AssertionError("native materializer executed")),
    )
    monkeypatch.setattr(
        impl, "_launch_reference_cached_prefill_materializer", reference
    )
    monkeypatch.setattr(kvarn_attn.logger, "info_once", marker)

    impl._materialize_cached_prefill_kv(query, cache, metadata)

    reference.assert_called_once()
    assert reference.call_args.args[3].data_ptr() == (
        metadata.fa_cu_seqlens_k.data_ptr()
    )
    assert KVarNAttentionImpl.cached_prefill_materializer_counters() == {
        "calls": 1,
        "native_launches": 0,
        "reference_launches": 1,
        "selected_native_fallbacks": 1,
    }
    marker.assert_called_once_with(
        "[KVARN_PREFILL_MATERIALIZER] selected=native_xe2; "
        "active=reference; layer=%s; cache_layout=%s; "
        "eligibility_fallback=reference",
        "model.layers.0.self_attn",
        "xe2_dpas",
    )


def test_mixed_cached_prefill_reuses_sliced_builder_metadata_without_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _cached_prefill_impl()
    impl.num_heads = 24
    impl._max_model_len = 2048
    cache = torch.empty(16, 4, 35_072, dtype=torch.uint8)
    query = torch.empty(18, 24, 256, dtype=torch.float16)
    key = torch.empty(18, 4, 256, dtype=torch.float16)
    value = torch.empty_like(key)
    prefill_cu_k = torch.tensor([0, 513, 1537], dtype=torch.int32)
    metadata = KVarNMetadata(
        seq_lens=torch.tensor([257, 513, 1024], dtype=torch.int32),
        slot_mapping=torch.arange(18, dtype=torch.int64),
        block_table=torch.zeros(3, 16, dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1, 7, 18], dtype=torch.int32),
        num_actual_tokens=18,
        max_query_len=11,
        max_seq_len=1024,
        is_prefill=True,
        num_decodes=1,
        num_decode_tokens=1,
        has_cached_multiquery=True,
        prefill_has_cached_multiquery=True,
        seq_lens_cpu=[257, 513, 1024],
        fa_cu_seqlens_k=torch.tensor([0, 257, 770, 1794], dtype=torch.int32),
        prefill_fa_cu_seqlens_k=prefill_cu_k,
    )
    launch = Mock()
    materialized_metadata = None
    cumsum_calls = []
    real_cumsum = torch.cumsum

    def record_cumsum(*args, **kwargs):
        cumsum_calls.append(args[0].shape)
        return real_cumsum(*args, **kwargs)

    def cached_prefill(q, kv_cache, md):
        nonlocal materialized_metadata
        materialized_metadata = md
        impl._materialize_cached_prefill_kv(q, kv_cache, md)
        return torch.zeros_like(q)

    monkeypatch.setattr(torch, "cumsum", record_cumsum)
    monkeypatch.setattr(kvarn_attn.logger, "info_once", Mock())
    monkeypatch.setattr(
        torch.Tensor,
        "item",
        Mock(side_effect=AssertionError("prefill materializer synchronized")),
    )
    monkeypatch.setattr(impl, "_decode_path", lambda q, *_: torch.zeros_like(q))
    monkeypatch.setattr(impl, "_cached_multiquery_path", cached_prefill)
    monkeypatch.setattr(
        impl,
        "_native_cached_prefill_materializer_eligibility",
        lambda *args, **kwargs: (True, "eligible"),
    )
    monkeypatch.setattr(impl, "_launch_native_cached_prefill_materializer", launch)
    monkeypatch.setattr(
        impl,
        "_launch_reference_cached_prefill_materializer",
        Mock(side_effect=AssertionError("reference materializer executed")),
    )

    output = impl._mixed_batch_path(query, key, value, cache, metadata)

    assert output.shape == query.shape
    assert materialized_metadata is not None
    assert materialized_metadata.has_cached_multiquery
    assert materialized_metadata.seq_lens_cpu == [513, 1024]
    assert materialized_metadata.fa_cu_seqlens_k.data_ptr() == prefill_cu_k.data_ptr()
    assert cumsum_calls == [torch.Size([1])]
    launch.assert_called_once()
    assert launch.call_args.args[3].data_ptr() == prefill_cu_k.data_ptr()


@pytest.mark.parametrize(
    ("seq_lens_cpu", "max_seq_len", "message"),
    [
        ([-1, 1535], 1535, "negative sequence length"),
        ([257, 1535], 1024, "max sequence length is inconsistent"),
    ],
)
def test_cached_prefill_rejects_inconsistent_cpu_extents_without_launch(
    monkeypatch: pytest.MonkeyPatch,
    seq_lens_cpu: list[int],
    max_seq_len: int,
    message: str,
) -> None:
    impl = _cached_prefill_impl()
    metadata = _cached_prefill_metadata()
    metadata.seq_lens_cpu = seq_lens_cpu
    metadata.max_seq_len = max_seq_len
    cache = torch.empty(16, 4, 35_072, dtype=torch.uint8)
    query = torch.empty(17, 24, 256, dtype=torch.float16)
    native = Mock()
    reference = Mock()
    monkeypatch.setattr(impl, "_launch_native_cached_prefill_materializer", native)
    monkeypatch.setattr(
        impl, "_launch_reference_cached_prefill_materializer", reference
    )

    with pytest.raises(ValueError, match=message):
        impl._materialize_cached_prefill_kv(query, cache, metadata)

    native.assert_not_called()
    reference.assert_not_called()


def test_native_prefill_scatter_eligibility_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _prefill_scatter_impl()
    metadata = _pure_prefill_metadata()
    key = torch.empty(4, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    slots = torch.tensor([0, 1, 128, 257], dtype=torch.int64)
    layer = SimpleNamespace()

    monkeypatch.setattr(
        kvarn_attn, "kvarn_native_prefill_store_supported", lambda **_: True
    )

    assert impl._native_prefill_scatter_eligible(layer, key, value, slots, metadata)
    impl._kvarn_prefill_store_variant = "reference"
    assert not impl._native_prefill_scatter_eligible(layer, key, value, slots, metadata)
    impl._kvarn_prefill_store_variant = "hadamard_scatter"
    assert not impl._native_prefill_scatter_eligible(
        layer, key, value, slots, _pure_decode_metadata(tokens=4)
    )
    assert not impl._native_prefill_scatter_eligible(
        layer, key[:, :, ::2], value[:, :, ::2], slots, metadata
    )
    assert not impl._native_prefill_scatter_eligible(
        layer, key, value, slots.to(torch.int32), metadata
    )
    impl._block_to_slot_t = None
    assert not impl._native_prefill_scatter_eligible(layer, key, value, slots, metadata)
    impl._block_to_slot_t = torch.zeros(2, dtype=torch.int32)
    impl._tail_V_pool = None
    assert not impl._native_prefill_scatter_eligible(layer, key, value, slots, metadata)


def test_native_prefill_scatter_dispatch_bypasses_reference_writer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _prefill_scatter_impl()
    metadata = _pure_prefill_metadata()
    key = torch.empty(4, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    slots = torch.tensor([0, 1, 128, 257], dtype=torch.int64)
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    layer = SimpleNamespace()
    launch = Mock()
    marker = Mock()

    monkeypatch.setattr(kvarn_attn, "_active_kvarn_metadata", lambda _: metadata)
    monkeypatch.setattr(kvarn_attn.logger, "info_once", marker)
    monkeypatch.setattr(impl, "_ensure_pool", Mock())
    monkeypatch.setattr(impl, "_native_prefill_scatter_eligible", lambda *_: True)
    monkeypatch.setattr(impl, "_launch_native_prefill_scatter", launch)
    monkeypatch.setattr(
        kvarn_attn,
        "_rotate_kvarn_kv_into_scratch",
        Mock(side_effect=AssertionError("reference writer executed")),
    )

    impl.do_kv_cache_update(layer, key, value, cache, slots)

    launch.assert_called_once_with(key, value, slots)
    marker.assert_called_once_with(
        "[KVARN_PREFILL_STORE] active=hadamard_scatter; "
        "layer=%s; native_op=kvarn_hadamard_scatter; tokens=%d; "
        "cache_layout=%s; fallback=reference",
        "model.layers.0.self_attn",
        4,
        "xe2_dpas",
    )


def _pure_decode_metadata(tokens: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        is_prefill=False,
        max_query_len=1,
        num_decodes=tokens,
        num_decode_tokens=tokens,
        num_actual_tokens=tokens,
        has_cached_multiquery=False,
        vq_seqlen=None,
        block_table=torch.zeros(tokens, 1, dtype=torch.int32),
        seq_lens=torch.ones(tokens, dtype=torch.int32),
        max_seq_len=1,
    )


def _bound_qlen1_metadata(tokens: int = 2) -> KVarNMetadata:
    return KVarNMetadata(
        seq_lens=torch.ones(tokens, dtype=torch.int32),
        slot_mapping=torch.arange(tokens, dtype=torch.int64),
        block_table=torch.zeros(tokens, 1, dtype=torch.int32),
        query_start_loc=torch.arange(tokens + 1, dtype=torch.int32),
        num_actual_tokens=tokens,
        max_query_len=1,
        max_seq_len=1,
        is_prefill=False,
        num_decodes=tokens,
        num_decode_tokens=tokens,
        qlen1_dispatch_kind=_KVarNQlen1MetadataKind.EAGER_PURE_DECODE,
    )


def test_fused_qkv_frontend_eligibility_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _fused_frontend_impl()
    metadata = _pure_decode_metadata()
    query = torch.empty(2, 24, 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    slots = torch.tensor([0, 129], dtype=torch.int64)
    layer = SimpleNamespace()

    monkeypatch.setattr(
        kvarn_attn,
        "kvarn_native_store_supported",
        lambda *, op_available, **_: op_available,
    )
    monkeypatch.setattr(kvarn_attn, "kvarn_native_decode_abi_supported", lambda _: True)
    monkeypatch.setattr(kvarn_attn, "kvarn_native_layout_abi_supported", lambda _: True)

    assert impl._native_qkv_scatter_eligible(layer, query, key, value, slots, metadata)
    assert not impl._native_qkv_scatter_eligible(
        layer, query.to(torch.float16), key, value, slots, metadata
    )
    assert not impl._native_qkv_scatter_eligible(
        layer, query, key, value, slots, _pure_decode_metadata(tokens=1)
    )
    assert not impl._native_qkv_scatter_eligible(
        layer, query, key, value, torch.tensor(0, dtype=torch.int64), metadata
    )
    assert not impl._native_qkv_scatter_eligible(
        layer, query, key, value, slots.to(torch.int32), metadata
    )
    impl._block_to_slot_t = None
    assert not impl._native_qkv_scatter_eligible(
        layer, query, key, value, slots, metadata
    )
    impl._block_to_slot_t = torch.zeros(2, dtype=torch.int32)
    impl._tail_V_pool = None
    assert not impl._native_qkv_scatter_eligible(
        layer, query, key, value, slots, metadata
    )
    impl._tail_V_pool = torch.empty_like(impl._tail_K_pool)
    impl._q_rot_fp16_buf = torch.empty(47, 256, dtype=torch.float16)
    assert not impl._native_qkv_scatter_eligible(
        layer, query, key, value, slots, metadata
    )
    impl._q_rot_fp16_buf = torch.empty(48, 256, dtype=torch.float16)
    impl._kvarn_qkv_scatter_op = None
    assert not impl._native_qkv_scatter_eligible(
        layer, query, key, value, slots, metadata
    )


def test_fused_qkv_frontend_dispatches_once_and_consumes_exact_query(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _fused_frontend_impl()
    metadata = _pure_decode_metadata()
    query = torch.empty(2, 24, 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    slots = torch.tensor([0, 129], dtype=torch.int64)
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    layer = SimpleNamespace()
    launch = Mock()

    monkeypatch.setattr(kvarn_attn, "_active_kvarn_metadata", lambda _: metadata)
    monkeypatch.setattr(impl, "_ensure_pool", Mock())
    monkeypatch.setattr(impl, "_native_qkv_scatter_eligible", lambda *_: True)
    monkeypatch.setattr(impl, "_launch_native_qkv_scatter", launch)

    impl.do_qkv_cache_update(layer, query, key, value, cache, slots)

    launch.assert_called_once_with(query, key, value, slots)
    assert impl._consume_fused_qkv_rotation(query, metadata)
    assert not impl._consume_fused_qkv_rotation(query, metadata)


def test_current_stream_frontend_dispatches_bound_native_op() -> None:
    impl = _fused_frontend_impl()
    impl._kvarn_frontend_variant = "qkv_scatter_inline_current_stream"
    impl._kvarn_qkv_scatter_op_name = "kvarn_hadamard_qkv_scatter_current_stream"
    events: list[str] = []
    current_stream_op = Mock(side_effect=lambda *_: events.append("launch"))
    impl._kvarn_qkv_scatter_op = current_stream_op
    query = torch.empty(2, 24, 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    slots = torch.tensor([0, 129], dtype=torch.int64)

    with patch(
        "vllm.v1.attention.backends.kvarn_attn.logger.info_once",
        side_effect=lambda *_args: events.append("marker"),
    ) as active_marker:
        impl._launch_native_qkv_scatter(query, key, value, slots)
        impl._launch_native_qkv_scatter(query, key, value, slots)

    assert events == ["launch", "marker", "launch"]
    assert current_stream_op.call_count == 2
    call = current_stream_op.call_args.args
    for actual, expected in zip(call[:4], (query, key, value, slots), strict=True):
        assert actual.data_ptr() == expected.data_ptr()
        assert actual.shape == expected.shape
    assert call[-2:] == (128, True)
    active_marker.assert_called_once_with(
        "[KVARN_FRONTEND] active=qkv_scatter; "
        "layer=%s; native_op=%s; "
        "qlen=1; cache_layout=%s",
        "model.layers.0.self_attn",
        "kvarn_hadamard_qkv_scatter_current_stream",
        "xe2_dpas",
    )


def test_fused_qkv_frontend_unsupported_step_uses_reference_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _fused_frontend_impl()
    query = torch.empty(2, 24, 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    slots = torch.tensor([0, 129], dtype=torch.int64)
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    layer = SimpleNamespace()
    reference_store = Mock()

    monkeypatch.setattr(kvarn_attn, "_active_kvarn_metadata", lambda _: None)
    monkeypatch.setattr(impl, "_ensure_pool", Mock())
    monkeypatch.setattr(impl, "_native_qkv_scatter_eligible", lambda *_: False)
    monkeypatch.setattr(impl, "do_kv_cache_update", reference_store)

    impl.do_qkv_cache_update(layer, query, key, value, cache, slots)

    reference_store.assert_called_once_with(layer, key, value, cache, slots)
    assert impl._pending_fused_qkv_signature is None


def test_reference_store_invalidates_stale_fused_rotation() -> None:
    impl = _fused_frontend_impl()
    query = torch.empty(2, 24, 256, dtype=torch.bfloat16)
    metadata = _pure_decode_metadata()
    impl._pending_fused_qkv_signature = impl._fused_qkv_signature(query, metadata)
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    key = torch.empty(0, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    slots = torch.empty(0, dtype=torch.int64)

    impl.do_kv_cache_update(SimpleNamespace(), key, value, cache, slots)

    assert impl._pending_fused_qkv_signature is None


def _configure_fused_forward_test(
    impl: KVarNAttentionImpl,
) -> tuple[Mock, Mock]:
    pool_ensure = Mock()
    decode = Mock(
        side_effect=lambda q, cache, metadata, output=None, **_: output.fill_(7)
    )
    impl._ensure_pool = pool_ensure
    impl._decode_path = decode
    return pool_ensure, decode


def _configure_native_qkv_receipt_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    impl: KVarNAttentionImpl,
    metadata,
    events: list[str],
) -> tuple[Mock, Mock, Mock]:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    pool_ensure = Mock(side_effect=lambda *_args, **_kwargs: events.append("ensure"))
    launch = Mock(side_effect=lambda *_args: events.append("launch"))
    decode = Mock(
        side_effect=lambda _q, _cache, _metadata, output=None, **_kwargs: (
            events.append("decode") or output.fill_(7)
        )
    )
    monkeypatch.setattr(kvarn_attn, "_active_kvarn_metadata", lambda _: metadata)
    monkeypatch.setattr(impl, "_ensure_pool", pool_ensure)
    monkeypatch.setattr(impl, "_native_qkv_scatter_eligible", lambda *_: True)
    monkeypatch.setattr(impl, "_launch_native_qkv_scatter", launch)
    monkeypatch.setattr(impl, "_decode_path", decode)
    return pool_ensure, launch, decode


@pytest.mark.parametrize("engine_batch_cap", [1, 4])
def test_trusted_native_decode_plan_caches_static_dispatch_facts(
    monkeypatch: pytest.MonkeyPatch,
    engine_batch_cap: int,
) -> None:
    import vllm.v1.attention.ops.triton_kvarn_decode as decode_module

    impl = _fused_frontend_impl()
    impl._kvarn_native_max_splits = 16
    impl._max_num_seqs = engine_batch_cap
    impl._q_rot_fp16_buf = torch.empty(12 * 24, 256, dtype=torch.float16)
    impl._fused_out_buf = torch.empty_like(impl._q_rot_fp16_buf)
    impl._native_output_fp16_buf = torch.empty_like(impl._fused_out_buf)
    impl._native_decode_scratch = (
        torch.empty(engine_batch_cap, 24 * 16, 256, dtype=torch.float16),
        torch.empty(engine_batch_cap, 24, 16, dtype=torch.float32),
        torch.empty(engine_batch_cap, 24, 16, dtype=torch.float32),
    )
    query = torch.empty(engine_batch_cap, 24, 256, dtype=torch.bfloat16)
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    metadata = _pure_decode_metadata(tokens=engine_batch_cap)

    monkeypatch.setattr(decode_module, "kvarn_native_feature_enabled", lambda _: True)
    monkeypatch.setattr(
        decode_module, "kvarn_native_problem_supported", lambda **_: True
    )
    monkeypatch.setattr(
        decode_module, "kvarn_native_decode_abi_supported", lambda _: True
    )
    monkeypatch.setattr(
        decode_module, "kvarn_native_output_hadamard_supported", lambda _: True
    )
    monkeypatch.setattr(
        decode_module, "kvarn_native_bf16_output_supported", lambda _: True
    )

    plan = decode_module.build_kvarn_trusted_native_decode_plan(
        query, cache, impl.kvarn_config, impl, metadata
    )

    assert plan is not None
    assert plan.max_batch == engine_batch_cap
    assert plan.use_scratch_op
    assert plan.output_hadamard_supported
    assert plan.bf16_output_supported
    impl._native_decode_scratch = tuple(
        tensor[: engine_batch_cap - 1] for tensor in impl._native_decode_scratch
    )
    undersized_plan = decode_module.build_kvarn_trusted_native_decode_plan(
        query, cache, impl.kvarn_config, impl, metadata
    )
    assert undersized_plan is not None
    assert not undersized_plan.use_scratch_op
    assert (
        decode_module.build_kvarn_trusted_native_decode_plan(
            query,
            cache.transpose(0, 1),
            impl.kvarn_config,
            impl,
            metadata,
        )
        is None
    )


def test_bound_native_decode_v2_consumes_cached_bindings_without_reproof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.ops.triton_kvarn_decode as decode_module

    batch_size, num_query_heads, num_kv_heads, head_dim = 2, 2, 1, 4
    max_splits = 16
    query = torch.empty(batch_size, num_query_heads, head_dim, dtype=torch.float16)
    cache = torch.empty(2, num_kv_heads, 8, dtype=torch.uint8)
    q_rot = torch.empty(batch_size * num_query_heads, head_dim, dtype=torch.float16)
    fused_out = torch.empty_like(q_rot)
    native_output = torch.empty_like(q_rot)
    scratch = (
        torch.empty(
            batch_size,
            num_query_heads * max_splits,
            head_dim,
            dtype=torch.float16,
        ),
        torch.empty(batch_size, num_query_heads, max_splits),
        torch.empty(batch_size, num_query_heads, max_splits),
    )
    block_to_slot = torch.zeros(2, dtype=torch.int32)
    tail_key = torch.empty(2, 128, num_kv_heads, head_dim, dtype=torch.float16)
    tail_value = torch.empty_like(tail_key)
    plan = decode_module.KVarNBoundNativeDecodePlanV2(
        max_batch=batch_size,
        dpas_layout=True,
        q_rot_fp16=q_rot,
        fused_out=fused_out,
        native_output_fp16=native_output,
        native_scratch=scratch,
        block_to_slot=block_to_slot,
        tail_key=tail_key,
        tail_value=tail_value,
        use_native_hadamard=True,
        use_scratch_op=True,
        output_hadamard_supported=True,
        direct_bf16_output=False,
        max_splits=max_splits,
        split_policy="fixed",
        kernel_variant=18,
    )
    impl = SimpleNamespace(
        _H_fp16=torch.empty(head_dim, head_dim, dtype=torch.float16),
    )
    metadata = SimpleNamespace(
        max_seq_len=4096,
        block_table=torch.zeros(batch_size, 1, dtype=torch.int32),
        seq_lens=torch.full((batch_size,), 4096, dtype=torch.int32),
    )
    cfg = SimpleNamespace(group=128)

    def repeated_proof_used(*_args, **_kwargs):
        raise AssertionError("bound v2 repeated a bind-time qualification")

    for name in (
        "_kvarn_dpas_layout_for_problem",
        "kvarn_native_feature_enabled",
        "kvarn_native_problem_supported",
        "kvarn_native_decode_abi_supported",
        "kvarn_native_output_hadamard_supported",
        "kvarn_native_bf16_output_supported",
    ):
        monkeypatch.setattr(decode_module, name, repeated_proof_used)
    monkeypatch.setattr(decode_module, "kvarn_native_split_count", Mock(return_value=4))
    monkeypatch.setattr(torch.profiler, "record_function", repeated_proof_used)
    native_launch = Mock(side_effect=lambda *args: args[10].fill_(3))
    monkeypatch.setattr(
        torch.ops._vllm_fa2_C,
        "kvarn_decode_with_scratch",
        native_launch,
        raising=False,
    )
    result = decode_module.kvarn_decode_attention(
        query,
        cache,
        impl._H_fp16,
        0.125,
        cfg,
        impl,
        metadata,
        output=torch.empty_like(query),
        query_rotation_precomputed=True,
        bound_native_plan_v2=plan,
    )

    assert result.eq(3).all()
    native_launch.assert_called_once()
    assert native_launch.call_args.args[4] is block_to_slot
    assert native_launch.call_args.args[5] is tail_key
    assert native_launch.call_args.args[6] is tail_value


def test_bound_qlen1_inline_v2_binds_once_with_exact_fp16_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _fused_frontend_impl()
    impl._kvarn_qlen1_inline_plan = "bound_native_v2"
    impl.use_bound_qlen1_inline_plan_v2 = True
    metadata = _bound_qlen1_metadata()
    query = torch.empty(2, 24, 256, dtype=torch.float16)
    key = torch.empty(2, 4, 256, dtype=torch.float16)
    value = torch.empty_like(key)
    output = torch.empty(2, 24 * 256, dtype=torch.float16)
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    slots = torch.tensor([0, 129], dtype=torch.int64)
    native_decode = SimpleNamespace(
        max_batch=12,
        use_native_hadamard=True,
        use_scratch_op=True,
        output_hadamard_supported=True,
        bf16_output_supported=True,
    )
    pool_ensure = Mock()
    eligibility = Mock(return_value=True)
    plan_builder = Mock(return_value=native_decode)
    launch = Mock()
    decode = Mock(
        side_effect=lambda _q, _cache, _metadata, output=None, **_: torch.full_like(
            output, 7
        )
    )

    monkeypatch.setattr(impl, "_ensure_pool", pool_ensure)
    monkeypatch.setattr(impl, "_native_qkv_scatter_eligible", eligibility)
    monkeypatch.setattr(
        kvarn_attn, "build_kvarn_trusted_native_decode_plan", plan_builder
    )
    monkeypatch.setattr(impl, "_launch_native_qkv_scatter", launch)
    monkeypatch.setattr(impl, "_decode_path", decode)
    split_count = Mock(return_value=4)
    monkeypatch.setattr(kvarn_attn, "kvarn_native_split_count", split_count)

    with patch("vllm.v1.attention.backends.kvarn_attn.logger.info") as marker:
        for _ in range(2):
            assert impl.forward_bound_qlen1_inline_v2(
                SimpleNamespace(),
                query,
                key,
                value,
                cache,
                slots,
                metadata,
                output,
            )

    pool_ensure.assert_called_once_with(query.device, num_blocks_hint=cache.shape[0])
    assert eligibility.call_count == 1
    assert plan_builder.call_count == 1
    assert launch.call_count == 2
    assert decode.call_count == 2
    assert decode.call_args.kwargs["query_rotation_precomputed"] is True
    assert "trusted_native_plan" not in decode.call_args.kwargs
    bound_native_decode = decode.call_args.kwargs["bound_native_plan_v2"]
    assert bound_native_decode.max_batch == 12
    assert bound_native_decode.dpas_layout is True
    assert bound_native_decode.q_rot_fp16 is impl._q_rot_fp16_buf
    assert bound_native_decode.fused_out is impl._fused_out_buf
    assert bound_native_decode.native_output_fp16 is impl._native_output_fp16_buf
    assert bound_native_decode.native_scratch is impl._native_decode_scratch
    assert bound_native_decode.block_to_slot is impl._block_to_slot_t
    assert bound_native_decode.tail_key is impl._tail_K_pool
    assert bound_native_decode.tail_value is impl._tail_V_pool
    assert bound_native_decode.use_native_hadamard is True
    assert bound_native_decode.use_scratch_op is True
    assert bound_native_decode.output_hadamard_supported is True
    assert bound_native_decode.direct_bf16_output is False
    assert bound_native_decode.max_splits == 16
    assert bound_native_decode.split_policy == "fixed"
    assert bound_native_decode.kernel_variant == 18
    assert output.dtype == torch.float16
    assert output.eq(7).all()
    plan = impl._bound_qlen1_inline_v2_plan
    assert plan is not None
    assert plan.cache_owner is cache
    assert plan.cache_view is cache
    assert plan.cache_layout == "xe2_dpas"
    assert plan.record_bytes == 35_072
    assert plan.activation_dtype == torch.float16
    assert plan.split_policy == "fixed"
    assert plan.max_splits == 16
    assert plan.kernel_variant == 18
    assert plan.output_dtype == torch.float16
    assert plan.qualified_batch_limit == 12
    split_count.assert_called_once_with(
        1,
        16,
        batch_size=2,
        split_policy="fixed",
        kernel_variant=18,
    )
    marker.assert_called_once_with(
        "[KVARN_BOUND_QLEN1_INLINE] active=bound_native_v2; variant=H; "
        "Using the native Xe2 KVarN qlen=1 decoder (batch limit %d; "
        "native H256 transforms=%s; fused output H256=%s; "
        "direct bf16 output=%s; cache layout=%s; splits=%d); "
        "hot_guards=process_generation+binding_epoch+cache_identity+"
        "metadata_kind+batch_limit; fp16_contract=exact; "
        "fallback=reference; layer=%s",
        12,
        True,
        True,
        False,
        "xe2_dpas",
        4,
        "model.layers.0.self_attn",
    )


def test_beta_bound_qlen1_inline_v2_rejects_no_scratch_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _fused_frontend_impl()
    impl._kvarn_xpu_beta_profile = True
    impl._kvarn_qlen1_inline_plan = "bound_native_v2"
    impl.use_bound_qlen1_inline_plan_v2 = True
    impl._max_model_len = 65_536
    metadata = _bound_qlen1_metadata()
    query = torch.empty(2, 24, 256, dtype=torch.float16)
    key = torch.empty(2, 4, 256, dtype=torch.float16)
    value = torch.empty_like(key)
    output = torch.empty(2, 24 * 256, dtype=torch.float16)
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    slots = torch.tensor([0, 129], dtype=torch.int64)
    native_decode = SimpleNamespace(
        max_batch=12,
        use_native_hadamard=True,
        use_scratch_op=False,
        output_hadamard_supported=True,
        bf16_output_supported=True,
    )

    monkeypatch.setattr(impl, "_ensure_pool", Mock())
    monkeypatch.setattr(impl, "_native_qkv_scatter_eligible", Mock(return_value=True))
    monkeypatch.setattr(
        kvarn_attn,
        "build_kvarn_trusted_native_decode_plan",
        Mock(return_value=native_decode),
    )

    with pytest.raises(
        RuntimeError,
        match="beta Variant H requires native scratch decode",
    ):
        impl.forward_bound_qlen1_inline_v2(
            SimpleNamespace(),
            query,
            key,
            value,
            cache,
            slots,
            metadata,
            output,
        )

    assert impl._bound_qlen1_inline_v2_plan is None


@pytest.mark.parametrize(
    "failure",
    [
        "process_generation",
        "binding_epoch",
        "cache_owner",
        "metadata_kind",
        "metadata_type",
        "batch_limit",
        "batch_alignment",
    ],
)
def test_bound_qlen1_inline_v2_hot_guards_fail_closed(
    failure: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _fused_frontend_impl()
    impl._kvarn_qlen1_inline_plan = "bound_native_v2"
    impl.use_bound_qlen1_inline_plan_v2 = True
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    metadata = _bound_qlen1_metadata()
    slots = torch.tensor([0, 129], dtype=torch.int64)
    plan = kvarn_attn._KVarNBoundQlen1InlinePlanV2(
        process_generation=KVarNAttentionImpl._process_generation,
        binding_epoch=impl._bound_qlen1_inline_v2_binding_epoch,
        cache_owner=cache,
        cache_view=cache,
        device=torch.device("cpu"),
        activation_dtype=torch.float16,
        cache_layout="xe2_dpas",
        record_bytes=35_072,
        split_policy="fixed",
        max_splits=16,
        kernel_variant=18,
        output_dtype=torch.float16,
        qualified_batch_limit=12,
        native_decode=SimpleNamespace(max_batch=12),
    )

    if failure == "process_generation":
        plan = type(plan)(**(vars(plan) | {"process_generation": -1}))
    elif failure == "binding_epoch":
        plan = type(plan)(**(vars(plan) | {"binding_epoch": -1}))
    elif failure == "cache_owner":
        cache = torch.empty_like(cache)
    elif failure == "metadata_kind":
        metadata.qlen1_dispatch_kind = _KVarNQlen1MetadataKind.REFERENCE
    elif failure == "metadata_type":
        metadata = SimpleNamespace(num_actual_tokens=2)
    elif failure == "batch_limit":
        plan = type(plan)(**(vars(plan) | {"qualified_batch_limit": 1}))
    else:
        slots = slots[:1]

    assert not impl._bound_qlen1_inline_v2_hot_path_is_current(
        plan, cache, slots, metadata
    )


def test_bound_qlen1_inline_v2_epoch_covers_every_binding_replacement() -> None:
    KVarNAttentionImpl.reset_process_state()
    impl = _fused_frontend_impl()
    impl.use_bound_qlen1_inline_plan_v2 = True
    KVarNAttentionImpl._all_impls.append(impl)
    try:

        def assert_invalidated(action) -> None:
            previous = impl._bound_qlen1_inline_v2_binding_epoch
            impl._bound_qlen1_inline_v2_plan = object()
            action()
            assert impl._bound_qlen1_inline_v2_binding_epoch > previous
            assert impl._bound_qlen1_inline_v2_plan is None

        mirror_key = impl._pool_ready_key
        shared_key = (torch.device("cpu"), impl.head_size, impl.num_kv_heads)
        assert_invalidated(
            lambda: KVarNAttentionImpl._mark_pool_mirror_changed(mirror_key)
        )
        assert_invalidated(
            lambda: KVarNAttentionImpl._mark_pool_shared_changed(shared_key)
        )
        assert_invalidated(impl._publish_bound_qlen1_inline_v2_pool_binding)
        assert_invalidated(
            lambda: impl._bind_bound_qlen1_inline_v2_group(("replacement",))
        )
        assert_invalidated(
            lambda: impl._bind_bound_qlen1_inline_v2_cache(torch.empty(1))
        )
        assert_invalidated(KVarNAttentionImpl.reset_process_state)
    finally:
        KVarNAttentionImpl.reset_process_state()


def test_bound_qlen1_inline_v2_miss_uses_reference_wrapper_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.attention.attention as attention_module
    import vllm.model_executor.layers.attention.kv_transfer_utils as kv_transfer_utils

    impl = _fused_frontend_impl()
    impl.use_inline_qkv_cache_update = True
    impl.use_bound_qlen1_inline_plan_v2 = True
    bound = Mock(return_value=False)
    store = Mock()
    reference = Mock()
    impl.forward_bound_qlen1_inline_v2 = bound
    impl.do_qkv_cache_update = store
    impl.forward = reference
    query = torch.empty(2, 24, 256, dtype=torch.float16)
    key = torch.empty(2, 4, 256, dtype=torch.float16)
    value = torch.empty_like(key)
    output = torch.empty_like(query)
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    slots = torch.tensor([0, 129], dtype=torch.int64)
    metadata = _bound_qlen1_metadata()
    layer = SimpleNamespace(impl=impl, _inline_qkv_attention_active_logged=True)
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        Mock(return_value=(metadata, layer, cache, slots)),
    )
    monkeypatch.setattr(kv_transfer_utils, "has_kv_transfer_group", lambda: False)

    attention_module.unified_qkv_attention_with_output(
        query,
        key,
        value,
        output,
        "model.layers.0.self_attn",
    )

    bound.assert_called_once_with(
        layer, query, key, value, cache, slots, metadata, output
    )
    store.assert_called_once_with(layer, query, key, value, cache, slots)
    reference.assert_called_once_with(
        layer,
        query,
        key,
        value,
        cache,
        metadata,
        output=output,
        output_scale=None,
        output_block_scale=None,
    )


def test_captured_native_qkv_fallback_does_not_publish_pool_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from torch._subclasses.fake_tensor import FakeTensorMode

    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    impl = _fused_frontend_impl()
    metadata = _pure_decode_metadata()
    with FakeTensorMode():
        query = torch.empty(2, 24, 256, dtype=torch.bfloat16, device="xpu")
        key = torch.empty(2, 4, 256, dtype=torch.bfloat16, device="xpu")
        value = torch.empty_like(key)
        cache = torch.empty(2, 4, 35_072, dtype=torch.uint8, device="xpu")
        slots = torch.tensor([0, 129], dtype=torch.int64, device="xpu")
    fallback = Mock()
    launch = Mock()
    native_store = Mock(wraps=kvarn_attn.kvarn_native_store_supported)
    pool_ensure = Mock()
    monkeypatch.setenv("KVARN_NATIVE_XPU", "1")
    monkeypatch.setattr(kvarn_attn, "_active_kvarn_metadata", lambda _: metadata)
    monkeypatch.setattr(kvarn_attn, "kvarn_native_decode_abi_supported", lambda _: True)
    monkeypatch.setattr(kvarn_attn, "kvarn_native_layout_abi_supported", lambda _: True)
    monkeypatch.setattr(kvarn_attn, "kvarn_native_store_supported", native_store)
    monkeypatch.setattr(torch.xpu, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(impl, "_ensure_pool", pool_ensure)
    monkeypatch.setattr(impl, "_launch_native_qkv_scatter", launch)
    monkeypatch.setattr(impl, "do_kv_cache_update", fallback)
    impl._pending_fused_qkv_signature = ("stale",)
    layer = SimpleNamespace()

    impl.do_qkv_cache_update(layer, query, key, value, cache, slots)

    launch.assert_not_called()
    pool_ensure.assert_called_once_with(query.device, num_blocks_hint=cache.shape[0])
    assert native_store.call_args.kwargs["is_capturing"] is True
    fallback.assert_called_once_with(layer, key, value, cache, slots)
    assert impl._pending_fused_qkv_signature is None


@pytest.mark.parametrize("receipt", ["missing", "query_mismatch"])
def test_fused_qkv_pool_proof_fails_closed_without_matching_signature(
    receipt: str,
) -> None:
    impl = _fused_frontend_impl()
    impl._kvarn_forward_pool_ensure = "fused_qkv_proof"
    metadata = _pure_decode_metadata()
    query = torch.empty(2, 24, 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    if receipt == "query_mismatch":
        other_query = torch.empty_like(query)
        impl._pending_fused_qkv_signature = impl._fused_qkv_signature(
            other_query, metadata
        )
    pool_ensure, decode = _configure_fused_forward_test(impl)

    with patch("vllm.v1.attention.backends.kvarn_attn.logger.info") as marker:
        impl.forward(SimpleNamespace(), query, key, value, cache, metadata)

    pool_ensure.assert_called_once_with(query.device, num_blocks_hint=cache.shape[0])
    assert "query_rotation_precomputed" not in decode.call_args.kwargs
    assert impl._pending_fused_qkv_signature is None
    marker.assert_not_called()


def test_fused_qkv_pool_proof_fails_closed_when_pool_binding_changes() -> None:
    impl = _fused_frontend_impl()
    impl._kvarn_forward_pool_ensure = "fused_qkv_proof"
    metadata = _pure_decode_metadata()
    query = torch.empty(2, 24, 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    impl._pending_fused_qkv_signature = impl._fused_qkv_signature(query, metadata)
    impl._group_key = (256, 4, 4096)
    pool_ensure, decode = _configure_fused_forward_test(impl)

    impl.forward(SimpleNamespace(), query, key, value, cache, metadata)

    pool_ensure.assert_called_once_with(query.device, num_blocks_hint=cache.shape[0])
    assert "query_rotation_precomputed" not in decode.call_args.kwargs


def test_forward_pool_ensure_default_keeps_second_check() -> None:
    impl = _fused_frontend_impl()
    assert impl._kvarn_forward_pool_ensure == "always"
    metadata = _pure_decode_metadata()
    query = torch.empty(2, 24, 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    impl._pending_fused_qkv_signature = impl._fused_qkv_signature(query, metadata)
    pool_ensure, decode = _configure_fused_forward_test(impl)

    with patch("vllm.v1.attention.backends.kvarn_attn.logger.info") as marker:
        impl.forward(SimpleNamespace(), query, key, value, cache, metadata)

    pool_ensure.assert_called_once_with(query.device, num_blocks_hint=cache.shape[0])
    assert decode.call_args.kwargs["query_rotation_precomputed"] is True
    marker.assert_not_called()


def test_fused_qkv_pool_proof_keeps_empty_cache_profile_ensure() -> None:
    impl = _fused_frontend_impl()
    impl._kvarn_forward_pool_ensure = "fused_qkv_proof"
    metadata = _pure_decode_metadata()
    query = torch.empty(2, 24, 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)
    impl._pending_fused_qkv_signature = impl._fused_qkv_signature(query, metadata)
    pool_ensure, decode = _configure_fused_forward_test(impl)

    output = impl.forward(
        SimpleNamespace(),
        query,
        key,
        value,
        torch.empty(0, dtype=torch.uint8),
        metadata,
    )

    pool_ensure.assert_called_once_with(query.device, num_blocks_hint=0)
    decode.assert_not_called()
    assert output.eq(0).all()


def test_unified_qkv_cache_update_passes_query_without_changing_cache_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.attention.attention as attention_module

    update = Mock()
    layer = SimpleNamespace(impl=SimpleNamespace(do_qkv_cache_update=update))
    cache = torch.empty(2, 4, 35_072, dtype=torch.uint8)
    slots = torch.tensor([0, 129], dtype=torch.int64)
    monkeypatch.setattr(
        attention_module,
        "get_attention_context",
        lambda _: (object(), layer, cache, slots),
    )
    query = torch.empty(2, 24, 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)

    dependency = attention_module.unified_qkv_cache_update(
        query, key, value, "model.layers.0.self_attn"
    )

    update.assert_called_once_with(layer, query, key, value, cache, slots)
    assert dependency.shape == (0,)
    assert dependency.dtype == key.dtype


def test_unified_qkv_cache_update_is_cudagraph_unsafe_and_has_fake_dispatch():
    import vllm.model_executor.layers.attention.attention  # noqa: F401

    op = torch.ops.vllm.unified_qkv_cache_update.default
    assert torch.Tag.cudagraph_unsafe in op.tags
    query = torch.empty(2, 24, 256, dtype=torch.bfloat16, device="meta")
    key = torch.empty(2, 4, 256, dtype=torch.bfloat16, device="meta")
    value = torch.empty_like(key)

    dependency = op(query, key, value, "model.layers.0.self_attn")

    assert dependency.device.type == "meta"
    assert dependency.shape == (0,)
    assert dependency.dtype == key.dtype


def test_inline_qkv_attention_uses_one_context_and_update_before_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.attention.attention as attention_module
    import vllm.model_executor.layers.attention.kv_transfer_utils as kv_transfer_utils

    class NoDummyTensor:
        def new_empty(self, *_args, **_kwargs):
            raise AssertionError("inline route allocated a dummy dependency")

    calls = []
    impl = SimpleNamespace(
        use_inline_qkv_cache_update=True,
        do_qkv_cache_update=Mock(side_effect=lambda *_args: calls.append("update")),
        forward=Mock(side_effect=lambda *_args, **_kwargs: calls.append("forward")),
    )
    layer = SimpleNamespace(
        impl=impl,
        _inline_qkv_attention_active_logged=False,
    )
    cache = object()
    metadata = object()
    slots = object()
    context = Mock(return_value=(metadata, layer, cache, slots))
    monkeypatch.setattr(attention_module, "get_attention_context", context)
    monkeypatch.setattr(kv_transfer_utils, "has_kv_transfer_group", lambda: False)
    query = NoDummyTensor()
    key = NoDummyTensor()
    value = NoDummyTensor()
    output = NoDummyTensor()

    result = attention_module.unified_qkv_attention_with_output(
        query,
        key,
        value,
        output,
        "model.layers.0.self_attn",
    )

    assert result is None
    context.assert_called_once_with("model.layers.0.self_attn")
    assert calls == ["update", "forward"]
    impl.do_qkv_cache_update.assert_called_once_with(
        layer, query, key, value, cache, slots
    )
    assert impl.forward.call_args.args == (
        layer,
        query,
        key,
        value,
        cache,
        metadata,
    )
    assert impl.forward.call_args.kwargs["output"] is output


def test_inline_qkv_attention_promotes_trusted_plan_without_legacy_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.attention.attention as attention_module
    import vllm.model_executor.layers.attention.kv_transfer_utils as kv_transfer_utils

    trusted = Mock(return_value=True)
    impl = SimpleNamespace(
        use_trusted_qlen1_inline_plan=True,
        forward_trusted_qlen1_inline=trusted,
        do_qkv_cache_update=Mock(
            side_effect=AssertionError("legacy update path executed")
        ),
        forward=Mock(side_effect=AssertionError("legacy receipt path executed")),
    )
    layer = SimpleNamespace(
        impl=impl,
        _inline_qkv_attention_active_logged=False,
    )
    cache = object()
    metadata = object()
    slots = object()
    context = Mock(return_value=(metadata, layer, cache, slots))
    monkeypatch.setattr(attention_module, "get_attention_context", context)
    monkeypatch.setattr(kv_transfer_utils, "has_kv_transfer_group", lambda: False)
    tensors = [object() for _ in range(4)]

    assert (
        attention_module.unified_qkv_attention_with_output(
            *tensors,
            "model.layers.0.self_attn",
        )
        is None
    )

    trusted.assert_called_once_with(
        layer,
        *tensors[:3],
        cache,
        slots,
        metadata,
        tensors[3],
    )
    impl.do_qkv_cache_update.assert_not_called()
    impl.forward.assert_not_called()


def test_inline_qkv_attention_custom_op_contract() -> None:
    import vllm.model_executor.layers.attention.attention  # noqa: F401

    op = torch.ops.vllm.unified_qkv_attention_with_output.default
    assert torch.Tag.cudagraph_unsafe in op.tags
    arguments = {argument.name: argument for argument in op._schema.arguments}
    assert arguments["output"].alias_info.is_write
    assert arguments["output_block_scale"].alias_info.is_write

    query = torch.empty(2, 24, 256, dtype=torch.bfloat16, device="meta")
    key = torch.empty(2, 4, 256, dtype=torch.bfloat16, device="meta")
    value = torch.empty_like(key)
    output = torch.empty_like(query)

    assert op(query, key, value, output, "model.layers.0.self_attn") is None


@pytest.mark.parametrize("opaque_attention", [False, True])
def test_inline_qkv_attention_accepts_xpu_direct_and_opaque_paths(
    monkeypatch: pytest.MonkeyPatch,
    opaque_attention: bool,
) -> None:
    import vllm.model_executor.layers.attention.attention as attention_module

    monkeypatch.setattr(attention_module.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(
        attention_module.current_platform,
        "opaque_attention_op",
        lambda: opaque_attention,
    )

    attention_module._validate_inline_qkv_cache_update(True, "KVARN")
    assert (not attention_module.current_platform.opaque_attention_op()) is (
        not opaque_attention
    )


def test_inline_qkv_attention_requires_xpu_at_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.attention.attention as attention_module

    monkeypatch.setattr(attention_module.current_platform, "is_xpu", lambda: False)

    with pytest.raises(RuntimeError, match="XPU KVarN"):
        attention_module._validate_inline_qkv_cache_update(True, "KVARN")

    attention_module._validate_inline_qkv_cache_update(False, "FLASH_ATTN")


def test_inline_qkv_attention_rejects_non_kvarn_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.attention.attention as attention_module

    monkeypatch.setattr(attention_module.current_platform, "is_xpu", lambda: True)

    with pytest.raises(RuntimeError, match="XPU KVarN"):
        attention_module._validate_inline_qkv_cache_update(True, "FLASH_ATTN")


def test_reference_frontend_skips_fused_signature_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _fused_frontend_impl()
    impl.use_fused_qkv_cache_update = False
    impl._pending_fused_qkv_signature = None
    monkeypatch.setattr(impl, "_record_cache_view", lambda cache: cache)
    monkeypatch.setattr(impl, "_ensure_pool", Mock())
    consume = Mock(side_effect=AssertionError("reference frontend built a signature"))
    monkeypatch.setattr(impl, "_consume_fused_qkv_rotation", consume)
    query = torch.empty(2, 24, 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4, 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)

    output = impl.forward(SimpleNamespace(), query, key, value, torch.empty(0), None)

    consume.assert_not_called()
    assert output.shape == (2, 24 * 256)


def test_attention_routes_selected_frontend_through_qkv_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.attention.attention as attention_module

    attention = object.__new__(attention_module.Attention)
    torch.nn.Module.__init__(attention)
    attention.query_quant = None
    attention.num_heads = 24
    attention.num_kv_heads = 4
    attention.head_size = 256
    attention.head_size_v = 256
    attention.impl = SimpleNamespace()
    attention.use_fused_qkv_cache_update = True
    attention.use_inline_qkv_cache_update = False
    attention.use_direct_call = True
    attention.attn_backend = SimpleNamespace(forward_includes_kv_cache_update=False)
    attention.kv_sharing_target_layer_name = None
    attention.layer_name = "model.layers.0.self_attn"
    dependency = torch.empty(0)
    fused_update = Mock(return_value=dependency)
    reference_update = Mock(side_effect=AssertionError("reference update selected"))
    attention_forward = Mock()
    inline_marker = Mock()
    monkeypatch.setattr(attention_module.logger, "info", inline_marker)
    monkeypatch.setattr(attention_module, "unified_qkv_cache_update", fused_update)
    monkeypatch.setattr(attention_module, "unified_kv_cache_update", reference_update)
    monkeypatch.setattr(
        attention_module,
        "unified_qkv_attention_with_output",
        Mock(side_effect=AssertionError("inline route selected")),
    )
    monkeypatch.setattr(
        attention_module, "unified_attention_with_output", attention_forward
    )
    query = torch.empty(2, 24 * 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4 * 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)

    result = attention.forward(query, key, value)

    assert result.shape == query.shape
    fused_update.assert_called_once()
    q_arg, k_arg, v_arg, layer_name = fused_update.call_args.args
    assert q_arg.shape == (2, 24, 256)
    assert k_arg.shape == v_arg.shape == (2, 4, 256)
    assert layer_name == attention.layer_name
    reference_update.assert_not_called()
    assert attention_forward.call_args.kwargs["kv_cache_dummy_dep"] is dependency
    inline_marker.assert_not_called()


def test_attention_routes_inline_frontend_without_dummy_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.attention.attention as attention_module

    attention = object.__new__(attention_module.Attention)
    torch.nn.Module.__init__(attention)
    attention.query_quant = None
    attention.num_heads = 24
    attention.num_kv_heads = 4
    attention.head_size = 256
    attention.head_size_v = 256
    attention.impl = SimpleNamespace()
    attention.use_fused_qkv_cache_update = True
    attention.use_inline_qkv_cache_update = True
    attention.use_direct_call = True
    attention.attn_backend = SimpleNamespace(forward_includes_kv_cache_update=False)
    attention.kv_sharing_target_layer_name = None
    attention.layer_name = "model.layers.0.self_attn"
    inline = Mock()
    monkeypatch.setattr(attention_module, "unified_qkv_attention_with_output", inline)
    monkeypatch.setattr(
        attention_module,
        "unified_qkv_cache_update",
        Mock(side_effect=AssertionError("dummy update route selected")),
    )
    monkeypatch.setattr(
        attention_module,
        "unified_kv_cache_update",
        Mock(side_effect=AssertionError("reference update selected")),
    )
    monkeypatch.setattr(
        attention_module,
        "unified_attention_with_output",
        Mock(side_effect=AssertionError("separate attention route selected")),
    )
    query = torch.empty(2, 24 * 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4 * 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)

    result = attention.forward(query, key, value)

    assert result.shape == query.shape
    q_arg, k_arg, v_arg, output_arg, layer_name = inline.call_args.args
    assert q_arg.shape == (2, 24, 256)
    assert k_arg.shape == v_arg.shape == (2, 4, 256)
    assert output_arg.shape == (2, 24, 256)
    assert layer_name == attention.layer_name


def test_attention_routes_inline_frontend_through_one_opaque_xpu_op(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.attention.attention as attention_module
    import vllm.model_executor.layers.attention.kv_transfer_utils as kv_transfer_utils

    monkeypatch.setattr(attention_module.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(
        attention_module.current_platform, "opaque_attention_op", lambda: True
    )
    attention_module._validate_inline_qkv_cache_update(True, "KVARN")

    calls = []

    def update(*_args):
        calls.append("update")

    def attention_forward(*_args, output=None, **_kwargs):
        assert calls[-1] == "update"
        calls.append("forward")
        output.fill_(7)
        return output

    impl = SimpleNamespace(
        do_qkv_cache_update=Mock(side_effect=update),
        forward=Mock(side_effect=attention_forward),
    )
    attention = object.__new__(attention_module.Attention)
    torch.nn.Module.__init__(attention)
    attention.query_quant = None
    attention.num_heads = 24
    attention.num_kv_heads = 4
    attention.head_size = 256
    attention.head_size_v = 256
    attention.impl = impl
    attention.use_fused_qkv_cache_update = True
    attention.use_inline_qkv_cache_update = True
    attention._inline_qkv_attention_active_logged = False
    attention.use_direct_call = (
        not attention_module.current_platform.opaque_attention_op()
    )
    attention.attn_backend = SimpleNamespace(forward_includes_kv_cache_update=False)
    attention.kv_sharing_target_layer_name = None
    attention.layer_name = "model.layers.0.self_attn"
    context = Mock(return_value=(object(), attention, object(), object()))
    monkeypatch.setattr(attention_module, "get_attention_context", context)
    monkeypatch.setattr(kv_transfer_utils, "has_kv_transfer_group", lambda: False)

    def combined_dispatch(*args, **kwargs):
        calls.append("combined")
        return attention_module.unified_qkv_attention_with_output(*args, **kwargs)

    combined = Mock(side_effect=combined_dispatch)
    old_qkv_update = Mock(side_effect=AssertionError("old QKV update op selected"))
    old_kv_update = Mock(side_effect=AssertionError("old KV update op selected"))
    old_attention = Mock(side_effect=AssertionError("old attention op selected"))
    monkeypatch.setattr(
        attention_module.torch.ops.vllm,
        "unified_qkv_attention_with_output",
        combined,
    )
    monkeypatch.setattr(
        attention_module.torch.ops.vllm,
        "unified_qkv_cache_update",
        old_qkv_update,
    )
    monkeypatch.setattr(
        attention_module.torch.ops.vllm,
        "unified_kv_cache_update",
        old_kv_update,
    )
    monkeypatch.setattr(
        attention_module.torch.ops.vllm,
        "unified_attention_with_output",
        old_attention,
    )
    query = torch.empty(2, 24 * 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4 * 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)

    result = attention.forward(query, key, value)

    assert attention.use_direct_call is False
    assert calls == ["combined", "update", "forward"]
    combined.assert_called_once()
    old_qkv_update.assert_not_called()
    old_kv_update.assert_not_called()
    old_attention.assert_not_called()
    context.assert_called_once_with(attention.layer_name)
    assert result.eq(7).all()


def test_inline_frontend_keeps_shared_kv_on_ordinary_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.layers.attention.attention as attention_module

    attention = object.__new__(attention_module.Attention)
    torch.nn.Module.__init__(attention)
    attention.query_quant = None
    attention.num_heads = 24
    attention.num_kv_heads = 4
    attention.head_size = 256
    attention.head_size_v = 256
    attention.impl = SimpleNamespace()
    attention.use_fused_qkv_cache_update = True
    attention.use_inline_qkv_cache_update = True
    attention.use_direct_call = True
    attention.attn_backend = SimpleNamespace(forward_includes_kv_cache_update=False)
    attention.kv_sharing_target_layer_name = "model.layers.0.self_attn"
    attention.layer_name = "model.layers.1.self_attn"
    ordinary = Mock()
    inline = Mock(side_effect=AssertionError("inline cache update selected"))
    monkeypatch.setattr(attention_module, "unified_qkv_attention_with_output", inline)
    monkeypatch.setattr(attention_module, "unified_attention_with_output", ordinary)
    query = torch.empty(2, 24 * 256, dtype=torch.bfloat16)
    key = torch.empty(2, 4 * 256, dtype=torch.bfloat16)
    value = torch.empty_like(key)

    attention.forward(query, key, value)

    inline.assert_not_called()
    ordinary.assert_called_once()
    assert ordinary.call_args.kwargs["kv_cache_dummy_dep"] is None


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
    assert not _can_elide_fa_cu_seqlens(for_cudagraph_capture=True, **kwargs)
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


def test_cache_layout_is_frozen_at_attention_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KVARN_NATIVE_XPU_CACHE_LAYOUT", raising=False)
    monkeypatch.delenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", raising=False)
    monkeypatch.delenv("KVARN_NATIVE_XPU_SPLIT_POLICY", raising=False)
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "16")
    monkeypatch.delenv("KVARN_NATIVE_XPU_KERNEL_VARIANT", raising=False)
    monkeypatch.delenv("KVARN_NATIVE_XPU_FRONTEND", raising=False)
    KVarNAttentionImpl.reset_process_state()
    try:
        with patch(
            "vllm.v1.attention.backends.kvarn_attn.get_flash_attn_version",
            return_value=2,
        ):
            impl = KVarNAttentionImpl(
                num_heads=24,
                head_size=256,
                scale=1.0 / 16.0,
                num_kv_heads=4,
                kv_cache_dtype="kvarn_k4v4_g128",
            )
        assert impl._kvarn_cache_layout == "natural"
        assert not impl._kvarn_dpas_layout
        assert impl._kvarn_native_split_policy == "fixed"
        assert impl._kvarn_native_max_splits == 16
        assert impl._kvarn_native_kernel_variant_name == "baseline"
        assert impl._kvarn_native_kernel_variant == 0
        assert impl._kvarn_frontend_variant == "reference"
        assert not impl.use_fused_qkv_cache_update

        monkeypatch.setenv("KVARN_NATIVE_XPU_CACHE_LAYOUT", "xe2_dpas")
        monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "32")
        monkeypatch.setenv("KVARN_NATIVE_XPU_KERNEL_VARIANT", "unknown")
        monkeypatch.setenv("KVARN_NATIVE_XPU_FRONTEND", "qkv_scatter")
        assert impl._kvarn_cache_layout == "natural"
        assert not impl._kvarn_dpas_layout
        assert impl._kvarn_native_split_policy == "fixed"
        assert impl._kvarn_native_max_splits == 16
        assert impl._kvarn_native_kernel_variant_name == "baseline"
        assert impl._kvarn_native_kernel_variant == 0
        assert impl._kvarn_frontend_variant == "reference"
        assert not impl.use_fused_qkv_cache_update
    finally:
        KVarNAttentionImpl.reset_process_state()


def test_compact_kvarn_dtype_selects_xpu_beta_profile_without_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    for name in (
        "KVARN_NATIVE_XPU_CACHE_LAYOUT",
        "KVARN_NATIVE_XPU_DPAS_LAYOUT",
        "KVARN_NATIVE_XPU_FRONTEND",
        "KVARN_NATIVE_XPU_PREFILL_STORE",
        "KVARN_NATIVE_XPU_SPLIT_POLICY",
        "KVARN_NATIVE_XPU_SPLITS",
        "KVARN_NATIVE_XPU_KERNEL_VARIANT",
        "KVARN_FLUSH_INDEX_MATERIALIZATION",
        "KVARN_FLUSH_WRITER",
        "KVARN_SINKHORN_SOURCE",
        "KVARN_FORWARD_POOL_ENSURE",
        "KVARN_QLEN1_INLINE_PLAN",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(kvarn_attn.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(
        kvarn_attn, "kvarn_native_layout_abi_supported", lambda _op: True
    )
    KVarNAttentionImpl.reset_process_state()
    try:
        with patch(
            "vllm.v1.attention.backends.kvarn_attn.get_flash_attn_version",
            return_value=2,
        ):
            impl = KVarNAttentionImpl(
                num_heads=24,
                head_size=256,
                scale=1.0 / 16.0,
                num_kv_heads=4,
                kv_cache_dtype="kvarn_k4v4_g128_compact",
            )

        assert impl._kvarn_xpu_beta_profile
        assert impl._kvarn_cache_layout == "xe2_dpas"
        assert impl._kvarn_frontend_variant == "qkv_scatter_inline_current_stream"
        assert impl._kvarn_prefill_store_variant == "hadamard_scatter"
        assert impl._kvarn_native_split_policy == "b70_q6_id18_v1"
        assert impl._kvarn_native_max_splits == 32
        assert impl._kvarn_native_kernel_variant_name == "q6_prefetch_record_cursor"
        assert impl._kvarn_native_kernel_variant == 18
        assert impl._kvarn_flush_writer == "native_xe2"
        assert impl._kvarn_sinkhorn_source == "fused_materialized"
        assert impl._kvarn_forward_pool_ensure == "always"
        assert impl._kvarn_qlen1_inline_plan == "bound_native_v2"
        assert impl.use_bound_qlen1_inline_plan_v2
        assert KVarNAttentionImpl._flush_index_materialization == "shared"
    finally:
        KVarNAttentionImpl.reset_process_state()


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({}, True),
        ({"cache_layout": "natural"}, False),
        ({"head_dim": 128}, False),
        ({"group": 64}, False),
        ({"key_bits": 2}, False),
        ({"value_bits": 2}, False),
        ({"num_kv_heads": 8}, False),
        ({"record_bytes": 35_071}, False),
        ({"record_bytes": 35_074}, False),
        ({"op_available": False}, False),
        ({"rtn_quantile": 0.005}, False),
    ],
)
def test_native_balanced_writer_requires_exact_cache_abi(override, expected) -> None:
    values = {
        "cache_layout": "xe2_dpas",
        "head_dim": 256,
        "group": 128,
        "key_bits": 4,
        "value_bits": 4,
        "num_kv_heads": 4,
        "record_bytes": 65_536,
        "op_available": True,
        "rtn_quantile": 0.0,
    }
    values.update(override)
    assert _kvarn_native_balanced_writer_supported(**values) is expected


def test_native_balanced_writer_dispatch_preserves_tensor_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    balanced = tuple(torch.empty(1) for _ in range(6))
    block_ids = torch.tensor([4, 1], dtype=torch.int64)
    packed_cache = torch.empty(7, 4, 65_536, dtype=torch.uint8)
    call = Mock()
    fake_ops = SimpleNamespace(kvarn_pack_balanced_kv=call)
    monkeypatch.setattr(kvarn_attn.torch.ops, "_vllm_fa2_C", fake_ops)

    _launch_kvarn_native_balanced_writer(balanced, block_ids, packed_cache)

    call.assert_called_once_with(*balanced, block_ids, packed_cache, True)


def test_batched_flush_native_writer_bypasses_reference_record_assembly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    KVarNAttentionImpl.reset_process_state()
    KVarNAttentionImpl._select_flush_writer("native_xe2")
    group_key = ("native-writer-test",)
    cfg = SimpleNamespace(
        head_dim=256,
        group=128,
        record_bytes=65_536,
        k_packed_bytes=16_384,
        v_packed_bytes=16_384,
    )
    impl = SimpleNamespace(
        kvarn_config=cfg,
        num_kv_heads=4,
        _group_key=group_key,
        _tail_K_pool=torch.zeros(2, 128, 4, 256, dtype=torch.float16),
        _tail_V_pool=torch.zeros(2, 128, 4, 256, dtype=torch.float16),
        _tails={3: object(), 1: object()},
        _kvarn_cache_layout="xe2_dpas",
        _kvarn_flush_writer="native_xe2",
    )
    cache = torch.full((4, 4, 65_536), 0xA5, dtype=torch.uint8)
    KVarNAttentionImpl._block_to_slot_dict[group_key] = {3: 0, 1: 1}
    balanced = tuple(torch.empty(0) for _ in range(6))
    launch = Mock()

    try:
        with (
            patch.object(kvarn_attn, "_sinkhorn_balance_kv", return_value=balanced),
            patch.object(
                kvarn_attn,
                "_sinkhorn_pack_kv",
                side_effect=AssertionError("reference packer must not run"),
            ),
            patch.object(kvarn_attn, "_launch_kvarn_native_balanced_writer", launch),
        ):
            KVarNAttentionImpl._batched_flush([(impl, 3, cache), (impl, 1, cache)])

        assert not impl._tails
        launch.assert_called_once()
        args = launch.call_args.args
        assert args[0] is balanced
        selected_block_ids = args[1].tolist()
        assert selected_block_ids == [3, 1]
        assert len(selected_block_ids) == len(set(selected_block_ids))
        assert all(0 <= block < cache.shape[0] for block in selected_block_ids)
        assert args[2] is cache
    finally:
        KVarNAttentionImpl.reset_process_state()


def test_batched_flush_fused_sinkhorn_bypasses_unfused_materialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.v1.attention.backends.kvarn_attn as kvarn_attn

    KVarNAttentionImpl.reset_process_state()
    group_key = ("pool-indexed-sinkhorn-test",)
    cfg = SimpleNamespace(
        head_dim=256,
        group=128,
        record_bytes=65_536,
        k_packed_bytes=16_384,
        v_packed_bytes=16_384,
        sinkhorn_iters=8,
    )
    tail_key = torch.zeros(2, 128, 4, 256, dtype=torch.float16)
    tail_value = torch.zeros_like(tail_key)
    impl = SimpleNamespace(
        kvarn_config=cfg,
        num_kv_heads=4,
        _group_key=group_key,
        _tail_K_pool=tail_key,
        _tail_V_pool=tail_value,
        _tails={3: object(), 1: object()},
        _kvarn_cache_layout="xe2_dpas",
        _kvarn_flush_writer="native_xe2",
        _kvarn_sinkhorn_source="fused_materialized",
    )
    cache = torch.full((4, 4, 65_536), 0xA5, dtype=torch.uint8)
    KVarNAttentionImpl._block_to_slot_dict[group_key] = {3: 0, 1: 1}
    balanced = tuple(torch.empty(0) for _ in range(6))
    balance = Mock(return_value=balanced)
    launch = Mock()

    try:
        with (
            patch.object(kvarn_attn, "_sinkhorn_balance_fused_pool_kv", balance),
            patch.object(
                kvarn_attn,
                "_sinkhorn_balance_kv",
                side_effect=AssertionError("materialized Sinkhorn must not run"),
            ),
            patch.object(
                kvarn_attn,
                "_sinkhorn_pack_kv",
                side_effect=AssertionError("reference packer must not run"),
            ),
            patch.object(kvarn_attn, "_launch_kvarn_native_balanced_writer", launch),
        ):
            KVarNAttentionImpl._batched_flush([(impl, 3, cache), (impl, 1, cache)])

        assert not impl._tails
        balance.assert_called_once()
        balance_args = balance.call_args.args
        assert balance_args[0] is tail_key
        assert balance_args[1] is tail_value
        assert balance_args[2].tolist() == [0, 1]
        assert balance_args[3] is cfg
        launch.assert_called_once()
        launch_args = launch.call_args.args
        assert launch_args[0] is balanced
        assert launch_args[1].tolist() == [3, 1]
        assert launch_args[2] is cache
    finally:
        KVarNAttentionImpl.reset_process_state()


def test_fused_qkv_frontend_layer_filter_freezes_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    impl = _fused_frontend_impl()
    impl._kvarn_frontend_bound = False
    impl.use_fused_qkv_cache_update = False
    monkeypatch.setenv("KVARN_NATIVE_XPU_LAYER", "model.layers.1.self_attn")

    assert not impl.configure_fused_qkv_cache_update("model.layers.0.self_attn")
    monkeypatch.setenv("KVARN_NATIVE_XPU_LAYER", "model.layers.0.self_attn")
    assert not impl.configure_fused_qkv_cache_update("model.layers.0.self_attn")


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
        KVarNAttentionImpl._allocator_lifecycle_epochs[group_key] = 11
        KVarNAttentionImpl._max_known_block_id[group_key] = 7
        KVarNAttentionImpl._block_to_slot_t_per_device[mirror_key] = torch.ones(1)
        KVarNAttentionImpl._is_sink_t_per_device[mirror_key] = torch.ones(1)
        KVarNAttentionImpl._kernel_warmed.add(("decode", device))
        KVarNAttentionImpl._flush_index_materialization = "shared"
        KVarNAttentionImpl._flush_index_counters["flush_calls"] = 3
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
            KVarNAttentionImpl._allocator_lifecycle_epochs,
            KVarNAttentionImpl._block_to_slot_t_per_device,
            KVarNAttentionImpl._is_sink_t_per_device,
            KVarNAttentionImpl._max_known_block_id,
        )
        assert all(not mapping for mapping in mappings)
        assert not KVarNAttentionImpl._all_impls
        assert not KVarNAttentionImpl._kernel_warmed
        assert KVarNAttentionImpl._flush_index_materialization is None
        assert KVarNAttentionImpl.flush_index_materialization_counters() == {
            "flush_calls": 0,
            "layer_batches": 0,
            "schedule_groups": 0,
            "device_index_tensor_materializations": 0,
            "shared_layer_reuses": 0,
        }
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


def _epoch_ready_pool_impl():
    impl = object.__new__(KVarNAttentionImpl)
    device = torch.device("cpu")
    group_key = ("model.layers.0.self_attn",)
    mirror_key = (device, group_key)
    shared_key = (device, 8, 2)
    impl._group_key = group_key
    impl.kvarn_config = SimpleNamespace(head_dim=8)
    impl.num_kv_heads = 2
    impl.layer_name = group_key[0]
    impl._kvarn_forward_pool_ensure = "epoch_latch"
    impl._pool_epoch_latch_active_logged = False
    impl._tail_K_pool = torch.empty(1)
    impl._tail_V_pool = torch.empty(1)
    impl._H_fp16 = torch.empty(1)
    impl._k_rot_scratch = torch.empty(1)
    impl._v_rot_scratch = torch.empty(1)
    impl._pool_ready_key = mirror_key
    impl._pool_ready_native_key = None
    impl._native_decode_scratch = None

    block_to_slot = torch.empty(2048)
    is_sink = torch.empty(2048)
    KVarNAttentionImpl._block_to_slot_t_per_device[mirror_key] = block_to_slot
    KVarNAttentionImpl._is_sink_t_per_device[mirror_key] = is_sink
    KVarNAttentionImpl._mark_pool_mirror_changed(mirror_key)
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
        mapping[shared_key] = tensor
        setattr(impl, attr, tensor)
    KVarNAttentionImpl._mark_pool_shared_changed(shared_key)
    impl._record_pool_epoch_latch(device)
    return impl, device, group_key, mirror_key, shared_key, attr_maps


def _enable_beta_native_scratch_contract(
    impl: KVarNAttentionImpl, device: torch.device
) -> tuple[tuple, int, int]:
    impl._kvarn_xpu_beta_profile = True
    impl.use_bound_qlen1_inline_plan_v2 = True
    impl.num_heads = 24
    impl.head_size = 8
    impl._max_num_seqs = 4
    impl._max_model_len = 4096
    impl._kvarn_native_max_splits = 4
    impl._kvarn_native_split_policy = "fixed"
    impl._kvarn_native_kernel_variant = 18
    spec = impl._beta_native_scratch_spec(device)
    assert spec is not None
    return spec


def _native_scratch_for_test(
    batch_capacity: int, split_capacity: int, *, head_dim: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty(
            batch_capacity,
            24 * split_capacity,
            head_dim,
            dtype=torch.float16,
        ),
        torch.empty(batch_capacity, 24, split_capacity, dtype=torch.float32),
        torch.empty(batch_capacity, 24, split_capacity, dtype=torch.float32),
    )


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


def test_kvarn_and_nvfp4_ds_mla_quant_modes_are_distinct():
    assert get_kv_quant_mode("nvfp4_ds_mla") is KVQuantMode.NVFP4_DS_MLA
    assert all(get_kv_quant_mode(name).is_kvarn for name in KVARN_PRESETS)

    members = KVQuantMode.__members__.values()
    assert len(KVQuantMode.__members__) == len({member.value for member in members})
    assert {get_kv_quant_mode(name).value for name in KVARN_PRESETS} == set(
        range(100, 105)
    )


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


def test_pool_default_keeps_natural_decode_policy(monkeypatch):
    monkeypatch.delenv("KVARN_PREFILL_FP16_WINDOW_BLOCKS", raising=False)
    monkeypatch.delenv("KVARN_DECODE_FP16_WINDOW_BLOCKS", raising=False)
    monkeypatch.delenv("KVARN_DECODE_FP16_LOW_WATER_BLOCKS", raising=False)
    config = KVarNConfig.from_cache_dtype("kvarn_k4v4_g128_compact", head_dim=256)

    assert kvarn_prefill_fp16_window_blocks() == 16
    assert kvarn_decode_fp16_window_blocks() == 0
    assert kvarn_decode_fp16_low_water_blocks() == 0
    assert config.pool_slots(1, 2048) == 42


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


@pytest.mark.parametrize("output_ndim", [2, 3])
def test_kvarn_forward_preserves_direct_decode_output(output_ndim):
    impl = object.__new__(KVarNAttentionImpl)
    impl.kvarn_config = SimpleNamespace(record_bytes=35072)
    impl.num_heads = 2
    impl.num_kv_heads = 1
    impl.head_size = 4
    impl._ensure_pool = lambda device, num_blocks_hint: None

    def direct_decode(q, cache, metadata, output=None):
        assert output is not None
        output.fill_(7)
        return output

    impl._decode_path = direct_decode
    metadata = SimpleNamespace(
        num_actual_tokens=2,
        is_prefill=False,
        max_query_len=1,
        num_decodes=2,
        num_decode_tokens=2,
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
        torch.empty((1, 1, 35072), dtype=torch.uint8),
        metadata,
        output=output,
    )

    assert result is output
    assert torch.equal(output[:2], torch.full_like(output[:2], 7))
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


def test_prefill_fp16_window_chunk2_retains_available_recent_history():
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


def test_prefill_fp16_window_chunk3_flushes_older_than_bounded_suffix():
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
    ("flush_scope", "expected_deferred", "expected_protected", "expected_flushed"),
    [
        ("per_row", {1, 2, 3}, {14, 20, 21, 22, 23, 30, 31, 40}, {0}),
        ("batch_cohort", set(), {14, 23, 31, 40}, {0, 1, 2}),
    ],
)
def test_decode_flush_scope_coordinates_ragged_b4_crossing(
    flush_scope, expected_deferred, expected_protected, expected_flushed
):
    resident_by_row = {
        0: [14, 13, 12, 11, 10],
        1: [23, 22, 21, 20],
        2: [31, 30],
        3: [40],
    }
    blocks_needed = {50, 51, 52, 53}
    protected: set[int] = set()

    triggering, deferred, flushed = _coordinate_kvarn_decode_window_blocks(
        resident_by_row,
        high_water=4,
        low_water=1,
        flush_scope=flush_scope,
        blocks_needed=blocks_needed,
        protected_blocks=protected,
    )

    assert triggering == {0}
    assert deferred == expected_deferred
    assert flushed == expected_flushed
    assert protected == expected_protected
    assert blocks_needed == {50, 51, 52, 53} | expected_protected


@pytest.mark.parametrize("flush_scope", ["per_row", "batch_cohort"])
def test_decode_flush_scope_handles_synchronized_b4_crossing(flush_scope):
    resident_by_row = {
        row_index: list(range(row_index * 10 + 5, row_index * 10, -1))
        for row_index in range(4)
    }
    protected: set[int] = set()

    triggering, deferred, flushed = _coordinate_kvarn_decode_window_blocks(
        resident_by_row,
        high_water=4,
        low_water=0,
        flush_scope=flush_scope,
        blocks_needed=set(),
        protected_blocks=protected,
    )

    assert triggering == {0, 1, 2, 3}
    assert deferred == set()
    assert flushed == {0, 1, 2, 3}
    assert protected == set()


def test_decode_cohort_scan_is_bounded_per_row():
    class CountingRow(list):
        accesses = 0

        def __getitem__(self, index):
            self.accesses += 1
            return super().__getitem__(index)

    rows = [CountingRow(range(1000)) for _ in range(4)]
    resident = {bid: bid for bid in range(1000)}

    suffixes = [
        _kvarn_decode_resident_suffix(
            row,
            q_len=1,
            committed_len=1000 * 128,
            group=128,
            bt_cols=len(row),
            high_water=4,
            resident_blocks=resident,
        )
        for row in rows
    ]

    assert all(len(suffix) == 5 for suffix in suffixes)
    assert [row.accesses for row in rows] == [5, 5, 5, 5]


@pytest.mark.parametrize(
    ("window", "q_len", "committed_len"),
    [
        (0, 1, 384),
        (20, 256, 384),
        (20, 1, 0),
    ],
)
def test_decode_fp16_window_does_not_change_other_steps(
    monkeypatch, window, q_len, committed_len
):
    monkeypatch.setenv("KVARN_DECODE_FP16_WINDOW_BLOCKS", str(window))
    monkeypatch.setenv("KVARN_DECODE_FP16_LOW_WATER_BLOCKS", "0")
    row = [10, 11, 12, 13]
    resident = {10: 0, 11: 1, 12: 2, 99: 3}
    blocks_needed = {13}
    protected: set[int] = set()

    active, flush_required = _protect_kvarn_decode_window_blocks(
        row,
        q_len=q_len,
        committed_len=committed_len,
        group=128,
        bt_cols=len(row),
        high_water=_kvarn_decode_fp16_window_blocks(),
        low_water=_kvarn_decode_fp16_low_water_blocks(window),
        resident_blocks=resident,
        blocks_needed=blocks_needed,
        protected_blocks=protected,
    )

    assert not active
    assert not flush_required
    assert protected == set()
    assert blocks_needed == {13}


def test_prefill_fp16_window_defaults_to_beta_guardrail(monkeypatch):
    monkeypatch.delenv("KVARN_PREFILL_FP16_WINDOW_BLOCKS", raising=False)

    assert _kvarn_prefill_fp16_window_blocks() == 16


def test_decode_fp16_window_defaults_to_natural_reference(monkeypatch):
    monkeypatch.delenv("KVARN_DECODE_FP16_WINDOW_BLOCKS", raising=False)
    monkeypatch.delenv("KVARN_DECODE_FP16_LOW_WATER_BLOCKS", raising=False)

    assert _kvarn_decode_fp16_window_blocks() == 0
    assert _kvarn_decode_fp16_low_water_blocks(0) == 0


def test_decode_flush_scope_defaults_to_per_row(monkeypatch):
    monkeypatch.delenv("KVARN_DECODE_FLUSH_SCOPE", raising=False)

    assert _kvarn_decode_flush_scope() == "per_row"


class _ImmediateKVarNMetadataStageRing:
    depth = 1

    def acquire(self):
        return 0

    def drain(self):
        pass

    def release(self, stage, event_factory):
        assert stage == 0


def _make_kvarn_lifecycle_builder(resident, sinks, pool_size):
    KVarNAttentionImpl.reset_process_state()
    builder = object.__new__(KVarNMetadataBuilder)
    builder.reorder_batch_threshold = 4
    builder._group = 128
    builder._group_key = ("model.layers.0.self_attn",)
    builder._layer_names_set = set(builder._group_key)
    builder._retired_sinks = {}
    builder._block_fill = {bid: 128 for bid in resident}
    builder._max_model_len = 65536
    builder._metadata_stages = _ImmediateKVarNMetadataStageRing()
    builder._has_built_cudagraph_metadata = False
    builder._cu_seqlens_q_buf = torch.empty(257, dtype=torch.int32)
    builder._cu_seqlens_k_buf = torch.empty(257, dtype=torch.int32)
    builder._prefill_cu_seqlens_k_buf = torch.empty(257, dtype=torch.int32)
    builder._cu_seqlens_q_host = torch.empty((1, 257), dtype=torch.int32)
    builder._cu_seqlens_k_host = torch.empty((1, 257), dtype=torch.int32)
    builder._prefill_cu_seqlens_k_host = torch.empty((1, 257), dtype=torch.int32)
    builder._vq_req_buf = torch.empty(4096, dtype=torch.int32)
    builder._vq_seqlen_buf = torch.empty(4096, dtype=torch.int32)
    builder._vq_req_host = torch.empty((1, 4096), dtype=torch.int32)
    builder._vq_seqlen_host = torch.empty((1, 4096), dtype=torch.int32)
    group_key = builder._group_key
    device = torch.device("cpu")
    resident = sorted(resident)
    assert len(resident) <= pool_size
    block_to_slot = {bid: slot for slot, bid in enumerate(resident)}
    lookup_size = max([4096, *(bid + 1 for bid in resident)])
    lookup = torch.full((lookup_size,), -1, dtype=torch.int32)
    is_sink = torch.zeros(lookup_size, dtype=torch.bool)
    for bid, slot in block_to_slot.items():
        lookup[bid] = slot
    is_sink[list(sinks)] = True
    KVarNAttentionImpl._block_to_slot_dict[group_key] = block_to_slot
    KVarNAttentionImpl._global_sink_blocks[group_key] = set(sinks)
    KVarNAttentionImpl._free_slots[group_key] = list(
        range(pool_size - 1, len(resident) - 1, -1)
    )
    KVarNAttentionImpl._allocator_pool_size[group_key] = pool_size
    KVarNAttentionImpl._block_to_slot_t_per_device[(device, group_key)] = lookup
    KVarNAttentionImpl._is_sink_t_per_device[(device, group_key)] = is_sink
    KVarNAttentionImpl._max_known_block_id[group_key] = max(resident, default=0)
    KVarNAttentionImpl._bump_allocator_lifecycle_epoch(group_key)

    impl = SimpleNamespace(
        layer_name=next(iter(group_key)),
        _group_key=group_key,
        _kv_cache_ref=object(),
        _ensure_pool=lambda device, num_blocks_hint: None,
    )
    KVarNAttentionImpl._all_impls.append(impl)
    return builder, block_to_slot


def _kvarn_lifecycle_common(rows, seq_lens, query_lens):
    width = max((len(row) for row in rows), default=0)
    block_table_cpu = np.full((len(rows), width), -1, dtype=np.int32)
    for i, row in enumerate(rows):
        block_table_cpu[i, : len(row)] = row
    query_start_loc = torch.tensor(
        [0, *np.cumsum(query_lens).tolist()], dtype=torch.int32
    )
    return SimpleNamespace(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        seq_lens_cpu=torch.tensor(seq_lens, dtype=torch.int32),
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        block_table_tensor=torch.from_numpy(block_table_cpu),
        block_table_cpu=block_table_cpu,
        slot_mapping=torch.arange(sum(query_lens), dtype=torch.int64),
        num_actual_tokens=sum(query_lens),
        max_query_len=max(query_lens, default=1),
        max_seq_len=max(seq_lens, default=0),
        causal=True,
    )


def _run_kvarn_lifecycle_build(
    builder,
    rows,
    seq_lens,
    query_lens,
    *,
    num_decodes,
    flush_calls,
):
    cam = _kvarn_lifecycle_common(rows, seq_lens, query_lens)
    num_decode_tokens = sum(query_lens[:num_decodes])

    def record_flush(flush_pairs):
        flush_calls.append([bid for _, bid, _ in flush_pairs])

    with (
        patch(
            "vllm.v1.attention.backends.kvarn_attn.split_decodes_and_prefills",
            return_value=(
                num_decodes,
                len(rows) - num_decodes,
                num_decode_tokens,
                sum(query_lens) - num_decode_tokens,
            ),
        ),
        patch.object(KVarNAttentionImpl, "_batched_flush", side_effect=record_flush),
    ):
        return builder.build(0, cam)


def test_lifecycle_builder_preserves_shared_multiquery_prefix():
    rows = [
        [0, 1, 2, 100, 101, 102, 103],
        [0, 1, 2, 200, 201, 202],
    ]
    resident = {0, 1, 2, 100, 101, 102, 200, 201}
    builder, block_to_slot = _make_kvarn_lifecycle_builder(
        resident, sinks={0}, pool_size=len(resident) + 2
    )
    flush_calls = []
    try:
        _run_kvarn_lifecycle_build(
            builder,
            rows,
            [6 * 128 + 1, 5 * 128 + 2],
            [1, 2],
            num_decodes=1,
            flush_calls=flush_calls,
        )

        assert flush_calls == [[102, 101, 100]]
        assert {0, 1, 2, 200, 201, 103, 202} <= block_to_slot.keys()
    finally:
        KVarNAttentionImpl.reset_process_state()


def test_lifecycle_builder_flushes_before_capacity_and_reclaims_completion():
    rows = []
    resident = set()
    sinks = set()
    current = set()
    expected_flush = []
    for batch_index in range(4):
        base = batch_index * 100
        sinks.add(base)
        resident.update(range(base, base + 6))
        current.add(base + 6)
        rows.append(list(range(base, base + 7)))
        expected_flush.extend(range(base + 5, base, -1))
    builder, block_to_slot = _make_kvarn_lifecycle_builder(
        resident, sinks=sinks, pool_size=len(resident)
    )
    flush_calls = []
    try:
        _run_kvarn_lifecycle_build(
            builder,
            rows,
            [6 * 128 + 1] * 4,
            [1] * 4,
            num_decodes=4,
            flush_calls=flush_calls,
        )

        assert flush_calls == [expected_flush]
        assert set(block_to_slot) == sinks | current
        assert len(KVarNAttentionImpl._free_slots[builder._group_key]) == 16

        _run_kvarn_lifecycle_build(
            builder,
            [],
            [],
            [],
            num_decodes=0,
            flush_calls=flush_calls,
        )

        assert set(block_to_slot) == sinks
        assert builder._retired_sinks == dict.fromkeys(sinks)
        assert len(KVarNAttentionImpl._free_slots[builder._group_key]) == 20
    finally:
        KVarNAttentionImpl.reset_process_state()


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
