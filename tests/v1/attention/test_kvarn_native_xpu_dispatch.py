# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

import vllm.v1.attention.ops.triton_kvarn_decode as kvarn_decode
from vllm.v1.attention.ops.triton_kvarn_decode import (
    _kvarn_dpas_layout_for_problem,
    _kvarn_native_output_hadamard_enabled,
    _kvarn_native_scratch_views,
    _kvarn_op_supports_argument,
    _require_kvarn_dpas_reader,
    kvarn_cache_layout_requested,
    kvarn_dpas_layout_requested,
    kvarn_frontend_variant_requested,
    kvarn_native_bf16_output_supported,
    kvarn_native_decode_abi_supported,
    kvarn_native_feature_enabled,
    kvarn_native_kernel_variant_requested,
    kvarn_native_layer_selected,
    kvarn_native_layout_abi_supported,
    kvarn_native_prefill_store_supported,
    kvarn_native_problem_supported,
    kvarn_native_split_count,
    kvarn_native_split_policy_requested,
    kvarn_native_split_scratch_count,
    kvarn_native_store_supported,
    kvarn_prefill_store_variant_requested,
    validate_kvarn_native_factory_selection,
)


def test_fused_qkv_frontend_requires_explicit_valid_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KVARN_NATIVE_XPU_FRONTEND", raising=False)
    assert kvarn_frontend_variant_requested() == "reference"

    monkeypatch.setenv("KVARN_NATIVE_XPU_FRONTEND", "qkv_scatter")
    assert kvarn_frontend_variant_requested() == "qkv_scatter"

    monkeypatch.setenv("KVARN_NATIVE_XPU_FRONTEND", "qkv_scatter_inline")
    assert kvarn_frontend_variant_requested() == "qkv_scatter_inline"

    monkeypatch.setenv("KVARN_NATIVE_XPU_FRONTEND", "qkv_scatter_inline_current_stream")
    assert kvarn_frontend_variant_requested() == "qkv_scatter_inline_current_stream"

    monkeypatch.setenv("KVARN_NATIVE_XPU_FRONTEND", "automatic")
    with pytest.raises(ValueError, match="KVARN_NATIVE_XPU_FRONTEND"):
        kvarn_frontend_variant_requested()


def test_prefill_store_requires_explicit_valid_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KVARN_NATIVE_XPU_PREFILL_STORE", raising=False)
    assert kvarn_prefill_store_variant_requested() == "reference"

    monkeypatch.setenv("KVARN_NATIVE_XPU_PREFILL_STORE", "hadamard_scatter")
    assert kvarn_prefill_store_variant_requested() == "hadamard_scatter"

    monkeypatch.setenv("KVARN_NATIVE_XPU_PREFILL_STORE", "automatic")
    with pytest.raises(ValueError, match="KVARN_NATIVE_XPU_PREFILL_STORE"):
        kvarn_prefill_store_variant_requested()


def _native_problem(**overrides) -> dict:
    problem = dict(
        device_type="xpu",
        batch_size=4,
        num_query_heads=24,
        num_kv_heads=4,
        head_dim=256,
        group=128,
        key_bits=4,
        value_bits=4,
        record_bytes=35_072,
        sliding_window=0,
        has_lookup=True,
        has_tail_pool=True,
        is_capturing=False,
    )
    problem.update(overrides)
    return problem


def test_native_feature_master_and_subfeature_toggles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KVARN_NATIVE_XPU", raising=False)
    monkeypatch.setattr(kvarn_decode.current_platform, "is_xpu", lambda: False)
    assert not kvarn_native_feature_enabled("DECODE")

    monkeypatch.setattr(kvarn_decode.current_platform, "is_xpu", lambda: True)
    assert kvarn_native_feature_enabled("DECODE")
    assert kvarn_native_feature_enabled("MATERIALIZE")

    monkeypatch.setenv("KVARN_NATIVE_XPU", "1")
    assert kvarn_native_feature_enabled("DECODE")
    assert kvarn_native_feature_enabled("MATERIALIZE")

    monkeypatch.setenv("KVARN_NATIVE_XPU_DECODE", "0")
    assert not kvarn_native_feature_enabled("DECODE")
    assert kvarn_native_feature_enabled("MATERIALIZE")

    monkeypatch.setenv("KVARN_NATIVE_XPU_MATERIALIZE", "0")
    assert not kvarn_native_feature_enabled("MATERIALIZE")


def _native_store(**overrides) -> dict:
    problem = dict(
        device_type="xpu",
        num_tokens=4,
        num_query_heads=24,
        num_kv_heads=4,
        head_dim=256,
        group=128,
        key_bits=4,
        value_bits=4,
        record_bytes=35_072,
        sliding_window=0,
        key_dtype=torch.bfloat16,
        value_dtype=torch.bfloat16,
        has_lookup=True,
        has_tail_pool=True,
        is_capturing=False,
        op_available=True,
    )
    problem.update(overrides)
    return problem


def _native_prefill_store(**overrides) -> dict:
    problem = _native_store(num_tokens=4096)
    problem.update(overrides)
    return problem


def test_native_store_follows_decode_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("KVARN_NATIVE_XPU", raising=False)
    monkeypatch.setattr(kvarn_decode.current_platform, "is_xpu", lambda: True)
    assert kvarn_native_store_supported(**_native_store())

    monkeypatch.setenv("KVARN_NATIVE_XPU", "0")
    assert not kvarn_native_store_supported(**_native_store())

    monkeypatch.setenv("KVARN_NATIVE_XPU", "1")
    assert kvarn_native_store_supported(**_native_store())

    monkeypatch.setenv("KVARN_NATIVE_XPU_DECODE", "0")
    assert not kvarn_native_store_supported(**_native_store())


def test_native_prefill_store_follows_master_switch_and_has_no_decode_batch_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU", "1")
    assert kvarn_native_prefill_store_supported(**_native_prefill_store())

    monkeypatch.setenv("KVARN_NATIVE_XPU", "0")
    assert not kvarn_native_prefill_store_supported(**_native_prefill_store())


@pytest.mark.parametrize(
    "override",
    [
        {"device_type": "cuda"},
        {"num_tokens": 1},
        {"num_query_heads": 32},
        {"num_kv_heads": 8},
        {"head_dim": 128},
        {"group": 64},
        {"value_bits": 2},
        {"record_bytes": 35_071},
        {"sliding_window": 1024},
        {"key_dtype": torch.float32, "value_dtype": torch.float32},
        {"value_dtype": torch.float16},
        {"has_lookup": False},
        {"has_tail_pool": False},
        {"is_capturing": True},
        {"op_available": False},
    ],
)
def test_native_prefill_store_rejects_unsupported_dispatch(
    monkeypatch: pytest.MonkeyPatch, override: dict
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU", "1")
    assert not kvarn_native_prefill_store_supported(**_native_prefill_store(**override))


@pytest.mark.parametrize(
    "override",
    [
        {"num_tokens": 0},
        {"num_tokens": 13},
        {"num_query_heads": 32},
        {"num_kv_heads": 8},
        {"head_dim": 128},
        {"group": 64},
        {"value_bits": 2},
        {"record_bytes": 35_071},
        {"sliding_window": 1024},
        {"key_dtype": torch.float32, "value_dtype": torch.float32},
        {"value_dtype": torch.float16},
        {"has_lookup": False},
        {"has_tail_pool": False},
        {"is_capturing": True},
        {"op_available": False},
    ],
)
def test_native_store_rejects_unsupported_dispatch(
    monkeypatch: pytest.MonkeyPatch, override: dict
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU", "1")
    assert not kvarn_native_store_supported(**_native_store(**override))


@pytest.mark.parametrize("record_bytes", [35_072, 65_536])
def test_native_problem_accepts_record_stride(record_bytes: int) -> None:
    assert kvarn_native_problem_supported(**_native_problem(record_bytes=record_bytes))


@pytest.mark.parametrize(
    ("override"),
    [
        {"batch_size": 13},
        {"head_dim": 128},
        {"num_kv_heads": 8},
        {"record_bytes": 35_071},
        {"record_bytes": 35_074},
        {"sliding_window": 1024},
        {"is_capturing": True},
        {"has_lookup": False},
        {"has_tail_pool": False},
    ],
)
def test_native_problem_rejects_unsupported_abi(override: dict) -> None:
    assert not kvarn_native_problem_supported(**_native_problem(**override))


def test_native_problem_accepts_matching_dpas_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KVARN_NATIVE_XPU_CACHE_LAYOUT", raising=False)
    monkeypatch.delenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", raising=False)
    assert kvarn_cache_layout_requested() == "natural"
    assert kvarn_native_problem_supported(**_native_problem())

    monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "1")
    assert kvarn_cache_layout_requested() == "xe2_dpas"
    assert kvarn_dpas_layout_requested()
    assert kvarn_native_problem_supported(**_native_problem())

    monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "0")
    assert not kvarn_dpas_layout_requested()
    assert kvarn_native_problem_supported(**_native_problem())


def test_dpas_layout_refuses_natural_reader_fallback() -> None:
    _require_kvarn_dpas_reader(True, True, "test reader")
    with pytest.raises(RuntimeError, match="refusing the natural-layout"):
        _require_kvarn_dpas_reader(True, False, "test reader")
    _require_kvarn_dpas_reader(False, False, "test reader")


def test_dpas_layout_problem_validation_is_fail_closed() -> None:
    assert not _kvarn_dpas_layout_for_problem(False, 128, 64, 2, 2)
    assert _kvarn_dpas_layout_for_problem(True, 256, 128, 4, 4)
    with pytest.raises(RuntimeError, match="requires D256/G128/K4V4"):
        _kvarn_dpas_layout_for_problem(True, 256, 128, 4, 2)


def test_named_layout_selector_rejects_conflicts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU_CACHE_LAYOUT", "natural")
    monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "1")
    with pytest.raises(ValueError, match="conflicts"):
        kvarn_cache_layout_requested()

    monkeypatch.delenv("KVARN_NATIVE_XPU_DPAS_LAYOUT")
    monkeypatch.setenv("KVARN_NATIVE_XPU_CACHE_LAYOUT", "future_layout")
    with pytest.raises(ValueError, match="must be one of"):
        kvarn_cache_layout_requested()


def test_native_kernel_variant_selection_is_named_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KVARN_NATIVE_XPU_KERNEL_VARIANT", raising=False)
    assert kvarn_native_kernel_variant_requested() == ("baseline", 0)

    monkeypatch.setenv("KVARN_NATIVE_XPU_KERNEL_VARIANT", "unknown")
    with pytest.raises(ValueError, match="must be one of"):
        kvarn_native_kernel_variant_requested()

    monkeypatch.setenv("KVARN_NATIVE_XPU_KERNEL_VARIANT", "q6_b1_short_last_producer")
    with pytest.raises(ValueError, match="must be one of"):
        kvarn_native_kernel_variant_requested()


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        ("baseline", 0),
        ("qk_i8u4", 1),
        ("q6_scalar", 2),
        ("q8_vector", 3),
        ("q6_vector", 4),
        ("q6_cached_weights", 6),
        ("q6_exact_rows", 7),
        ("q6_cached_weights_exact_rows", 8),
        ("q6_page_pair", 9),
        ("q6_main_grf128", 10),
        ("q6_split_reducer_specialized", 11),
        ("q6_next_page_prefetch", 12),
        ("q6_next_page_prefetch_split_reducer", 13),
        ("q6_simd_unpack", 14),
        ("q6_block_output_store", 15),
        ("q6_current_half_v_prefetch", 16),
        ("q6_page_record_cursor", 17),
        ("q6_prefetch_record_cursor", 18),
        ("q6_page_metadata_cursor", 20),
        ("q6_paired_nibble_half2", 21),
    ],
)
def test_native_kernel_variant_factory_ids_are_stable(
    monkeypatch: pytest.MonkeyPatch, name: str, variant: int
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU_KERNEL_VARIANT", name)
    assert kvarn_native_kernel_variant_requested() == (name, variant)


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        ("qk_i8u4", 1),
        ("q6_scalar", 2),
        ("q8_vector", 3),
        ("q6_vector", 4),
        ("q6_cached_weights", 6),
        ("q6_exact_rows", 7),
        ("q6_cached_weights_exact_rows", 8),
        ("q6_page_pair", 9),
        ("q6_main_grf128", 10),
        ("q6_split_reducer_specialized", 11),
        ("q6_next_page_prefetch", 12),
        ("q6_next_page_prefetch_split_reducer", 13),
        ("q6_simd_unpack", 14),
        ("q6_block_output_store", 15),
        ("q6_current_half_v_prefetch", 16),
        ("q6_page_record_cursor", 17),
        ("q6_prefetch_record_cursor", 18),
        ("q6_page_metadata_cursor", 20),
        ("q6_paired_nibble_half2", 21),
    ],
)
def test_native_experimental_variants_require_dpas_cache_layout(
    name: str, variant: int
) -> None:
    validate_kvarn_native_factory_selection("xe2_dpas", name, variant)
    with pytest.raises(ValueError, match="requires.*xe2_dpas"):
        validate_kvarn_native_factory_selection("natural", name, variant)

    validate_kvarn_native_factory_selection("natural", "baseline", 0)


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        ("q6_scalar", 2),
        ("q6_vector", 4),
        ("q6_cached_weights", 6),
        ("q6_exact_rows", 7),
        ("q6_cached_weights_exact_rows", 8),
        ("q6_page_pair", 9),
        ("q6_main_grf128", 10),
        ("q6_split_reducer_specialized", 11),
        ("q6_next_page_prefetch", 12),
        ("q6_next_page_prefetch_split_reducer", 13),
        ("q6_simd_unpack", 14),
        ("q6_block_output_store", 15),
        ("q6_current_half_v_prefetch", 16),
        ("q6_page_record_cursor", 17),
        ("q6_prefetch_record_cursor", 18),
        ("q6_page_metadata_cursor", 20),
        ("q6_paired_nibble_half2", 21),
    ],
)
def test_b70_q6_split_policy_requires_q6_kernel(name: str, variant: int) -> None:
    validate_kvarn_native_factory_selection(
        "xe2_dpas", name, variant, split_policy="b70_q6"
    )
    with pytest.raises(ValueError, match="requires a Q6 kernel variant"):
        validate_kvarn_native_factory_selection(
            "xe2_dpas", "baseline", 0, split_policy="b70_q6"
        )


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        ("q6_next_page_prefetch", 12),
        ("q6_next_page_prefetch_split_reducer", 13),
    ],
)
def test_b70_q6_v2_split_policy_requires_profiled_kernels(
    name: str, variant: int
) -> None:
    validate_kvarn_native_factory_selection(
        "xe2_dpas", name, variant, split_policy="b70_q6_v2"
    )
    with pytest.raises(
        ValueError, match="requires kernel variant.*q6_next_page_prefetch"
    ):
        validate_kvarn_native_factory_selection(
            "xe2_dpas", "q6_scalar", 2, split_policy="b70_q6_v2"
        )


@pytest.mark.parametrize(
    ("name", "variant"),
    [
        ("q6_page_metadata_cursor", 20),
        ("q6_paired_nibble_half2", 21),
    ],
)
def test_round6_variants_do_not_expand_b70_q6_v2(name: str, variant: int) -> None:
    with pytest.raises(
        ValueError, match="requires kernel variant.*q6_next_page_prefetch"
    ):
        validate_kvarn_native_factory_selection(
            "xe2_dpas", name, variant, split_policy="b70_q6_v2"
        )


def test_b70_q6_id18_v1_requires_profiled_kernel() -> None:
    validate_kvarn_native_factory_selection(
        "xe2_dpas",
        "q6_prefetch_record_cursor",
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_PREFETCH_RECORD_CURSOR,
        split_policy="b70_q6_id18_v1",
    )
    with pytest.raises(
        ValueError, match="requires kernel variant.*q6_prefetch_record_cursor"
    ):
        validate_kvarn_native_factory_selection(
            "xe2_dpas",
            "q6_next_page_prefetch",
            kvarn_decode.KVARN_NATIVE_KERNEL_Q6_NEXT_PAGE_PREFETCH,
            split_policy="b70_q6_id18_v1",
        )


def test_native_kernel_variant_five_remains_reserved() -> None:
    with pytest.raises(ValueError, match="variant 5 is reserved"):
        validate_kvarn_native_factory_selection("xe2_dpas", "page128", 5)


@pytest.mark.parametrize(
    ("name", "variant"),
    [("q6_page_pair", 10), ("unknown", 9), ("baseline", 99)],
)
def test_native_kernel_variant_validation_rejects_unregistered_pairs(
    name: str, variant: int
) -> None:
    with pytest.raises(ValueError, match="name/id pair is not registered"):
        validate_kvarn_native_factory_selection("xe2_dpas", name, variant)


def test_native_layer_filter_matches_components() -> None:
    kvarn_native_layer_selected.cache_clear()
    name = "model.layers.17.self_attn.attn"
    assert kvarn_native_layer_selected(name, "")
    assert kvarn_native_layer_selected(name, "layers.17")
    assert kvarn_native_layer_selected(name, "layers.3,layers.17.self_attn")
    assert not kvarn_native_layer_selected(name, "layers.1")
    assert not kvarn_native_layer_selected(name, "layers.170")

    before = kvarn_native_layer_selected.cache_info()
    assert kvarn_native_layer_selected(name, "layers.17")
    after = kvarn_native_layer_selected.cache_info()
    assert after.hits == before.hits + 1


def test_native_split_count_matches_cpp_short_context_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kvarn_decode._kvarn_native_split_count_cached.cache_clear()
    monkeypatch.delenv("KVARN_NATIVE_XPU_SPLIT_POLICY", raising=False)
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "32")
    assert kvarn_native_split_count(4096) == 32
    assert kvarn_native_split_count(4096) == 32
    assert kvarn_decode._kvarn_native_split_count_cached.cache_info().hits == 1
    assert kvarn_native_split_count(1024) == 1

    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "16")
    assert kvarn_native_split_count(4096) == 16

    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "3")
    with pytest.raises(ValueError, match="must be one of"):
        kvarn_native_split_count(4096)


def test_native_split_count_defaults_to_validated_multisplit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kvarn_decode._kvarn_native_split_count_cached.cache_clear()
    monkeypatch.delenv("KVARN_NATIVE_XPU_SPLIT_POLICY", raising=False)
    monkeypatch.delenv("KVARN_NATIVE_XPU_SPLITS", raising=False)

    assert kvarn_native_split_policy_requested() == ("fixed", 16)
    assert kvarn_native_split_count(4096) == 16
    assert "KVARN_NATIVE_XPU_SPLITS" not in os.environ
    assert kvarn_native_split_count(512) == 1


def test_page_pair_split_count_matches_cpp_k128_boundaries() -> None:
    page_pair = kvarn_decode.KVARN_NATIVE_KERNEL_Q6_PAGE_PAIR

    # At S=32 the K64 variants have 32 work units here, but page-pair has only
    # 16 and must collapse exactly as the C++ wrapper does.
    assert kvarn_native_split_count(1985, 32) == 32
    assert kvarn_native_split_count(1985, 32, kernel_variant=page_pair) == 1

    # The page-pair launch reaches its 32nd K128 unit immediately after the
    # end of page 31.
    assert kvarn_native_split_count(3968, 32, kernel_variant=page_pair) == 1
    assert kvarn_native_split_count(3969, 32, kernel_variant=page_pair) == 32
    assert kvarn_native_split_scratch_count(1985, 32, "fixed") == 32
    assert (
        kvarn_native_split_scratch_count(
            1985,
            32,
            "fixed",
            page_pair,
        )
        == 1
    )


@pytest.mark.parametrize(
    "kernel_variant",
    [
        kvarn_decode.KVARN_NATIVE_KERNEL_BASELINE,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_SCALAR,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_NEXT_PAGE_PREFETCH,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_NEXT_PAGE_PREFETCH_SPLIT_REDUCER,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_SIMD_UNPACK,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_BLOCK_OUTPUT_STORE,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_CURRENT_HALF_V_PREFETCH,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_PAGE_RECORD_CURSOR,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_PREFETCH_RECORD_CURSOR,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_PAGE_METADATA_CURSOR,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_PAIRED_NIBBLE_HALF2,
    ],
)
def test_k64_variants_keep_established_split_boundaries(kernel_variant: int) -> None:
    assert kvarn_native_split_count(1984, 32, kernel_variant=kernel_variant) == 1
    assert kvarn_native_split_count(1985, 32, kernel_variant=kernel_variant) == 32


@pytest.mark.parametrize("kernel_variant", [5, 99, -1])
def test_native_split_count_rejects_reserved_or_unknown_variants(
    kernel_variant: int,
) -> None:
    with pytest.raises(ValueError, match="reserved|unknown"):
        kvarn_native_split_count(4096, 32, kernel_variant=kernel_variant)


@pytest.mark.parametrize(
    ("batch_size", "expected_splits"),
    [(1, 32), (2, 16), (3, 8), (4, 8), (5, 4), (8, 4), (9, 2), (12, 2)],
)
def test_b70_q6_split_policy_is_batch_aware(
    monkeypatch: pytest.MonkeyPatch, batch_size: int, expected_splits: int
) -> None:
    kvarn_decode._kvarn_native_split_count_cached.cache_clear()
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLIT_POLICY", "b70_q6")
    monkeypatch.delenv("KVARN_NATIVE_XPU_SPLITS", raising=False)

    policy, max_splits = kvarn_native_split_policy_requested()

    assert (policy, max_splits) == ("b70_q6", 32)
    assert (
        kvarn_native_split_count(
            4096,
            max_splits,
            batch_size=batch_size,
            split_policy=policy,
        )
        == expected_splits
    )
    assert kvarn_native_split_scratch_count(4096, max_splits, policy) == 32


def test_b70_q6_split_policy_preserves_short_context_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLIT_POLICY", "b70_q6")
    monkeypatch.delenv("KVARN_NATIVE_XPU_SPLITS", raising=False)

    assert kvarn_native_split_count(1024, batch_size=1) == 1
    assert kvarn_native_split_count(1024, batch_size=2) == 16


@pytest.mark.parametrize(
    ("batch_size", "context", "expected_splits"),
    [
        (1, 4096, 32),
        (1, 16_384, 32),
        (1, 65_023, 32),
        (2, 65_023, 16),
        (3, 65_023, 8),
        (4, 4096, 8),
        (4, 16_384, 8),
        (4, 48 * 1024, 8),
        (4, 48 * 1024 + 1, 32),
        (4, 65_023, 32),
        (5, 65_023, 4),
        (8, 65_023, 4),
        (9, 65_023, 2),
        (12, 65_023, 2),
    ],
)
@pytest.mark.parametrize(
    "kernel_variant",
    [
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_NEXT_PAGE_PREFETCH,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_NEXT_PAGE_PREFETCH_SPLIT_REDUCER,
    ],
)
def test_b70_q6_v2_split_policy_context_and_batch_boundaries(
    batch_size: int, context: int, expected_splits: int, kernel_variant: int
) -> None:
    assert (
        kvarn_native_split_count(
            context,
            32,
            batch_size=batch_size,
            split_policy="b70_q6_v2",
            kernel_variant=kernel_variant,
        )
        == expected_splits
    )


@pytest.mark.parametrize(
    ("batch_size", "context", "expected_splits"),
    [
        # Both profiled variants use K64 work units. S=32 becomes valid on the
        # 32nd work unit.
        (1, 1984, 1),
        (1, 1985, 32),
        # Before the long-context B4 switch, S=8 becomes valid on unit eight.
        (4, 448, 1),
        (4, 449, 8),
    ],
)
@pytest.mark.parametrize(
    "kernel_variant",
    [
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_NEXT_PAGE_PREFETCH,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_NEXT_PAGE_PREFETCH_SPLIT_REDUCER,
    ],
)
def test_b70_q6_v2_preserves_k64_work_unit_collapse(
    batch_size: int, context: int, expected_splits: int, kernel_variant: int
) -> None:
    assert (
        kvarn_native_split_count(
            context,
            32,
            batch_size=batch_size,
            split_policy="b70_q6_v2",
            kernel_variant=kernel_variant,
        )
        == expected_splits
    )


@pytest.mark.parametrize(
    "variant",
    [
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_NEXT_PAGE_PREFETCH,
        kvarn_decode.KVARN_NATIVE_KERNEL_Q6_NEXT_PAGE_PREFETCH_SPLIT_REDUCER,
    ],
)
def test_b70_q6_v2_reserves_max_scratch_across_dynamic_schedule(
    variant: int,
) -> None:
    assert kvarn_native_split_scratch_count(4096, 32, "b70_q6_v2", variant) == 32
    assert kvarn_native_split_scratch_count(65_023, 32, "b70_q6_v2", variant) == 32
    with pytest.raises(ValueError, match="requires max_splits=32"):
        kvarn_native_split_count(
            65_023,
            16,
            batch_size=4,
            split_policy="b70_q6_v2",
            kernel_variant=variant,
        )


def test_b70_q6_v2_policy_request_and_legacy_policy_remains_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLIT_POLICY", "b70_q6_v2")
    monkeypatch.delenv("KVARN_NATIVE_XPU_SPLITS", raising=False)
    assert kvarn_native_split_policy_requested() == ("b70_q6_v2", 32)

    # V1 is intentionally stable even where V2 changes B4 to S=32.
    assert (
        kvarn_native_split_count(
            65_023,
            32,
            batch_size=4,
            split_policy="b70_q6",
            kernel_variant=kvarn_decode.KVARN_NATIVE_KERNEL_Q6_NEXT_PAGE_PREFETCH,
        )
        == 8
    )

    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "32")
    with pytest.raises(ValueError, match="conflicts.*b70_q6_v2"):
        kvarn_native_split_policy_requested()


@pytest.mark.parametrize("batch_size", [0, 13])
def test_b70_q6_v2_rejects_unsupported_batch_boundaries(batch_size: int) -> None:
    with pytest.raises(ValueError, match="batch sizes 1 through 12"):
        kvarn_native_split_count(
            4096,
            32,
            batch_size=batch_size,
            split_policy="b70_q6_v2",
            kernel_variant=kvarn_decode.KVARN_NATIVE_KERNEL_Q6_NEXT_PAGE_PREFETCH,
        )


@pytest.mark.parametrize(
    ("batch_size", "expected_splits"),
    [(1, 32), (2, 16), (3, 8), (4, 24), (5, 4), (8, 4), (9, 2), (12, 2)],
)
def test_b70_q6_id18_v1_split_policy_is_batch_aware(
    batch_size: int, expected_splits: int
) -> None:
    assert (
        kvarn_native_split_count(
            65_023,
            32,
            batch_size=batch_size,
            split_policy="b70_q6_id18_v1",
            kernel_variant=kvarn_decode.KVARN_NATIVE_KERNEL_Q6_PREFETCH_RECORD_CURSOR,
        )
        == expected_splits
    )


def test_b70_q6_id18_v1_policy_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLIT_POLICY", "b70_q6_id18_v1")
    monkeypatch.delenv("KVARN_NATIVE_XPU_SPLITS", raising=False)

    assert kvarn_native_split_policy_requested() == ("b70_q6_id18_v1", 32)
    assert (
        kvarn_native_split_scratch_count(
            65_023,
            32,
            "b70_q6_id18_v1",
            kvarn_decode.KVARN_NATIVE_KERNEL_Q6_PREFETCH_RECORD_CURSOR,
        )
        == 32
    )
    assert (
        kvarn_native_split_count(
            1472,
            32,
            batch_size=4,
            split_policy="b70_q6_id18_v1",
            kernel_variant=kvarn_decode.KVARN_NATIVE_KERNEL_Q6_PREFETCH_RECORD_CURSOR,
        )
        == 1
    )
    assert (
        kvarn_native_split_count(
            1473,
            32,
            batch_size=4,
            split_policy="b70_q6_id18_v1",
            kernel_variant=kvarn_decode.KVARN_NATIVE_KERNEL_Q6_PREFETCH_RECORD_CURSOR,
        )
        == 24
    )
    with pytest.raises(ValueError, match="requires max_splits=32"):
        kvarn_native_split_count(
            65_023,
            24,
            batch_size=4,
            split_policy="b70_q6_id18_v1",
            kernel_variant=kvarn_decode.KVARN_NATIVE_KERNEL_Q6_PREFETCH_RECORD_CURSOR,
        )

    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "24")
    with pytest.raises(ValueError, match="conflicts.*b70_q6_id18_v1"):
        kvarn_native_split_policy_requested()


def test_b70_q6_page_pair_uses_k128_but_keeps_full_scratch_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLIT_POLICY", "b70_q6")
    monkeypatch.delenv("KVARN_NATIVE_XPU_SPLITS", raising=False)
    page_pair = kvarn_decode.KVARN_NATIVE_KERNEL_Q6_PAGE_PAIR

    assert (
        kvarn_native_split_count(
            1985,
            batch_size=1,
            kernel_variant=page_pair,
        )
        == 1
    )
    assert (
        kvarn_native_split_count(
            3969,
            batch_size=1,
            kernel_variant=page_pair,
        )
        == 32
    )
    assert (
        kvarn_native_split_scratch_count(
            1985,
            32,
            "b70_q6",
            page_pair,
        )
        == 32
    )


def test_batch_aware_policy_views_capacity_scratch_contiguously_for_every_batch() -> (
    None
):
    scratch = (
        torch.empty((12, 24 * 32, 256), dtype=torch.float16),
        torch.empty((12, 24, 32), dtype=torch.float32),
        torch.empty((12, 24, 32), dtype=torch.float32),
    )
    expected_by_batch = {
        1: 32,
        2: 16,
        3: 8,
        4: 8,
        5: 4,
        6: 4,
        7: 4,
        8: 4,
        9: 2,
        10: 2,
        11: 2,
        12: 2,
    }

    for batch_size, expected_splits in expected_by_batch.items():
        num_splits = kvarn_native_split_count(
            4096,
            32,
            batch_size=batch_size,
            split_policy="b70_q6",
        )
        assert num_splits == expected_splits

        temp_output, exp_sums, max_logits = _kvarn_native_scratch_views(
            scratch,
            batch_size=batch_size,
            num_query_heads=24,
            num_splits=num_splits,
        )

        assert temp_output.shape == (batch_size, 24 * num_splits, 256)
        assert exp_sums.shape == (batch_size, 24, num_splits)
        assert max_logits.shape == (batch_size, 24, num_splits)
        assert temp_output.is_contiguous()
        assert exp_sums.is_contiguous()
        assert max_logits.is_contiguous()
        assert temp_output.data_ptr() == scratch[0].data_ptr()
        assert exp_sums.data_ptr() == scratch[1].data_ptr()
        assert max_logits.data_ptr() == scratch[2].data_ptr()


def test_b70_q6_split_policy_rejects_ambiguous_or_unsupported_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLIT_POLICY", "b70_q6")
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "32")
    with pytest.raises(ValueError, match="conflicts"):
        kvarn_native_split_policy_requested()

    monkeypatch.delenv("KVARN_NATIVE_XPU_SPLITS")
    with pytest.raises(ValueError, match="batch sizes 1 through 12"):
        kvarn_native_split_count(4096, batch_size=13)

    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLIT_POLICY", "unknown")
    with pytest.raises(ValueError, match="must be one of"):
        kvarn_native_split_policy_requested()


def test_native_output_hadamard_schema_detection_is_backward_compatible() -> None:
    old_op = SimpleNamespace(
        default=SimpleNamespace(
            _schema=SimpleNamespace(arguments=[SimpleNamespace(name="softmax_scale")])
        )
    )
    new_op = SimpleNamespace(
        default=SimpleNamespace(
            _schema=SimpleNamespace(
                arguments=[
                    SimpleNamespace(name="softmax_scale"),
                    SimpleNamespace(name="unrotate_output"),
                ]
            )
        )
    )

    assert not _kvarn_op_supports_argument(old_op, "unrotate_output")
    assert _kvarn_op_supports_argument(new_op, "unrotate_output")
    assert not _kvarn_op_supports_argument(SimpleNamespace(), "unrotate_output")


def test_native_bf16_output_schema_detection_is_backward_compatible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_op = SimpleNamespace(
        default=SimpleNamespace(
            _schema=SimpleNamespace(arguments=[SimpleNamespace(name="unrotate_output")])
        )
    )
    new_op = SimpleNamespace(
        default=SimpleNamespace(
            _schema=SimpleNamespace(
                arguments=[
                    SimpleNamespace(name="unrotate_output"),
                    SimpleNamespace(name="write_bf16_output"),
                ]
            )
        )
    )
    fake_ops = SimpleNamespace(kvarn_decode=old_op, kvarn_decode_with_scratch=new_op)
    monkeypatch.setattr(kvarn_decode.torch.ops, "_vllm_fa2_C", fake_ops)
    kvarn_native_bf16_output_supported.cache_clear()

    assert not kvarn_native_bf16_output_supported(False)
    assert kvarn_native_bf16_output_supported(True)

    kvarn_native_bf16_output_supported.cache_clear()


def test_native_decode_requires_explicit_factory_abi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout_only = SimpleNamespace(
        default=SimpleNamespace(
            _schema=SimpleNamespace(arguments=[SimpleNamespace(name="dpas_layout")])
        )
    )
    factory_abi = SimpleNamespace(
        default=SimpleNamespace(
            _schema=SimpleNamespace(
                arguments=[
                    SimpleNamespace(name="num_kv_splits"),
                    SimpleNamespace(name="kernel_variant"),
                    SimpleNamespace(name="dpas_layout"),
                ]
            )
        )
    )
    fake_ops = SimpleNamespace(
        kvarn_decode=layout_only,
        kvarn_decode_with_scratch=factory_abi,
    )
    monkeypatch.setattr(kvarn_decode.torch.ops, "_vllm_fa2_C", fake_ops)
    kvarn_native_layout_abi_supported.cache_clear()
    kvarn_native_decode_abi_supported.cache_clear()

    assert not kvarn_native_decode_abi_supported(False)
    assert kvarn_native_decode_abi_supported(True)

    kvarn_native_decode_abi_supported.cache_clear()
    kvarn_native_layout_abi_supported.cache_clear()


def test_native_output_hadamard_requires_multisplit_and_matching_op_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[bool] = []

    def supported(with_scratch: bool) -> bool:
        seen.append(with_scratch)
        return with_scratch

    monkeypatch.setattr(
        kvarn_decode, "kvarn_native_output_hadamard_supported", supported
    )
    assert not _kvarn_native_output_hadamard_enabled(1, True)
    assert not seen
    assert _kvarn_native_output_hadamard_enabled(16, True)
    assert not _kvarn_native_output_hadamard_enabled(16, False)
    assert seen == [True, False]


def test_native_xpu_schemas_have_fake_dispatch() -> None:
    # Load the extension first so CPU-only/non-Xe2 builds can skip without
    # importing the rest of the XPU platform adapter.
    pytest.importorskip("vllm_xpu_kernels.flash_attn_interface")

    if not hasattr(torch.ops._vllm_fa2_C, "kvarn_decode"):
        pytest.skip("vllm-xpu-kernels was built without the Xe2 KVarN operators")

    # This adapter installs the guarded fake registrations for native builds.
    import vllm._xpu_ops  # noqa: F401

    operator_names = [
        "kvarn_decode",
        "kvarn_decode_with_scratch",
        "kvarn_materialize_packed_kv",
        "kvarn_dequant",
        "kvarn_hadamard_scatter",
        "kvarn_hadamard_qkv_scatter",
        "kvarn_hadamard",
    ]
    if hasattr(torch.ops._vllm_fa2_C, "kvarn_hadamard_qkv_scatter_current_stream"):
        operator_names.append("kvarn_hadamard_qkv_scatter_current_stream")
    for name in operator_names:
        qualified_name = f"_vllm_fa2_C::{name}"
        assert hasattr(torch.ops._vllm_fa2_C, name)
        assert torch._C._dispatch_has_kernel_for_dispatch_key(qualified_name, "Meta")
        if name != "kvarn_hadamard":
            assert kvarn_native_layout_abi_supported(name)
        if name in ("kvarn_decode", "kvarn_decode_with_scratch"):
            assert kvarn_native_decode_abi_supported(
                name == "kvarn_decode_with_scratch"
            )

    query = torch.empty((1, 24, 256), dtype=torch.float16, device="meta")
    cache = torch.empty((2, 4, 35_072), dtype=torch.uint8, device="meta")
    block_table = torch.empty((1, 2), dtype=torch.int32, device="meta")
    seq_lens = torch.empty((1,), dtype=torch.int32, device="meta")
    cu_seqlens = torch.empty((2,), dtype=torch.int32, device="meta")
    block_to_slot = torch.empty((2,), dtype=torch.int32, device="meta")
    tail_key = torch.empty((2, 128, 4, 256), dtype=torch.float16, device="meta")
    tail_value = torch.empty_like(tail_key)
    output = torch.empty_like(query)

    assert (
        torch.ops._vllm_fa2_C.kvarn_decode(
            query,
            cache,
            block_table,
            seq_lens,
            block_to_slot,
            tail_key,
            tail_value,
            output,
            128,
            0.0625,
            False,
            False,
            1,
            0,
            False,
        )
        is None
    )
    if _kvarn_op_supports_argument(
        torch.ops._vllm_fa2_C.kvarn_decode, "unrotate_output"
    ):
        assert (
            torch.ops._vllm_fa2_C.kvarn_decode(
                query,
                cache,
                block_table,
                seq_lens,
                block_to_slot,
                tail_key,
                tail_value,
                output,
                128,
                0.0625,
                True,
                False,
                1,
                0,
                False,
            )
            is None
        )
    if _kvarn_op_supports_argument(
        torch.ops._vllm_fa2_C.kvarn_decode, "write_bf16_output"
    ):
        bf16_output = torch.empty_like(query, dtype=torch.bfloat16)
        assert (
            torch.ops._vllm_fa2_C.kvarn_decode(
                query,
                cache,
                block_table,
                seq_lens,
                block_to_slot,
                tail_key,
                tail_value,
                bf16_output,
                128,
                0.0625,
                True,
                True,
                1,
                0,
                False,
            )
            is None
        )

    temp_output = torch.empty((1, 24, 256), dtype=torch.float16, device="meta")
    exp_sums = torch.empty((1, 24, 1), dtype=torch.float32, device="meta")
    max_logits = torch.empty_like(exp_sums)
    assert (
        torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
            query,
            cache,
            block_table,
            seq_lens,
            block_to_slot,
            tail_key,
            tail_value,
            temp_output,
            exp_sums,
            max_logits,
            output,
            128,
            0.0625,
            False,
            False,
            1,
            0,
            False,
        )
        is None
    )
    if _kvarn_op_supports_argument(
        torch.ops._vllm_fa2_C.kvarn_decode_with_scratch, "unrotate_output"
    ):
        assert (
            torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
                query,
                cache,
                block_table,
                seq_lens,
                block_to_slot,
                tail_key,
                tail_value,
                temp_output,
                exp_sums,
                max_logits,
                output,
                128,
                0.0625,
                True,
                False,
                1,
                0,
                False,
            )
            is None
        )
    if _kvarn_op_supports_argument(
        torch.ops._vllm_fa2_C.kvarn_decode_with_scratch, "write_bf16_output"
    ):
        bf16_output = torch.empty_like(query, dtype=torch.bfloat16)
        assert (
            torch.ops._vllm_fa2_C.kvarn_decode_with_scratch(
                query,
                cache,
                block_table,
                seq_lens,
                block_to_slot,
                tail_key,
                tail_value,
                temp_output,
                exp_sums,
                max_logits,
                bf16_output,
                128,
                0.0625,
                True,
                True,
                1,
                0,
                False,
            )
            is None
        )

    materialized_key = torch.empty((128, 4, 256), dtype=torch.float16, device="meta")
    materialized_value = torch.empty_like(materialized_key)
    assert (
        torch.ops._vllm_fa2_C.kvarn_materialize_packed_kv(
            cache,
            block_table,
            seq_lens,
            cu_seqlens,
            block_to_slot,
            tail_key,
            tail_value,
            materialized_key,
            materialized_value,
            128,
            False,
        )
        is None
    )

    dequantized_key = torch.empty((2, 4, 256, 128), dtype=torch.float16, device="meta")
    dequantized_value = torch.empty(
        (2, 4, 128, 256), dtype=torch.float16, device="meta"
    )
    assert (
        torch.ops._vllm_fa2_C.kvarn_dequant(
            cache, dequantized_key, dequantized_value, False
        )
        is None
    )

    assert (
        torch.ops._vllm_fa2_C.kvarn_hadamard_scatter(
            query[:, :4],
            query[:, :4],
            torch.empty((1,), dtype=torch.int64, device="meta"),
            block_to_slot,
            tail_key,
            tail_value,
            128,
            False,
        )
        is None
    )
    query_output = torch.empty_like(query)
    assert (
        torch.ops._vllm_fa2_C.kvarn_hadamard_qkv_scatter(
            query,
            query[:, :4],
            query[:, :4],
            torch.empty((1,), dtype=torch.int64, device="meta"),
            block_to_slot,
            query_output,
            tail_key,
            tail_value,
            128,
            False,
        )
        is None
    )
    transformed = torch.empty((24, 256), dtype=torch.float16, device="meta")
    assert (
        torch.ops._vllm_fa2_C.kvarn_hadamard(query.view(24, 256), transformed) is None
    )


@pytest.mark.parametrize(
    (
        "num_prefills",
        "num_decodes",
        "num_spec_decodes",
        "has_non_spec_index",
        "has_spec_mask",
        "num_prefill_tokens",
        "num_decode_tokens",
        "expected",
    ),
    [
        (1, 1, 0, False, False, 1, 1, True),
        (1, 0, 0, False, False, 1, 0, False),
        (0, 1, 0, False, False, 0, 1, False),
        (1, 1, 1, True, True, 1, 1, False),
        (1, 1, 0, True, False, 1, 1, False),
        (1, 1, 0, False, True, 1, 1, False),
        (1, 1, 0, False, False, 1, 2, False),
        (1, 1, 0, False, False, 2, 1, False),
    ],
)
def test_gdn_adapter_splits_only_ordinary_mixed_non_spec(
    monkeypatch: pytest.MonkeyPatch,
    num_prefills: int,
    num_decodes: int,
    num_spec_decodes: int,
    has_non_spec_index: bool,
    has_spec_mask: bool,
    num_prefill_tokens: int,
    num_decode_tokens: int,
    expected: bool,
) -> None:
    pytest.importorskip("vllm_xpu_kernels._xpu_C")

    import vllm._xpu_ops as xpu_ops
    import vllm.forward_context as forward_context
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    num_non_spec = num_prefills + num_decodes
    num_actual_tokens = num_non_spec + num_spec_decodes
    non_spec_index = (
        torch.arange(num_non_spec, dtype=torch.int32) if has_non_spec_index else None
    )
    spec_mask = torch.ones(num_non_spec, dtype=torch.bool) if has_spec_mask else None
    metadata = GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefill_tokens,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
        num_spec_decodes=num_spec_decodes,
        num_spec_decode_tokens=num_spec_decodes,
        num_actual_tokens=num_actual_tokens,
        non_spec_query_start_loc=torch.arange(num_non_spec + 1, dtype=torch.int32),
        non_spec_token_indx=non_spec_index,
        non_spec_state_indices_tensor=torch.arange(num_non_spec, dtype=torch.int32),
        spec_query_start_loc=torch.arange(num_spec_decodes + 1, dtype=torch.int32),
        spec_state_indices_tensor=torch.arange(num_spec_decodes, dtype=torch.int32),
        spec_sequence_masks=spec_mask,
    )
    layer = SimpleNamespace(
        prefix="layer",
        num_k_heads=1,
        num_v_heads=1,
        head_k_dim=1,
        head_v_dim=1,
        conv1d=SimpleNamespace(weight=torch.ones(1, 1, 1), bias=None),
        kv_cache=(torch.empty(1), torch.empty(1)),
        activation="silu",
        A_log=torch.empty(1),
        dt_bias=torch.empty(1),
        tp_size=1,
        gqa_interleaved_layout=False,
    )
    context = SimpleNamespace(
        no_compile_layers={"layer_name": layer},
        attn_metadata={"layer": metadata},
    )
    call = Mock()
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)
    monkeypatch.setattr(torch.ops._xpu_C, "gdn_attention", call)
    output = torch.empty(num_actual_tokens, 1)

    xpu_ops._gdn_attention_core_xpu_impl(
        output,
        torch.empty_like(output),
        torch.empty_like(output),
        torch.empty_like(output),
        "layer_name",
    )

    assert call.call_args.kwargs["split_mixed_non_spec"] is expected


def test_qwen_gdn_xpu_block_size_uses_64_token_kernel_grid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm_xpu_kernels._xpu_C")

    import vllm.config.vllm as config_module
    from vllm.platforms.xpu import XPUPlatform
    from vllm.v1.attention.backends.gdn_attn import QwenGDNAttentionBackend

    layer = SimpleNamespace(get_attn_backend=lambda: QwenGDNAttentionBackend)
    monkeypatch.setattr(
        config_module,
        "get_layers_from_vllm_config",
        lambda *_args, **_kwargs: {"qwen_gdn": layer},
    )
    cache_config = SimpleNamespace(
        block_size=16,
        mamba_cache_mode="none",
        mamba_block_size=None,
        mamba_page_size_padded=16,
    )
    vllm_config = SimpleNamespace(cache_config=cache_config, model_config=object())

    XPUPlatform.update_block_size_for_backend(vllm_config)

    assert cache_config.block_size == 64
    assert cache_config.mamba_page_size_padded == 64


def _request_stable_config():
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_5_text"),
            model=(
                "jasonboukheir/"
                "Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound"
            ),
            revision="6b0622f4354481d5d04577d48ba0db844efc1330",
            architectures=["Qwen3_5ForConditionalGeneration"],
            dtype=torch.bfloat16,
            quantization="compressed-tensors",
            enforce_eager=True,
            multimodal_config=SimpleNamespace(language_model_only=True),
        ),
        cache_config=SimpleNamespace(
            cache_dtype="kvarn_k4v4_g128_compact",
            mamba_cache_mode="none",
            enable_prefix_caching=False,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            enable_dbo=False,
            ubatch_size=1,
        ),
        use_v2_model_runner=False,
        speculative_config=None,
        lora_config=None,
        compilation_config=SimpleNamespace(
            mode=SimpleNamespace(name="NONE"),
            cudagraph_mode=SimpleNamespace(name="NONE"),
            static_forward_context={
                "linear_attn": SimpleNamespace(
                    mamba_type=MambaAttentionBackendEnum.QWEN_GDN_ATTN
                )
            },
        ),
    )


def _enable_xe2_request_stable_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    import vllm.model_executor.determinism.request_stable_linear as request_stable
    from vllm.platforms.xpu import XPUPlatform

    monkeypatch.setattr(
        request_stable, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )
    monkeypatch.setattr(XPUPlatform, "_kvarn_request_stable_xe2_validated", False)
    monkeypatch.setattr(torch.ops._xpu_C, "is_xe2_arch", lambda: True)


def test_xpu_forward_context_is_omitted_when_both_stability_axes_are_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vllm.model_executor.determinism.request_stable_linear as request_stable
    from vllm.platforms.xpu import XPUPlatform

    _enable_xe2_request_stable_profile(monkeypatch)
    monkeypatch.setenv(request_stable.XPU_KVARN_REQUEST_STABLE_PROJECTION_ROWS_ENV, "0")
    monkeypatch.setenv(request_stable.XPU_KVARN_REQUEST_STABLE_RMSNORM_ENV, "0")
    request_stable._get_xpu_kvarn_request_stability_policy.cache_clear()
    try:
        assert (
            XPUPlatform.set_additional_forward_context(
                attn_metadata=None,
                vllm_config=_request_stable_config(),
                num_tokens=1,
            )
            == {}
        )
    finally:
        request_stable._get_xpu_kvarn_request_stability_policy.cache_clear()


def test_xpu_forward_context_exposes_validated_kvarn_request_slices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm_xpu_kernels._xpu_C")

    import vllm.model_executor.determinism.request_stable_linear as request_stable
    from vllm.config import CUDAGraphMode
    from vllm.platforms.xpu import XPUPlatform
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    _enable_xe2_request_stable_profile(monkeypatch)
    metadata = GDNAttentionMetadata(
        num_prefills=2,
        num_prefill_tokens=6,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=7,
        non_spec_query_start_loc_cpu=(0, 1, 3, 7),
        non_spec_num_computed_tokens_cpu=(200, 0, 64),
        non_spec_is_prefilling_cpu=(False, True, True),
    )

    additional = XPUPlatform.set_additional_forward_context(
        attn_metadata={"linear_attn": metadata},
        vllm_config=_request_stable_config(),
        num_tokens=7,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert additional[request_stable.XPU_KVARN_REQUEST_SLICES_KEY] == (
        (0, 1, 200, False),
        (1, 3, 0, True),
        (3, 7, 64, True),
    )


def test_xpu_forward_context_rejects_inconsistent_kvarn_boundaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm_xpu_kernels._xpu_C")

    from vllm.config import CUDAGraphMode
    from vllm.platforms.xpu import XPUPlatform
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    _enable_xe2_request_stable_profile(monkeypatch)
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=3,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=4,
        non_spec_query_start_loc_cpu=(0, 2, 4),
        non_spec_num_computed_tokens_cpu=(200, 0),
        non_spec_is_prefilling_cpu=(False, True),
    )

    with pytest.raises(RuntimeError, match="decode/prefill token counts"):
        XPUPlatform.set_additional_forward_context(
            attn_metadata={"linear_attn": metadata},
            vllm_config=_request_stable_config(),
            num_tokens=4,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )


def test_xpu_forward_context_rejects_misaligned_kvarn_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm_xpu_kernels._xpu_C")

    from vllm.config import CUDAGraphMode
    from vllm.platforms.xpu import XPUPlatform
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    _enable_xe2_request_stable_profile(monkeypatch)
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=3,
        non_spec_query_start_loc_cpu=(0, 1, 3),
        non_spec_num_computed_tokens_cpu=(200, 63),
        non_spec_is_prefilling_cpu=(False, True),
    )

    with pytest.raises(RuntimeError, match="canonical 64-row grid"):
        XPUPlatform.set_additional_forward_context(
            attn_metadata={"linear_attn": metadata},
            vllm_config=_request_stable_config(),
            num_tokens=3,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )


def test_xpu_forward_context_rejects_any_non_gdn_qwen_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm_xpu_kernels._xpu_C")

    from vllm.config import CUDAGraphMode
    from vllm.platforms.xpu import XPUPlatform
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

    _enable_xe2_request_stable_profile(monkeypatch)
    config = _request_stable_config()
    config.compilation_config.static_forward_context["linear_attn_2"] = SimpleNamespace(
        mamba_type=MambaAttentionBackendEnum.QWEN_GDN_ATTN
    )
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=3,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=4,
        non_spec_query_start_loc_cpu=(0, 1, 4),
        non_spec_num_computed_tokens_cpu=(200, 0),
        non_spec_is_prefilling_cpu=(False, True),
    )

    with pytest.raises(RuntimeError, match="non-GDN Qwen metadata"):
        XPUPlatform.set_additional_forward_context(
            attn_metadata={"linear_attn": metadata, "linear_attn_2": object()},
            vllm_config=config,
            num_tokens=4,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )


def test_xpu_forward_context_canonicalizes_post_prompt_multirow_recompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm_xpu_kernels._xpu_C")

    import vllm.model_executor.determinism.request_stable_linear as request_stable
    from vllm.config import CUDAGraphMode
    from vllm.platforms.xpu import XPUPlatform
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    _enable_xe2_request_stable_profile(monkeypatch)
    metadata = GDNAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=2,
        num_decodes=0,
        num_decode_tokens=0,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=2,
        non_spec_query_start_loc_cpu=(0, 2),
        non_spec_num_computed_tokens_cpu=(64,),
        non_spec_is_prefilling_cpu=(False,),
    )

    additional = XPUPlatform.set_additional_forward_context(
        attn_metadata={"linear_attn": metadata},
        vllm_config=_request_stable_config(),
        num_tokens=2,
        cudagraph_runtime_mode=CUDAGraphMode.NONE,
    )

    assert additional[request_stable.XPU_KVARN_REQUEST_SLICES_KEY] == (
        (0, 2, 64, False),
    )


def test_xpu_forward_context_rejects_non_xe2_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("vllm_xpu_kernels._xpu_C")

    import vllm.model_executor.determinism.request_stable_linear as request_stable
    from vllm.config import CUDAGraphMode
    from vllm.platforms.xpu import XPUPlatform

    monkeypatch.setattr(
        request_stable, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )
    monkeypatch.setattr(XPUPlatform, "_kvarn_request_stable_xe2_validated", False)
    monkeypatch.setattr(torch.ops._xpu_C, "is_xe2_arch", lambda: False)

    with pytest.raises(RuntimeError, match="requires an Xe2 device"):
        XPUPlatform.set_additional_forward_context(
            attn_metadata=None,
            vllm_config=_request_stable_config(),
            num_tokens=1,
            cudagraph_runtime_mode=CUDAGraphMode.NONE,
        )
