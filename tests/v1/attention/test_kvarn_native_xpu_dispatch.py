# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.ops.triton_kvarn_decode import (
    kvarn_native_feature_enabled,
    kvarn_native_layer_selected,
    kvarn_native_problem_supported,
    kvarn_native_split_count,
)


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
    assert not kvarn_native_feature_enabled("DECODE")

    monkeypatch.setenv("KVARN_NATIVE_XPU", "1")
    assert kvarn_native_feature_enabled("DECODE")
    assert kvarn_native_feature_enabled("MATERIALIZE")

    monkeypatch.setenv("KVARN_NATIVE_XPU_DECODE", "0")
    assert not kvarn_native_feature_enabled("DECODE")
    assert kvarn_native_feature_enabled("MATERIALIZE")


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


def test_native_problem_rejects_unported_dpas_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU_DPAS_LAYOUT", "1")
    assert not kvarn_native_problem_supported(**_native_problem())


def test_native_layer_filter_matches_components() -> None:
    name = "model.layers.17.self_attn.attn"
    assert kvarn_native_layer_selected(name, "")
    assert kvarn_native_layer_selected(name, "layers.17")
    assert kvarn_native_layer_selected(name, "layers.3,layers.17.self_attn")
    assert not kvarn_native_layer_selected(name, "layers.1")
    assert not kvarn_native_layer_selected(name, "layers.170")


def test_native_split_count_matches_cpp_short_context_rule(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "32")
    assert kvarn_native_split_count(4096) == 32
    assert kvarn_native_split_count(1024) == 1

    monkeypatch.setenv("KVARN_NATIVE_XPU_SPLITS", "3")
    with pytest.raises(ValueError, match="must be one of"):
        kvarn_native_split_count(4096)


def test_native_xpu_schemas_have_fake_dispatch() -> None:
    # Load the extension first so CPU-only/non-Xe2 builds can skip without
    # importing the rest of the XPU platform adapter.
    pytest.importorskip("vllm_xpu_kernels.flash_attn_interface")

    if not hasattr(torch.ops._vllm_fa2_C, "kvarn_decode"):
        pytest.skip("vllm-xpu-kernels was built without the Xe2 KVarN operators")

    # This adapter installs the guarded fake registrations for native builds.
    import vllm._xpu_ops  # noqa: F401

    operator_names = (
        "kvarn_decode",
        "kvarn_decode_with_scratch",
        "kvarn_materialize_packed_kv",
    )
    for name in operator_names:
        qualified_name = f"_vllm_fa2_C::{name}"
        assert hasattr(torch.ops._vllm_fa2_C, name)
        assert torch._C._dispatch_has_kernel_for_dispatch_key(qualified_name, "Meta")

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
        )
        is None
    )
