# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


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
