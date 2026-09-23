# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The draft must honour ``attention_backend`` from --speculative-config.

``init_attn_backend`` reads the backend off each constructed layer, so these
assert on the config ``load_eagle_model`` hands to ``get_model``.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config import LoadConfig
from vllm.v1.worker.gpu.spec_decode.eagle.utils import load_eagle_model


@dataclass
class _AttentionConfig:
    backend: str | None = None


@dataclass
class _KernelConfig:
    moe_backend: str | None = None


@dataclass
class _CacheConfig:
    cache_dtype: str = "auto"


@dataclass
class _SpeculativeConfig:
    attention_backend: str | None = None
    moe_backend: str | None = None
    kv_cache_dtype: str | None = None
    draft_model_config: object = None


@dataclass
class _VllmConfig:
    attention_config: _AttentionConfig
    kernel_config: _KernelConfig
    cache_config: _CacheConfig
    speculative_config: _SpeculativeConfig
    load_config: LoadConfig = field(default_factory=LoadConfig)


def _config(target_backend: str, draft_backend: str | None) -> _VllmConfig:
    return _VllmConfig(
        attention_config=_AttentionConfig(backend=target_backend),
        kernel_config=_KernelConfig(),
        cache_config=_CacheConfig(),
        speculative_config=_SpeculativeConfig(attention_backend=draft_backend),
    )


class _Captured(Exception):
    def __init__(self, vllm_config):
        self.vllm_config = vllm_config


def _capture_draft_config(cfg):
    def _fake_get_model(*, vllm_config, model_config):
        raise _Captured(vllm_config)

    with (
        patch("vllm.v1.worker.gpu.spec_decode.eagle.utils.get_model", _fake_get_model),
        patch(
            "vllm.v1.worker.gpu.spec_decode.utils.get_pp_group",
            return_value=SimpleNamespace(world_size=1),
        ),
        pytest.raises(_Captured) as exc,
    ):
        load_eagle_model(object(), cfg)
    return exc.value.vllm_config


def test_draft_attention_backend_overrides_the_target():
    used = _capture_draft_config(_config("FLASHINFER", "TRITON_ATTN"))
    assert used.attention_config.backend == "TRITON_ATTN"


def test_unset_leaves_the_target_backend_in_place():
    """Clearing it lets the draft autoselect a KV layout the target lacks."""
    cfg = _config("FLEX_ATTENTION", None)
    used = _capture_draft_config(cfg)
    assert used is cfg
    assert used.attention_config.backend == "FLEX_ATTENTION"


def test_override_does_not_mutate_the_target_config():
    cfg = _config("FLASHINFER", "TRITON_ATTN")
    _capture_draft_config(cfg)
    assert cfg.attention_config.backend == "FLASHINFER"


@pytest.mark.parametrize(
    "draft_dtype",
    ["kvarn_k4v2_g128_compact", "kvarn_k4v4_g128_compact", "bfloat16"],
)
def test_v1_mtp_draft_precision_keeps_target_config_and_selects_own_backend(
    draft_dtype,
):
    """A BF16 draft must not inherit the target's packed KVarN reader."""
    import torch

    from vllm.platforms.xpu import XPUPlatform
    from vllm.v1.spec_decode.llm_base_proposer import SpecDecodeBaseProposer

    cfg = _config("KVARN", None)
    cfg.cache_config.cache_dtype = "kvarn_k4v2_g128_compact"
    cfg.speculative_config.kv_cache_dtype = draft_dtype
    proposer = SimpleNamespace(
        vllm_config=cfg, speculative_config=cfg.speculative_config
    )
    used = SpecDecodeBaseProposer._create_draft_vllm_config(proposer)
    assert cfg.cache_config.cache_dtype == "kvarn_k4v2_g128_compact"
    assert cfg.attention_config.backend == "KVARN"
    assert used.cache_config.cache_dtype == draft_dtype
    assert used.attention_config.backend is None
    selector = SimpleNamespace(
        kv_cache_dtype=used.cache_config.cache_dtype,
        dtype=torch.bfloat16,
        use_mla=False,
        use_sparse=False,
        use_mm_prefix=False,
    )
    backend = XPUPlatform.get_attn_backend_cls(None, selector)
    expected = (
        "FlashAttentionBackend"
        if draft_dtype == "bfloat16"
        else "KVarNAttentionBackend"
    )
    assert backend.rsplit(".", 1)[-1] == expected
