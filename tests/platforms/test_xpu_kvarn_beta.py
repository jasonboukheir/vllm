# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.config import CUDAGraphMode
from vllm.platforms.xpu import (
    _RETIRED_KVARN_ENV_VARS,
    _check_kvarn_beta_unsupported_config,
)


def _config(
    *,
    cache_dtype: str = "kvarn_k4v4_g128_compact",
    speculative=False,
    use_v2=False,
    graph=False,
    prefix_caching=False,
    multimodal=False,
    language_model_only=False,
    model_type=None,
    video_limit=0,
):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            cache_dtype=cache_dtype,
            enable_prefix_caching=prefix_caching,
        ),
        speculative_config=SimpleNamespace() if speculative else None,
        scheduler_config=SimpleNamespace(max_num_seqs=1, max_num_batched_tokens=2048),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1, pipeline_parallel_size=1
        ),
        use_v2_model_runner=use_v2,
        compilation_config=SimpleNamespace(
            cudagraph_mode=(CUDAGraphMode.FULL if graph else CUDAGraphMode.NONE)
        ),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            max_model_len=8192,
            is_multimodal_model=multimodal,
            hf_config=SimpleNamespace(model_type=model_type),
            multimodal_config=(
                SimpleNamespace(
                    language_model_only=language_model_only,
                    get_limit_per_prompt=lambda modality: video_limit,
                )
                if multimodal
                else None
            ),
        ),
    )


def _mtp_config():
    config = _config(multimodal=True, model_type="qwen3_5")
    config.speculative_config = SimpleNamespace(
        method="mtp", num_speculative_tokens=1, kv_cache_dtype=None
    )
    return config


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("speculative_config", "method", "draft_model"),
        ("speculative_config", "num_speculative_tokens", 0),
        ("speculative_config", "num_speculative_tokens", 3),
        ("speculative_config", "kv_cache_dtype", "auto"),
        ("scheduler_config", "max_num_batched_tokens", 4096),
        ("model_config", "dtype", torch.float16),
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("cache_config", "cache_dtype", "kvarn_k4v2_g128_compact"),
    ],
)
def test_kvarn_mtp_rejects_unqualified_envelope(section, field, value):
    config = _mtp_config()
    setattr(getattr(config, section), field, value)
    with pytest.raises(ValueError, match="speculative decoding/MTP"):
        _check_kvarn_beta_unsupported_config(config, CUDAGraphMode.NONE)


@pytest.mark.parametrize("max_num_seqs", [1, 4, 16])
@pytest.mark.parametrize("num_speculative_tokens", [1, 2])
def test_kvarn_mtp_accepts_scheduler_concurrency(max_num_seqs, num_speculative_tokens):
    config = _mtp_config()
    config.scheduler_config.max_num_seqs = max_num_seqs
    config.speculative_config.num_speculative_tokens = num_speculative_tokens
    _check_kvarn_beta_unsupported_config(config, CUDAGraphMode.NONE)


def test_kvarn_beta_accepts_supported_eager_text_configuration() -> None:
    _check_kvarn_beta_unsupported_config(_config(), CUDAGraphMode.NONE)


@pytest.mark.parametrize(
    "name", (*_RETIRED_KVARN_ENV_VARS, "KVARN_NATIVE_XPU_KERNEL_VARIANT")
)
def test_kvarn_release_rejects_retired_server_overrides(monkeypatch, name):
    monkeypatch.setenv(name, "1")
    with pytest.raises(ValueError, match="retired KVarN experiment"):
        _check_kvarn_beta_unsupported_config(_config(), CUDAGraphMode.NONE)


def test_auto_is_not_affected_by_retired_kvarn_overrides(monkeypatch):
    monkeypatch.setenv("KVARN_NATIVE_XPU_KERNEL_VARIANT", "baseline")
    _check_kvarn_beta_unsupported_config(
        _config(cache_dtype="auto"), CUDAGraphMode.NONE
    )


def test_kvarn_beta_accepts_multimodal_checkpoint_in_language_only_mode() -> None:
    _check_kvarn_beta_unsupported_config(
        _config(multimodal=True, language_model_only=True), CUDAGraphMode.NONE
    )


def test_kvarn_beta_accepts_qwen35_images_with_video_disabled() -> None:
    _check_kvarn_beta_unsupported_config(
        _config(multimodal=True, model_type="qwen3_5"), CUDAGraphMode.NONE
    )


def test_kvarn_beta_does_not_enable_unvalidated_video() -> None:
    with pytest.raises(ValueError, match="video inputs disabled"):
        _check_kvarn_beta_unsupported_config(
            _config(multimodal=True, model_type="qwen3_5", video_limit=1),
            CUDAGraphMode.NONE,
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"speculative": True}, "speculative decoding/MTP"),
        ({"use_v2": True}, "requires Model Runner V1"),
        ({"graph": True}, "graph mode"),
        ({"prefix_caching": True}, "prefix caching"),
        ({"multimodal": True}, "vision/multimodal"),
    ],
)
def test_kvarn_beta_rejects_unsupported_configuration(override, message) -> None:
    with pytest.raises(ValueError, match=message):
        _check_kvarn_beta_unsupported_config(_config(**override), CUDAGraphMode.NONE)


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
def test_non_kvarn_configuration_is_unchanged(cache_dtype: str) -> None:
    _check_kvarn_beta_unsupported_config(
        _config(
            cache_dtype=cache_dtype,
            speculative=True,
            use_v2=True,
            graph=True,
            prefix_caching=True,
            multimodal=True,
        ),
        CUDAGraphMode.NONE,
    )
