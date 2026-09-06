# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.config import CUDAGraphMode
from vllm.platforms.xpu import _check_kvarn_beta_unsupported_config


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
        use_v2_model_runner=use_v2,
        compilation_config=SimpleNamespace(
            cudagraph_mode=(CUDAGraphMode.FULL if graph else CUDAGraphMode.NONE)
        ),
        model_config=SimpleNamespace(
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


def test_kvarn_beta_accepts_supported_eager_text_configuration() -> None:
    _check_kvarn_beta_unsupported_config(_config(), CUDAGraphMode.NONE)


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
