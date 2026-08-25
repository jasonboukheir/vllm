# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for Qwen3.5 text-only MTP speculative decoding."""

from unittest.mock import MagicMock, patch

import pytest
from transformers import PretrainedConfig

from vllm.config.speculative import SpeculativeConfig


def _mtp_config(model_type: str) -> PretrainedConfig:
    return PretrainedConfig(
        model_type=model_type,
        architectures=["SomeArch"],
        mtp_num_hidden_layers=1,
    )


@pytest.mark.parametrize(
    "model_type,expected_arch",
    [
        ("qwen3_5", "Qwen3_5MTP"),
        ("qwen3_5_moe", "Qwen3_5MoeMTP"),
        # Text-only config variants must map to the same MTP architectures.
        ("qwen3_5_text", "Qwen3_5MTP"),
        ("qwen3_5_moe_text", "Qwen3_5MoeMTP"),
    ],
)
def test_mtp_override_recognizes_text_only_types(model_type, expected_arch):
    cfg = SpeculativeConfig.hf_config_override(_mtp_config(model_type))
    assert cfg.model_type == "qwen3_5_mtp"
    assert cfg.architectures == [expected_arch]
    assert cfg.n_predict == 1


@pytest.mark.parametrize(
    (
        "draft_revision",
        "draft_code_revision",
        "expected_revision",
        "expected_code_revision",
    ),
    [
        (None, None, "a" * 40, "b" * 40),
        ("c" * 40, "d" * 40, "c" * 40, "d" * 40),
    ],
)
def test_embedded_mtp_preserves_revisions(
    draft_revision,
    draft_code_revision,
    expected_revision,
    expected_code_revision,
):
    target = MagicMock()
    target.model = "org/model"
    target.revision = "a" * 40
    target.code_revision = "b" * 40
    target.hf_text_config.model_type = "qwen3_5"

    with (
        patch(
            "vllm.config.speculative.ModelConfig",
            side_effect=RuntimeError("stop after capturing draft arguments"),
        ) as model,
        pytest.raises(RuntimeError, match="capturing draft arguments"),
    ):
        SpeculativeConfig(
            method="mtp",
            num_speculative_tokens=1,
            revision=draft_revision,
            code_revision=draft_code_revision,
            target_model_config=target,
            target_parallel_config=MagicMock(),
        )

    assert model.call_args.kwargs["revision"] == expected_revision
    assert model.call_args.kwargs["code_revision"] == expected_code_revision
