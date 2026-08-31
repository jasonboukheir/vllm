# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-stable linear dispatch for the scoped XPU KVarN profile."""

from typing import Any

import torch

from vllm.forward_context import (
    get_forward_context,
    is_forward_context_available,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

XPU_KVARN_REQUEST_SLICES_KEY = "xpu_kvarn_request_slices"
XPU_KVARN_CANONICAL_LINEAR_ROWS = 64
_QWEN_GDN_CANONICAL_LINEAR_SUFFIXES = (
    ".linear_attn.in_proj_qkvz",
    ".linear_attn.in_proj_ba",
    ".linear_attn.out_proj",
)
_BRUTUS_MODEL_ID = (
    "jasonboukheir/"
    "Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound"
)
_BRUTUS_MODEL_REVISION = "6b0622f4354481d5d04577d48ba0db844efc1330"


def use_xpu_kvarn_request_stable_linears(vllm_config: Any) -> bool:
    """Return whether the frozen eager Qwen/XPU profile needs stable GEMM rows.

    This is intentionally narrower than a platform-wide batch-invariance claim.
    Unsupported execution modes retain their existing linear dispatch.
    """
    if not current_platform.is_xpu():
        return False

    model_config = getattr(vllm_config, "model_config", None)
    cache_config = getattr(vllm_config, "cache_config", None)
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if model_config is None or cache_config is None or parallel_config is None:
        return False

    if getattr(cache_config, "cache_dtype", None) != "kvarn_k4v4_g128_compact":
        return False
    if getattr(cache_config, "mamba_cache_mode", None) != "none":
        return False
    if getattr(cache_config, "enable_prefix_caching", True):
        return False

    hf_text_config = getattr(model_config, "hf_text_config", None)
    if getattr(hf_text_config, "model_type", None) != "qwen3_5_text":
        return False
    if getattr(model_config, "model", None) != _BRUTUS_MODEL_ID:
        return False
    if getattr(model_config, "revision", None) != _BRUTUS_MODEL_REVISION:
        return False
    if "Qwen3_5ForConditionalGeneration" not in getattr(
        model_config, "architectures", ()
    ):
        return False
    if getattr(model_config, "dtype", None) != torch.bfloat16:
        return False
    if getattr(model_config, "quantization", None) != "compressed-tensors":
        return False
    if not getattr(model_config, "enforce_eager", False):
        return False
    if getattr(vllm_config, "use_v2_model_runner", True):
        return False
    multimodal_config = getattr(model_config, "multimodal_config", None)
    if multimodal_config is not None and not getattr(
        multimodal_config, "language_model_only", False
    ):
        return False

    parallel_fields = (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "decode_context_parallel_size",
        "prefill_context_parallel_size",
    )
    if any(getattr(parallel_config, field, 1) != 1 for field in parallel_fields):
        return False
    if getattr(parallel_config, "enable_dbo", True):
        return False
    if getattr(parallel_config, "ubatch_size", 2) != 1:
        return False
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if (
        getattr(getattr(compilation_config, "mode", None), "name", None) != "NONE"
        or getattr(
            getattr(compilation_config, "cudagraph_mode", None), "name", None
        )
        != "NONE"
    ):
        return False
    if getattr(vllm_config, "speculative_config", None) is not None:
        return False
    return getattr(vllm_config, "lora_config", None) is None


def _validate_request_slices(
    value: Any,
) -> tuple[tuple[int, int, int, bool], ...]:
    if not isinstance(value, (tuple, list)) or not value:
        raise ValueError("XPU KVarN request slices must be a non-empty sequence")

    result: list[tuple[int, int, int, bool]] = []
    expected_start = 0
    for entry in value:
        if not isinstance(entry, (tuple, list)) or len(entry) != 4:
            raise ValueError(
                "each XPU KVarN request slice must be a "
                "(start, stop, position, is_prefill) tuple"
            )
        start, stop, position, is_prefill = entry
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(stop, int)
            or isinstance(stop, bool)
        ):
            raise ValueError("XPU KVarN request slice bounds must be integers")
        if not isinstance(is_prefill, bool):
            raise ValueError("XPU KVarN request slice phase must be boolean")
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise ValueError(
                "XPU KVarN request slice position must be a non-negative integer"
            )
        if start != expected_start or stop <= start:
            raise ValueError(
                "XPU KVarN request slices must be positive and contiguous from row 0"
            )
        result.append((start, stop, position, is_prefill))
        expected_start = stop
    return tuple(result)


def get_xpu_kvarn_request_slices() -> (
    tuple[tuple[int, int, int, bool], ...] | None
):
    if not is_forward_context_available():
        return None
    value = get_forward_context().additional_kwargs.get(
        XPU_KVARN_REQUEST_SLICES_KEY
    )
    if value is None:
        return None
    return _validate_request_slices(value)


def _uses_canonical_linear_rows(layer: Any) -> bool:
    kernel = getattr(getattr(layer, "scheme", None), "kernel", None)
    if bool(getattr(layer, "xpu_kvarn_request_stable_m64", False)) or bool(
        getattr(kernel, "xpu_kvarn_request_stable_m64", False)
    ):
        return True

    kernel_type = type(kernel)
    kernel_config = getattr(kernel, "config", None)
    if (
        kernel_type.__module__
        == "vllm.model_executor.kernels.linear.mixed_precision.xpu"
        and kernel_type.__name__ == "XPUwNa16LinearKernel"
        and getattr(kernel_config, "group_size", None) == 128
        and getattr(kernel_config, "act_type", None) == torch.bfloat16
        and getattr(kernel_config, "weight_type", None)
        in (scalar_types.uint4, scalar_types.uint4b8)
    ):
        return True

    quant_method_type = type(layer.quant_method)
    if (
        quant_method_type.__module__
        == (
            "vllm.model_executor.layers.quantization.compressed_tensors."
            "compressed_tensors"
        )
        and quant_method_type.__name__ == "CompressedTensorsLinearMethod"
    ):
        raise RuntimeError(
            "the frozen XPU KVarN profile requires every compressed linear to "
            "use the BF16 XPU W4A16 group-128 kernel"
        )

    return (
        quant_method_type.__module__ == "vllm.model_executor.layers.linear"
        and quant_method_type.__name__ == "UnquantizedLinearMethod"
        and getattr(layer, "prefix", "").endswith(
            _QWEN_GDN_CANONICAL_LINEAR_SUFFIXES
        )
    )


def apply_linear_by_request(
    layer: Any,
    x: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Apply one linear method call per packed request when explicitly enabled."""
    request_slices = get_xpu_kvarn_request_slices()
    if request_slices is None:
        return layer.quant_method.apply(layer, x, bias)
    if x.ndim != 2:
        raise RuntimeError(
            "XPU KVarN request-stable linears require packed two-dimensional input"
        )
    if request_slices[-1][1] != x.shape[0]:
        raise RuntimeError(
            "XPU KVarN request slices do not cover this packed linear input"
        )

    output: torch.Tensor | None = None
    expected_tail: tuple[int, ...] | None = None
    expected_dtype: torch.dtype | None = None
    expected_device: torch.device | None = None
    canonical_m64 = _uses_canonical_linear_rows(layer)
    for request_start, request_stop, position, is_prefill in request_slices:
        canonical_request = canonical_m64 and (
            is_prefill or request_stop - request_start > 1
        )
        start = request_start
        while start < request_stop:
            lane = position % XPU_KVARN_CANONICAL_LINEAR_ROWS
            if canonical_request:
                stop = min(
                    start + XPU_KVARN_CANONICAL_LINEAR_ROWS - lane,
                    request_stop,
                )
            else:
                stop = request_stop
            part_input = x[start:stop]
            actual_rows = stop - start
            if canonical_request and (
                lane != 0 or actual_rows < XPU_KVARN_CANONICAL_LINEAR_ROWS
            ):
                padded_input = x.new_zeros(
                    (XPU_KVARN_CANONICAL_LINEAR_ROWS, x.shape[1])
                )
                padded_input[lane : lane + actual_rows].copy_(part_input)
                part_input = padded_input
            part = layer.quant_method.apply(layer, part_input, bias)
            if part.ndim < 1 or part.shape[0] != part_input.shape[0]:
                raise RuntimeError(
                    "request-stable linear method changed the packed row dimension"
                )
            if part.shape[0] != actual_rows:
                part = part[lane : lane + actual_rows]
            if output is None:
                output = part.new_empty((x.shape[0], *part.shape[1:]))
                expected_tail = part.shape[1:]
                expected_dtype = part.dtype
                expected_device = part.device
            elif (
                part.shape[1:] != expected_tail
                or part.dtype != expected_dtype
                or part.device != expected_device
            ):
                raise RuntimeError(
                    "request-stable linear method changed output shape, dtype, "
                    "or device"
                )
            output[start:stop].copy_(part)
            start = stop
            position += actual_rows
    assert output is not None
    return output
