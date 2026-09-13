# SPDX-License-Identifier: Apache-2.0
"""Explicit unquantized cache settings must reach XPU's input-matching writer."""

import pytest
import torch

from vllm import _custom_ops as ops


@pytest.mark.parametrize("dtype_name", ["float16", "bfloat16"])
def test_explicit_unquantized_cache_preserves_values_and_sparse_slots(dtype_name):
    if not torch.xpu.is_available():
        pytest.skip("requires XPU")
    dtype = getattr(torch, dtype_name)
    key = torch.arange(6 * 4 * 256, device="xpu").reshape(6, 4, 256).to(dtype)
    value = -key
    cache_k = torch.full((3, 128, 4, 256), float("nan"), dtype=dtype, device="xpu")
    cache_v = torch.full_like(cache_k, float("nan"))
    slots_cpu = torch.tensor([0, 127, 128, 383, 256, -1], dtype=torch.int64)
    slots = slots_cpu.xpu()
    scale = torch.ones(1, device="xpu")
    ops.reshape_and_cache_flash(
        key, value, cache_k, cache_v, slots, dtype_name, scale, scale
    )
    expected_k = torch.full_like(cache_k, float("nan"), device="cpu")
    expected_v = torch.full_like(expected_k, float("nan"))
    expected_k.view(-1, 4, 256)[slots_cpu[:-1]] = key.cpu()[:-1]
    expected_v.view(-1, 4, 256)[slots_cpu[:-1]] = value.cpu()[:-1]
    torch.testing.assert_close(
        cache_k.cpu(), expected_k, atol=0, rtol=0, equal_nan=True
    )
    torch.testing.assert_close(
        cache_v.cpu(), expected_v, atol=0, rtol=0, equal_nan=True
    )
    wrong = cache_v.to(torch.float32)
    with pytest.raises(ValueError, match="matching input and cache dtypes"):
        ops.reshape_and_cache_flash(
            key, value, cache_k, wrong, slots, dtype_name, scale, scale
        )
