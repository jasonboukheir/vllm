# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.determinism.request_stable_linear as request_stable
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor


@pytest.fixture(autouse=True)
def _reset_request_stability_policy_cache():
    request_stable._get_xpu_kvarn_request_stability_policy.cache_clear()
    yield
    request_stable._get_xpu_kvarn_request_stability_policy.cache_clear()


class RecordingQuantMethod:
    def __init__(self) -> None:
        self.calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []

    def apply(self, _layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        self.calls.append((x, bias))
        # Deliberately shape-sensitive so request-wise and batch-wise dispatch
        # have observably different results on CPU.
        return x + x.shape[0]


class LaneSensitiveQuantMethod(RecordingQuantMethod):
    def apply(self, _layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        self.calls.append((x, bias))
        lanes = torch.arange(x.shape[0], dtype=x.dtype).reshape(-1, 1)
        return x + lanes


def _set_request_slices(monkeypatch: pytest.MonkeyPatch, value) -> None:
    context = SimpleNamespace(
        additional_kwargs={request_stable.XPU_KVARN_REQUEST_SLICES_KEY: value}
    )
    monkeypatch.setattr(request_stable, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(request_stable, "get_forward_context", lambda: context)
    monkeypatch.setattr(
        request_stable, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )


def _production_xpu_w4_layer(monkeypatch: pytest.MonkeyPatch, apply):
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.xpu import (
        XPUwNa16LinearKernel,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as compressed_tensors_module,
    )

    kernel = XPUwNa16LinearKernel.__new__(XPUwNa16LinearKernel)
    kernel.config = MPLinearLayerConfig(
        full_weight_shape=(1, 1),
        partition_weight_shape=(1, 1),
        weight_type=request_stable.scalar_types.uint4b8,
        act_type=torch.bfloat16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )
    method = compressed_tensors_module.CompressedTensorsLinearMethod(SimpleNamespace())
    monkeypatch.setattr(method, "apply", apply)
    return SimpleNamespace(
        quant_method=method,
        scheme=SimpleNamespace(kernel=kernel),
        prefix="model.layers.0.self_attn.q_proj",
    )


def test_apply_linear_by_request_preserves_order_and_bias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(
        monkeypatch,
        ((0, 1, 200, False), (1, 4, 0, True), (4, 8, 64, True)),
    )
    method = RecordingQuantMethod()
    layer = SimpleNamespace(quant_method=method)
    x = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    bias = torch.arange(3, dtype=torch.float32)

    actual = request_stable.apply_linear_by_request(layer, x, bias)

    expected = torch.cat((x[:1] + 1, x[1:4] + 3, x[4:] + 4))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [call_x.shape[0] for call_x, _bias in method.calls] == [1, 3, 4]
    assert all(call_bias is bias for _call_x, call_bias in method.calls)


def test_apply_linear_by_request_is_invariant_to_scheduler_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    x = torch.arange(4095, dtype=torch.float32).reshape(4095, 1)

    def run(value: torch.Tensor, slices) -> tuple[torch.Tensor, list[int]]:
        _set_request_slices(monkeypatch, slices)
        method = RecordingQuantMethod()
        kernel = SimpleNamespace(xpu_kvarn_request_stable_m64=True)
        actual = request_stable.apply_linear_by_request(
            SimpleNamespace(
                quant_method=method,
                scheme=SimpleNamespace(kernel=kernel),
            ),
            value,
            None,
        )
        return actual, [part.shape[0] for part, _bias in method.calls]

    one_shot, one_shot_calls = run(x, ((0, 4095, 0, True),))
    first, first_calls = run(x[:3968], ((0, 3968, 0, True),))
    second, second_calls = run(x[3968:], ((0, 127, 3968, True),))

    torch.testing.assert_close(one_shot, torch.cat((first, second)), rtol=0, atol=0)
    assert one_shot_calls == first_calls + second_calls
    assert one_shot_calls == [64] * 64


def test_canonical_prefill_65023_uses_1016_fixed_row_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_rows = 65_023
    _set_request_slices(monkeypatch, ((0, num_rows, 0, True),))
    method = RecordingQuantMethod()
    layer = SimpleNamespace(
        quant_method=method,
        xpu_kvarn_request_stable_m64=True,
    )
    x = torch.arange(num_rows, dtype=torch.float32).reshape(-1, 1)

    actual = request_stable.apply_linear_by_request(layer, x, None)

    torch.testing.assert_close(actual, x + 64, rtol=0, atol=0)
    assert len(method.calls) == 1_016
    assert all(part.shape == (64, 1) for part, _bias in method.calls)


@pytest.mark.parametrize(
    "x,slices",
    [
        (torch.ones(8, 3), None),
        (torch.ones(8, 3), ((0, 8, 0, True),)),
    ],
)
def test_apply_linear_by_request_falls_back_to_one_call(
    monkeypatch: pytest.MonkeyPatch,
    x: torch.Tensor,
    slices,
) -> None:
    if slices is None:
        monkeypatch.setattr(
            request_stable, "is_forward_context_available", lambda: False
        )
    else:
        _set_request_slices(monkeypatch, slices)
    method = RecordingQuantMethod()
    layer = SimpleNamespace(quant_method=method)

    request_stable.apply_linear_by_request(layer, x, None)

    assert len(method.calls) == 1
    if slices is None:
        assert method.calls[0][0] is x
    else:
        torch.testing.assert_close(method.calls[0][0], x, rtol=0, atol=0)


def test_apply_linear_b1_returns_validated_method_result_directly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(monkeypatch, ((0, 1, 200, False),))
    x = torch.arange(3, dtype=torch.float32).reshape(1, 3)
    bias = torch.arange(3, dtype=torch.float32)
    expected = x * 2 + bias
    calls = []

    class ReturningMethod:
        def apply(self, layer, value, actual_bias):
            calls.append((layer, value, actual_bias))
            return expected

    layer = SimpleNamespace(quant_method=ReturningMethod())

    actual = request_stable.apply_linear_by_request(layer, x, bias)

    assert actual is expected
    assert len(calls) == 1
    assert calls[0][0] is layer
    assert calls[0][1] is x
    assert calls[0][2] is bias
    torch.testing.assert_close(actual, x * 2 + bias, rtol=0, atol=0)


def test_apply_linear_b1_skips_kernel_classification(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(monkeypatch, ((0, 1, 200, False),))
    monkeypatch.setattr(
        request_stable,
        "_is_production_xpu_w4a16_linear",
        lambda _layer: pytest.fail("B1 decode must not classify the linear kernel"),
    )
    x = torch.ones(1, 3)
    method = RecordingQuantMethod()

    actual = request_stable.apply_linear_by_request(
        SimpleNamespace(quant_method=method), x, None
    )

    assert len(method.calls) == 1
    torch.testing.assert_close(actual, x + 1)


def test_validated_request_slices_skip_revalidation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    slices = request_stable.XPUKvarnRequestSlices(((0, 1, 200, False),))
    _set_request_slices(monkeypatch, slices)
    monkeypatch.setattr(
        request_stable,
        "_validate_request_slices",
        lambda _value: pytest.fail("trusted slices must not be revalidated"),
    )

    assert request_stable.get_xpu_kvarn_request_slices() is slices


@pytest.mark.parametrize(
    "invalid",
    [torch.tensor(1.0), torch.ones(2, 3)],
)
def test_apply_linear_b1_fast_path_validates_underlying_row_count(
    monkeypatch: pytest.MonkeyPatch,
    invalid: torch.Tensor,
) -> None:
    _set_request_slices(monkeypatch, ((0, 1, 200, False),))
    layer = SimpleNamespace(
        quant_method=SimpleNamespace(apply=lambda _layer, _x, _bias: invalid)
    )

    with pytest.raises(RuntimeError, match="changed the packed row dimension"):
        request_stable.apply_linear_by_request(layer, torch.ones(1, 3), None)


def test_apply_linear_by_request_rejects_active_non_packed_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(monkeypatch, ((0, 2, 0, True),))
    layer = SimpleNamespace(quant_method=RecordingQuantMethod())

    with pytest.raises(RuntimeError, match="two-dimensional"):
        request_stable.apply_linear_by_request(layer, torch.ones(2, 4, 3), None)


def test_canonical_w4_prefill_uses_absolute_lane(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(monkeypatch, ((0, 2, 62, True),))
    method = LaneSensitiveQuantMethod()
    layer = SimpleNamespace(
        quant_method=method,
        scheme=SimpleNamespace(
            kernel=SimpleNamespace(xpu_kvarn_request_stable_m64=True)
        ),
    )
    x = torch.tensor([[10.0], [20.0]])

    actual = request_stable.apply_linear_by_request(layer, x, None)

    torch.testing.assert_close(
        actual,
        x + torch.tensor([[62.0], [63.0]]),
        rtol=0,
        atol=0,
    )
    padded = method.calls[0][0]
    assert padded.shape == (64, 1)
    torch.testing.assert_close(padded[62:64], x, rtol=0, atol=0)
    assert torch.count_nonzero(padded[:62]) == 0


def test_production_xpu_w4_classifier_uses_canonical_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.model_executor.kernels.linear.mixed_precision.xpu import (
        XPUwNa16LinearKernel,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as compressed_tensors_module,
    )
    from vllm.scalar_type import scalar_types

    _set_request_slices(monkeypatch, ((0, 63, 0, True),))
    kernel = XPUwNa16LinearKernel.__new__(XPUwNa16LinearKernel)
    kernel.config = MPLinearLayerConfig(
        full_weight_shape=(1, 1),
        partition_weight_shape=(1, 1),
        weight_type=scalar_types.uint4b8,
        act_type=torch.bfloat16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )
    calls: list[torch.Tensor] = []

    def apply_weights(_layer, value: torch.Tensor, bias=None) -> torch.Tensor:
        calls.append(value)
        lanes = torch.arange(value.shape[0], dtype=value.dtype).reshape(-1, 1)
        return value + lanes

    scheme = SimpleNamespace(kernel=kernel, apply_weights=apply_weights)
    layer = SimpleNamespace(
        quant_method=compressed_tensors_module.CompressedTensorsLinearMethod(
            SimpleNamespace()
        ),
        scheme=scheme,
        prefix="model.layers.0.self_attn.q_proj",
    )
    x = torch.arange(63, dtype=torch.bfloat16).reshape(-1, 1)

    actual = request_stable.apply_linear_by_request(layer, x, None)

    torch.testing.assert_close(
        actual,
        x + torch.arange(63, dtype=x.dtype).reshape(-1, 1),
        rtol=0,
        atol=0,
    )
    assert [call.shape[0] for call in calls] == [64]


def test_production_xpu_w4_packed_b4_uses_one_direct_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(
        monkeypatch,
        tuple((row, row + 1, 200 + row, False) for row in range(4)),
    )
    calls = []
    results = []

    def apply(_layer, value: torch.Tensor, bias=None) -> torch.Tensor:
        calls.append((value, bias))
        result = value + 7
        results.append(result)
        return result

    layer = _production_xpu_w4_layer(monkeypatch, apply)
    x = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)
    bias = torch.arange(3, dtype=torch.bfloat16)

    actual = request_stable.apply_linear_by_request(layer, x, bias)

    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1] is bias
    assert actual is results[0]
    torch.testing.assert_close(actual, x + 7, rtol=0, atol=0)


def test_non_w4_packed_b4_remains_request_sliced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(
        monkeypatch,
        tuple((row, row + 1, 200 + row, False) for row in range(4)),
    )
    method = RecordingQuantMethod()
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    actual = request_stable.apply_linear_by_request(
        SimpleNamespace(quant_method=method), x, None
    )

    assert [part.shape[0] for part, _bias in method.calls] == [1, 1, 1, 1]
    torch.testing.assert_close(actual, x + 1, rtol=0, atol=0)


def test_unquantized_gdn_packed_b4_uses_one_direct_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(
        monkeypatch,
        tuple((row, row + 1, 200 + row, False) for row in range(4)),
    )
    calls = []
    method = UnquantizedLinearMethod()

    def apply(_layer, value: torch.Tensor, bias=None) -> torch.Tensor:
        calls.append(value)
        return value + 1

    monkeypatch.setattr(method, "apply", apply)
    layer = SimpleNamespace(
        quant_method=method,
        prefix="model.layers.0.linear_attn.out_proj",
    )
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    actual = request_stable.apply_linear_by_request(layer, x, None)

    assert len(calls) == 1
    assert calls[0] is x
    torch.testing.assert_close(actual, x + 1, rtol=0, atol=0)


def test_unquantized_non_gdn_packed_b4_remains_request_sliced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(
        monkeypatch,
        tuple((row, row + 1, 200 + row, False) for row in range(4)),
    )
    calls = []
    method = UnquantizedLinearMethod()

    def apply(_layer, value: torch.Tensor, bias=None) -> torch.Tensor:
        calls.append(value)
        return value + 1

    monkeypatch.setattr(method, "apply", apply)
    layer = SimpleNamespace(
        quant_method=method,
        prefix="model.layers.0.self_attn.unscoped_projection",
    )
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    actual = request_stable.apply_linear_by_request(layer, x, None)

    assert [part.shape[0] for part in calls] == [1, 1, 1, 1]
    torch.testing.assert_close(actual, x + 1, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("slices", "expected_call_rows"),
    [
        (
            tuple((row, row + 1, row, True) for row in range(4)),
            [64, 64, 64, 64],
        ),
        (((0, 4, 200, False),), [64]),
    ],
)
def test_production_xpu_w4_non_decode_b4_keeps_canonical_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    slices,
    expected_call_rows: list[int],
) -> None:
    _set_request_slices(monkeypatch, slices)
    calls = []

    def apply(_layer, value: torch.Tensor, bias=None) -> torch.Tensor:
        calls.append(value)
        return value

    layer = _production_xpu_w4_layer(monkeypatch, apply)
    x = torch.arange(12, dtype=torch.bfloat16).reshape(4, 3)

    actual = request_stable.apply_linear_by_request(layer, x, None)

    assert [part.shape[0] for part in calls] == expected_call_rows
    torch.testing.assert_close(actual, x, rtol=0, atol=0)


def test_production_xpu_w4_packed_b4_validates_direct_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(
        monkeypatch,
        tuple((row, row + 1, 200 + row, False) for row in range(4)),
    )
    layer = _production_xpu_w4_layer(
        monkeypatch,
        lambda _layer, value, bias=None: value.new_empty(5, value.shape[1]),
    )

    with pytest.raises(RuntimeError, match="changed the packed row dimension"):
        request_stable.apply_linear_by_request(
            layer, torch.ones(4, 3, dtype=torch.bfloat16), None
        )


def test_xpu_w4_lookalike_type_cannot_enable_packed_b4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (
        MPLinearLayerConfig,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as compressed_tensors_module,
    )

    lookalike_type = type("XPUwNa16LinearKernel", (), {})
    lookalike_type.__module__ = "vllm.model_executor.kernels.linear.mixed_precision.xpu"
    kernel = lookalike_type()
    kernel.config = MPLinearLayerConfig(
        full_weight_shape=(1, 1),
        partition_weight_shape=(1, 1),
        weight_type=request_stable.scalar_types.uint4b8,
        act_type=torch.bfloat16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )
    calls = []
    method = compressed_tensors_module.CompressedTensorsLinearMethod(SimpleNamespace())
    monkeypatch.setattr(
        method,
        "apply",
        lambda _layer, value, bias=None: calls.append(value) or value,
    )
    layer = SimpleNamespace(
        quant_method=method,
        scheme=SimpleNamespace(kernel=kernel),
        prefix="model.layers.0.self_attn.q_proj",
    )
    _set_request_slices(
        monkeypatch,
        tuple((row, row + 1, 200 + row, False) for row in range(4)),
    )

    with pytest.raises(RuntimeError, match="every compressed linear"):
        request_stable.apply_linear_by_request(
            layer, torch.ones(4, 3, dtype=torch.bfloat16), None
        )
    assert not calls


def test_production_xpu_w4_packed_b4_rejects_malformed_slices_before_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(
        monkeypatch,
        ((0, 1, 200, False), (2, 3, 201, False)),
    )
    calls = []
    layer = _production_xpu_w4_layer(
        monkeypatch,
        lambda _layer, value, bias=None: calls.append(value) or value,
    )

    with pytest.raises(ValueError, match="positive and contiguous"):
        request_stable.apply_linear_by_request(
            layer, torch.ones(3, 2, dtype=torch.bfloat16), None
        )
    assert not calls


def test_production_compressed_linear_classifier_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.quantization.compressed_tensors import (
        compressed_tensors as compressed_tensors_module,
    )

    _set_request_slices(monkeypatch, ((0, 2, 0, True),))
    scheme = SimpleNamespace(
        kernel=object(),
        apply_weights=lambda _layer, value, bias=None: value,
    )
    layer = SimpleNamespace(
        quant_method=compressed_tensors_module.CompressedTensorsLinearMethod(
            SimpleNamespace()
        ),
        scheme=scheme,
        prefix="model.layers.0.self_attn.q_proj",
    )

    with pytest.raises(RuntimeError, match="every compressed linear"):
        request_stable.apply_linear_by_request(
            layer, torch.ones(2, 1, dtype=torch.bfloat16), None
        )


def test_production_gdn_classifier_preserves_mixed_packed_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(
        monkeypatch,
        (
            (0, 1, 200, False),
            (1, 64, 0, True),
            (64, 66, 64, False),
        ),
    )
    calls: list[tuple[torch.Tensor, torch.Tensor | None]] = []
    method = UnquantizedLinearMethod()

    def apply(_layer, value: torch.Tensor, bias=None) -> torch.Tensor:
        calls.append((value, None if bias is None else bias.clone()))
        lanes = torch.arange(value.shape[0], dtype=value.dtype).reshape(-1, 1)
        return value + lanes + (0 if bias is None else bias)

    monkeypatch.setattr(method, "apply", apply)
    bias = torch.tensor([0.25])
    layer = SimpleNamespace(
        quant_method=method,
        prefix="model.layers.0.linear_attn.out_proj",
    )
    x = torch.arange(66, dtype=torch.float32).reshape(-1, 1)

    actual = request_stable.apply_linear_by_request(layer, x, bias)

    expected = torch.cat(
        (
            x[:1] + bias,
            x[1:64] + torch.arange(63).reshape(-1, 1) + bias,
            x[64:] + torch.arange(2).reshape(-1, 1) + bias,
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert [value.shape[0] for value, _bias in calls] == [1, 64, 64]
    assert all(
        call_bias is not None and torch.equal(call_bias, bias)
        for _value, call_bias in calls
    )


def test_apply_linear_by_request_rejects_row_count_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(
        monkeypatch,
        ((0, 1, 200, False), (1, 4, 0, True), (4, 8, 64, True)),
    )
    layer = SimpleNamespace(quant_method=RecordingQuantMethod())

    with pytest.raises(RuntimeError, match="do not cover"):
        request_stable.apply_linear_by_request(layer, torch.ones(7, 3), None)


def test_apply_linear_by_request_rejects_inconsistent_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(monkeypatch, ((0, 2, 0, True), (2, 4, 64, True)))

    class InconsistentMethod:
        calls = 0

        def apply(self, _layer, x: torch.Tensor, _bias=None) -> torch.Tensor:
            self.calls += 1
            width = 3 if self.calls == 1 else 4
            return torch.empty((x.shape[0], width), dtype=x.dtype)

    layer = SimpleNamespace(quant_method=InconsistentMethod())

    with pytest.raises(RuntimeError, match="shape, dtype, or device"):
        request_stable.apply_linear_by_request(layer, torch.ones(4, 3), None)


@pytest.mark.parametrize("fused", [False, True])
def test_apply_rms_norm_by_request_is_batch_and_partition_invariant(
    monkeypatch: pytest.MonkeyPatch,
    fused: bool,
) -> None:
    x = torch.arange(13 * 2 * 3, dtype=torch.float32).reshape(13, 2, 3)
    residual = x / 10 if fused else None

    def run(value, value_residual, slices):
        _set_request_slices(monkeypatch, slices)
        calls = []

        def shape_sensitive(part_x, part_residual):
            calls.append(part_x.shape[0])
            normalized = part_x + part_x.shape[0]
            if part_residual is None:
                return normalized
            return normalized + part_residual, part_x + part_residual

        result = request_stable.apply_rms_norm_by_request(
            value, value_residual, shape_sensitive
        )
        return result, calls

    one, one_calls = run(x, residual, ((0, 13, 0, True),))
    batch_x = x.repeat(4, 1, 1)
    batch_residual = None if residual is None else residual.repeat(4, 1, 1)
    batch, batch_calls = run(
        batch_x,
        batch_residual,
        tuple((lane * 13, (lane + 1) * 13, 0, True) for lane in range(4)),
    )

    one_outputs = one if isinstance(one, tuple) else (one,)
    batch_outputs = batch if isinstance(batch, tuple) else (batch,)
    assert one_calls == [64]
    assert batch_calls == [64, 64, 64, 64]
    for one_output, batch_output in zip(one_outputs, batch_outputs):
        for lane in range(4):
            torch.testing.assert_close(
                batch_output[lane * 13 : (lane + 1) * 13],
                one_output,
                rtol=0,
                atol=0,
            )

    long_x = torch.arange(127 * 3, dtype=torch.float32).reshape(127, 3)
    long_residual = long_x / 10 if fused else None
    one_shot, one_shot_calls = run(long_x, long_residual, ((0, 127, 0, True),))
    first, first_calls = run(
        long_x[:64],
        None if long_residual is None else long_residual[:64],
        ((0, 64, 0, True),),
    )
    second, second_calls = run(
        long_x[64:],
        None if long_residual is None else long_residual[64:],
        ((0, 63, 64, True),),
    )
    one_shot_outputs = one_shot if isinstance(one_shot, tuple) else (one_shot,)
    first_outputs = first if isinstance(first, tuple) else (first,)
    second_outputs = second if isinstance(second, tuple) else (second,)
    assert one_shot_calls == first_calls + second_calls == [64, 64]
    for expected, first_part, second_part in zip(
        one_shot_outputs, first_outputs, second_outputs
    ):
        torch.testing.assert_close(
            expected, torch.cat((first_part, second_part)), rtol=0, atol=0
        )


@pytest.mark.parametrize(
    "position,num_rows,expected_lanes,expected_calls",
    [
        (0, 1, [0], [1]),
        (62, 3, [62, 63, 0], [64, 64]),
        (63, 65, [63, *range(64)], [64, 64]),
        (64, 64, list(range(64)), [64]),
    ],
)
def test_apply_rms_norm_by_request_uses_absolute_canonical_lanes(
    monkeypatch: pytest.MonkeyPatch,
    position: int,
    num_rows: int,
    expected_lanes: list[int],
    expected_calls: list[int],
) -> None:
    _set_request_slices(monkeypatch, ((0, num_rows, position, num_rows != 1),))
    calls = []

    def lane_sensitive(part_x, _part_residual):
        calls.append(part_x.shape[0])
        lanes = torch.arange(part_x.shape[0], dtype=part_x.dtype).reshape(-1, 1)
        return part_x + lanes

    x = torch.zeros(num_rows, 1)
    actual = request_stable.apply_rms_norm_by_request(x, None, lane_sensitive)

    torch.testing.assert_close(
        actual,
        torch.tensor(expected_lanes, dtype=x.dtype).reshape(-1, 1),
        rtol=0,
        atol=0,
    )
    assert calls == expected_calls


def test_apply_rms_norm_by_request_is_inert_off_xpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        request_stable, "current_platform", SimpleNamespace(is_xpu=lambda: False)
    )
    monkeypatch.setattr(
        request_stable,
        "get_xpu_kvarn_request_slices",
        lambda: pytest.fail("non-XPU RMSNorm read the forward context"),
    )
    x = torch.ones(2, 3)

    actual = request_stable.apply_rms_norm_by_request(
        x, None, lambda value, _residual: value + 1
    )

    torch.testing.assert_close(actual, x + 1, rtol=0, atol=0)


@pytest.mark.parametrize("fused", [False, True])
def test_apply_rms_norm_uses_one_request_invariant_call(
    monkeypatch: pytest.MonkeyPatch,
    fused: bool,
) -> None:
    _set_request_slices(
        monkeypatch,
        tuple((row, row + 1, 200 + row, False) for row in range(4)),
    )
    x = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    residual = torch.ones_like(x) if fused else None
    calls = []

    def request_invariant(value, actual_residual):
        calls.append((value, actual_residual))
        output = value + 1
        if actual_residual is None:
            return output
        return output, value + actual_residual

    actual = request_stable.apply_rms_norm_by_request(
        x,
        residual,
        lambda *_args: pytest.fail("request-sliced RMSNorm was called"),
        request_invariant,
    )

    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1] is residual
    actual_outputs = actual if isinstance(actual, tuple) else (actual,)
    assert len(actual_outputs) == (2 if fused else 1)
    torch.testing.assert_close(actual_outputs[0], x + 1, rtol=0, atol=0)
    if fused:
        torch.testing.assert_close(actual_outputs[1], x + residual, rtol=0, atol=0)


@pytest.mark.parametrize("fused", [False, True])
def test_apply_rms_norm_b1_returns_validated_result_directly(
    monkeypatch: pytest.MonkeyPatch,
    fused: bool,
) -> None:
    _set_request_slices(monkeypatch, ((0, 1, 200, False),))
    x = torch.arange(3, dtype=torch.float32).reshape(1, 3)
    residual = torch.ones_like(x) if fused else None
    output = x + 1
    returned = (output, x + residual) if residual is not None else output
    calls = []

    def apply_once(value, actual_residual):
        calls.append((value, actual_residual))
        return returned

    actual = request_stable.apply_rms_norm_by_request(x, residual, apply_once)

    assert actual is returned
    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1] is residual
    actual_outputs = actual if isinstance(actual, tuple) else (actual,)
    expected_outputs = returned if isinstance(returned, tuple) else (returned,)
    for actual_output, expected_output in zip(actual_outputs, expected_outputs):
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("invalid_kind", "message"),
    [
        ("not_tuple", "did not return output and residual"),
        ("output_shape", "changed the packed input shape"),
        ("residual_shape", "changed the residual shape"),
        ("residual_non_tensor", "returned a non-tensor output"),
        ("residual_dtype", "changed residual dtype or device"),
    ],
)
def test_apply_rms_norm_b1_fused_fast_path_validates_underlying_result(
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
    message: str,
) -> None:
    _set_request_slices(monkeypatch, ((0, 1, 200, False),))
    x = torch.ones(1, 3)
    residual = torch.ones_like(x)

    def invalid_output(value, actual_residual):
        if invalid_kind == "not_tuple":
            return value
        if invalid_kind == "output_shape":
            return value[:, :2], actual_residual
        if invalid_kind == "residual_shape":
            return value, actual_residual[:, :2]
        if invalid_kind == "residual_non_tensor":
            return value, "not a tensor"
        return value, actual_residual.to(torch.float64)

    with pytest.raises(RuntimeError, match=message):
        request_stable.apply_rms_norm_by_request(x, residual, invalid_output)


@pytest.mark.parametrize("invalid_kind", ["dtype", "non_tensor"])
def test_apply_rms_norm_by_request_rejects_invalid_first_output(
    monkeypatch: pytest.MonkeyPatch,
    invalid_kind: str,
) -> None:
    _set_request_slices(monkeypatch, ((0, 1, 0, False),))
    x = torch.ones(1, 3)

    def invalid_output(value, _residual):
        if invalid_kind == "dtype":
            return value.to(torch.float64)
        return "not a tensor"

    with pytest.raises(RuntimeError, match="request-stable RMSNorm"):
        request_stable.apply_rms_norm_by_request(x, None, invalid_output)


@pytest.mark.parametrize("fused", [False, True])
def test_gemma_rms_norm_uses_request_stable_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    fused: bool,
) -> None:
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    calls = []

    def recording_dispatch(x, residual, apply_once, request_invariant_apply_once):
        calls.append((x, residual, request_invariant_apply_once))
        return apply_once(x, residual)

    monkeypatch.setattr(request_stable, "apply_rms_norm_by_request", recording_dispatch)
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(torch.zeros(4)),
        variance_epsilon=1e-6,
        _xpu_kvarn_request_stable_rmsnorm=True,
    )
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    residual = torch.ones_like(x) if fused else None

    result = GemmaRMSNorm.forward_native(layer, x, residual)

    assert len(calls) == 1
    assert calls[0][0] is x
    assert calls[0][1] is residual
    assert callable(calls[0][2])
    if fused:
        assert isinstance(result, tuple)
    else:
        assert isinstance(result, torch.Tensor)


@pytest.mark.parametrize("fused", [False, True])
def test_gemma_rms_norm_xpu_preserves_request_stable_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    fused: bool,
) -> None:
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    monkeypatch.setattr(
        request_stable,
        "get_xpu_kvarn_request_slices",
        lambda: request_stable.XPUKvarnRequestSlices(((0, 2, 0, True),)),
    )
    calls = []

    def recording_native(x, residual=None):
        calls.append((x, residual))
        if residual is None:
            return x
        return x, residual

    layer = SimpleNamespace(
        forward_native=recording_native,
        _xpu_kvarn_request_stable_rmsnorm=True,
    )
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    residual = torch.ones_like(x) if fused else None

    result = GemmaRMSNorm.forward_xpu(layer, x, residual)

    assert calls == [(x, residual)]
    if residual is None:
        assert result is x
    else:
        assert isinstance(result, tuple)
        assert result[0] is x
        assert result[1] is residual


def test_gemma_rms_norm_disabled_skips_request_stable_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm

    monkeypatch.setattr(
        request_stable,
        "apply_rms_norm_by_request",
        lambda *_args: pytest.fail("disabled RMSNorm policy used stable dispatch"),
    )
    layer = SimpleNamespace(
        weight=torch.nn.Parameter(torch.zeros(4)),
        variance_epsilon=1e-6,
        _xpu_kvarn_request_stable_rmsnorm=False,
    )
    x = torch.arange(8, dtype=torch.float32).reshape(2, 4)

    result = GemmaRMSNorm.forward_native(layer, x)

    assert isinstance(result, torch.Tensor)


def test_inactive_linear_fast_path_does_not_read_forward_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    method = RecordingQuantMethod()
    layer = SimpleNamespace(
        _xpu_kvarn_request_stable=False,
        quant_method=method,
    )
    monkeypatch.setattr(
        request_stable,
        "get_xpu_kvarn_request_slices",
        lambda: pytest.fail("inactive linear read the forward context"),
    )
    x = torch.ones(2, 3)

    actual = LinearBase._apply_quant_method(layer, x, None)

    torch.testing.assert_close(actual, x + 2, rtol=0, atol=0)
    assert len(method.calls) == 1


@pytest.mark.parametrize(
    "slices",
    [
        (),
        ((1, 2, 0, True),),
        ((0, 2, 0, True), (3, 4, 2, True)),
        ((0, 2, 0, True), (1, 4, 2, True)),
        ((0, 0, 0, True),),
        ((0, "2", 0, True),),
        ((0, 2, -1, True),),
        ((0, 2, 0, "yes"),),
    ],
)
def test_apply_linear_by_request_rejects_malformed_context(
    monkeypatch: pytest.MonkeyPatch,
    slices,
) -> None:
    _set_request_slices(monkeypatch, slices)
    layer = SimpleNamespace(quant_method=RecordingQuantMethod())

    with pytest.raises(ValueError, match="XPU KVarN request slice"):
        request_stable.apply_linear_by_request(layer, torch.ones(4, 3), None)


def _scoped_config(**overrides):
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type="qwen3_5_text"),
            model=request_stable._BRUTUS_MODEL_ID,
            revision=request_stable._BRUTUS_MODEL_REVISION,
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
            ubatch_size=0,
        ),
        use_v2_model_runner=False,
        speculative_config=None,
        lora_config=None,
        compilation_config=SimpleNamespace(
            mode=SimpleNamespace(name="NONE"),
            cudagraph_mode=SimpleNamespace(name="NONE"),
        ),
    )
    for path, value in overrides.items():
        target_name, field = path.split("__", 1)
        setattr(getattr(config, target_name), field, value)
    return config


@pytest.mark.parametrize(
    "overrides",
    [
        {"cache_config__cache_dtype": "bfloat16"},
        {"cache_config__cache_dtype": "kvarn_k4v4_g128"},
        {"cache_config__mamba_cache_mode": "align"},
        {"cache_config__enable_prefix_caching": True},
        {"model_config__enforce_eager": False},
        {"model_config__revision": "other"},
        {"model_config__quantization": "inc"},
        {"parallel_config__enable_dbo": True},
        {"parallel_config__ubatch_size": 2},
        {
            "model_config__hf_text_config": SimpleNamespace(
                model_type="qwen3_5_moe_text"
            )
        },
        {"parallel_config__tensor_parallel_size": 2},
        {"parallel_config__data_parallel_size": 2},
    ],
)
def test_request_stable_config_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict,
) -> None:
    monkeypatch.setattr(
        request_stable, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )

    assert not request_stable.use_xpu_kvarn_request_stable_linears(
        _scoped_config(**overrides)
    )


def test_request_stable_config_accepts_frozen_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        request_stable, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )
    assert request_stable.use_xpu_kvarn_request_stable_linears(_scoped_config())


@pytest.mark.parametrize(
    (
        "projection_rows",
        "rmsnorm",
        "expected_projection_rows",
        "expected_rmsnorm",
        "expected_context",
    ),
    [
        (None, None, True, True, True),
        ("0", "1", False, True, True),
        ("1", "0", True, False, True),
        ("0", "0", False, False, False),
    ],
)
def test_request_stability_axes_are_independent(
    monkeypatch: pytest.MonkeyPatch,
    projection_rows: str | None,
    rmsnorm: str | None,
    expected_projection_rows: bool,
    expected_rmsnorm: bool,
    expected_context: bool,
) -> None:
    monkeypatch.setattr(
        request_stable, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )
    selector_values = {
        request_stable.XPU_KVARN_REQUEST_STABLE_PROJECTION_ROWS_ENV: projection_rows,
        request_stable.XPU_KVARN_REQUEST_STABLE_RMSNORM_ENV: rmsnorm,
    }
    for name, value in selector_values.items():
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    config = _scoped_config()
    assert (
        request_stable.use_xpu_kvarn_request_stable_projection_rows(config)
        is expected_projection_rows
    )
    assert (
        request_stable.use_xpu_kvarn_request_stable_rmsnorm(config) is expected_rmsnorm
    )
    assert (
        request_stable.use_xpu_kvarn_request_stable_context(config) is expected_context
    )


@pytest.mark.parametrize(
    "selector",
    [
        request_stable.XPU_KVARN_REQUEST_STABLE_PROJECTION_ROWS_ENV,
        request_stable.XPU_KVARN_REQUEST_STABLE_RMSNORM_ENV,
    ],
)
@pytest.mark.parametrize("raw_value", ["", "false", "2", " 0"])
def test_request_stability_axes_reject_invalid_values_at_startup(
    monkeypatch: pytest.MonkeyPatch,
    selector: str,
    raw_value: str,
) -> None:
    monkeypatch.setattr(
        request_stable, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )
    monkeypatch.setenv(selector, raw_value)

    with pytest.raises(ValueError, match=f"{selector} must be exactly '0' or '1'"):
        request_stable.configure_xpu_kvarn_request_stability(_scoped_config())


def test_request_stability_axes_are_immutable_after_first_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(request_stable.XPU_KVARN_REQUEST_STABLE_PROJECTION_ROWS_ENV, "0")
    policy = request_stable._get_xpu_kvarn_request_stability_policy()
    monkeypatch.setenv(request_stable.XPU_KVARN_REQUEST_STABLE_PROJECTION_ROWS_ENV, "1")

    assert policy.projection_rows is False
    assert request_stable._get_xpu_kvarn_request_stability_policy() is policy


def test_request_stability_startup_logs_effective_factory_axes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        request_stable, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )
    monkeypatch.setenv(request_stable.XPU_KVARN_REQUEST_STABLE_PROJECTION_ROWS_ENV, "0")
    monkeypatch.delenv(
        request_stable.XPU_KVARN_REQUEST_STABLE_RMSNORM_ENV, raising=False
    )
    calls = []
    monkeypatch.setattr(
        request_stable.logger,
        "info_once",
        lambda *args: calls.append(args),
    )

    request_stable.configure_xpu_kvarn_request_stability(_scoped_config())

    assert calls == [
        (
            (
                "[KVARN_FACTORY] selected_request_stable_projection_rows=%s; "
                "projection_rows_selector_source=%s; "
                "selected_request_stable_rmsnorm=%s; rmsnorm_selector_source=%s; "
                "profile_eligible=%s; immutable for engine lifetime"
            ),
            "false",
            request_stable.XPU_KVARN_REQUEST_STABLE_PROJECTION_ROWS_ENV,
            "true",
            "safe-default",
            "true",
        )
    ]


def test_request_stability_axes_do_not_own_non_kvarn_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        request_stable.XPU_KVARN_REQUEST_STABLE_PROJECTION_ROWS_ENV, "invalid"
    )
    config = _scoped_config(cache_config__cache_dtype="auto")

    assert request_stable.configure_xpu_kvarn_request_stability(config) is None


def test_request_stable_config_accepts_explicit_single_row_ubatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        request_stable, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )
    config = _scoped_config()
    config.parallel_config.ubatch_size = 1

    assert request_stable.use_xpu_kvarn_request_stable_linears(config)


def test_request_stable_config_rejects_v2_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        request_stable, "current_platform", SimpleNamespace(is_xpu=lambda: True)
    )
    config = _scoped_config()
    config.use_v2_model_runner = True

    assert not request_stable.use_xpu_kvarn_request_stable_linears(config)


def test_logits_head_packed_b4_uses_one_direct_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(
        monkeypatch,
        tuple((row, row + 1, 200 + row, False) for row in range(4)),
    )
    processor = LogitsProcessor.__new__(LogitsProcessor)
    processor._xpu_kvarn_request_stable = True
    processor.head_dtype = None
    method = RecordingQuantMethod()
    lm_head = SimpleNamespace(quant_method=method)
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    bias = torch.arange(3, dtype=torch.float32)

    actual = processor._apply_head(lm_head, hidden_states, bias)

    torch.testing.assert_close(actual, hidden_states + 4, rtol=0, atol=0)
    assert len(method.calls) == 1
    assert method.calls[0][0] is hidden_states
    assert method.calls[0][1] is bias


def test_logits_head_non_decode_batch_remains_request_sliced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_request_slices(monkeypatch, ((0, 4, 200, True),))
    processor = LogitsProcessor.__new__(LogitsProcessor)
    processor._xpu_kvarn_request_stable = True
    processor.head_dtype = None
    method = RecordingQuantMethod()
    lm_head = SimpleNamespace(quant_method=method)
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    actual = processor._apply_head(lm_head, hidden_states, None)

    torch.testing.assert_close(actual, hidden_states + 1, rtol=0, atol=0)
    assert [call_x.shape[0] for call_x, _bias in method.calls] == [1, 1, 1, 1]


def test_logits_head_disabled_profile_preserves_batched_projection() -> None:
    processor = LogitsProcessor.__new__(LogitsProcessor)
    processor._xpu_kvarn_request_stable = False
    processor.head_dtype = None
    method = RecordingQuantMethod()
    lm_head = SimpleNamespace(quant_method=method)
    hidden_states = torch.arange(12, dtype=torch.float32).reshape(4, 3)

    actual = processor._apply_head(lm_head, hidden_states, None)

    torch.testing.assert_close(actual, hidden_states + 4, rtol=0, atol=0)
    assert len(method.calls) == 1
    assert method.calls[0][0] is hidden_states


def test_logits_head_active_profile_rejects_non_matrix_input() -> None:
    processor = LogitsProcessor.__new__(LogitsProcessor)
    processor._xpu_kvarn_request_stable = True
    processor.head_dtype = None
    lm_head = SimpleNamespace(quant_method=RecordingQuantMethod())

    with pytest.raises(RuntimeError, match="two-dimensional"):
        processor._apply_head(lm_head, torch.ones(1, 2, 3), None)
