# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``vllm.v1.worker.xpu_model_runner`` (XPU worker / CUDA shims)."""

import pytest
import torch
from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

from vllm.v1.worker.xpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.xpu_worker import _configure_kvarn_onednn_determinism


@pytest.mark.parametrize(
    ("cache_dtype", "initial", "expected_selection", "expected_deterministic"),
    [
        ("kvarn_k4v4_g128_compact", False, True, True),
        ("kvarn_mla_k4v4_g128", False, True, True),
        ("auto", False, None, False),
        (None, False, None, False),
        ("auto", True, None, True),
    ],
)
def test_configure_kvarn_onednn_determinism_safe_default(
    monkeypatch: pytest.MonkeyPatch,
    cache_dtype: str | None,
    initial: bool,
    expected_selection: bool | None,
    expected_deterministic: bool,
) -> None:
    monkeypatch.delenv("KVARN_ONEDNN_DETERMINISTIC", raising=False)
    previous = torch.backends.mkldnn.deterministic
    try:
        torch.backends.mkldnn.deterministic = initial
        selection = _configure_kvarn_onednn_determinism(cache_dtype)
        assert selection is expected_selection
        assert torch.backends.mkldnn.deterministic is expected_deterministic
    finally:
        torch.backends.mkldnn.deterministic = previous


@pytest.mark.parametrize(("raw_value", "expected"), [("0", False), ("1", True)])
def test_configure_kvarn_onednn_determinism_explicit_selection(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
    expected: bool,
) -> None:
    monkeypatch.setenv("KVARN_ONEDNN_DETERMINISTIC", raw_value)
    previous = torch.backends.mkldnn.deterministic
    try:
        torch.backends.mkldnn.deterministic = not expected
        assert (
            _configure_kvarn_onednn_determinism("kvarn_k4v4_g128_compact") is expected
        )
        assert torch.backends.mkldnn.deterministic is expected
    finally:
        torch.backends.mkldnn.deterministic = previous


@pytest.mark.parametrize("raw_value", ["", "false", "2", " 0"])
def test_configure_kvarn_onednn_determinism_rejects_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
    raw_value: str,
) -> None:
    monkeypatch.setenv("KVARN_ONEDNN_DETERMINISTIC", raw_value)
    previous = torch.backends.mkldnn.deterministic
    try:
        torch.backends.mkldnn.deterministic = True
        with pytest.raises(
            ValueError,
            match="KVARN_ONEDNN_DETERMINISTIC must be exactly '0' or '1'",
        ):
            _configure_kvarn_onednn_determinism("kvarn_k4v4_g128_compact")
        assert torch.backends.mkldnn.deterministic is True
    finally:
        torch.backends.mkldnn.deterministic = previous


def test_configure_kvarn_onednn_determinism_logs_factory_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_ONEDNN_DETERMINISTIC", "0")
    calls: list[tuple[str, str, str]] = []
    monkeypatch.setattr(
        "vllm.v1.worker.xpu_worker.logger.info_once",
        lambda message, value, source: calls.append((message, value, source)),
    )
    previous = torch.backends.mkldnn.deterministic
    try:
        _configure_kvarn_onednn_determinism("kvarn_k4v4_g128_compact")
    finally:
        torch.backends.mkldnn.deterministic = previous

    assert calls == [
        (
            (
                "[KVARN_FACTORY] selected_onednn_deterministic=%s; "
                "selector_source=%s; immutable for engine lifetime"
            ),
            "false",
            "KVARN_ONEDNN_DETERMINISTIC",
        )
    ]


# Child process: patched torch.cuda must not leak to other tests in the session.
@pytest.mark.skipif(
    not hasattr(torch, "xpu") or not hasattr(torch.xpu, "current_stream"),
    reason="torch.xpu.current_stream is required",
)
@pytest.mark.forked
def test_torch_cuda_wrapper_allows_dynamo_handler_registration() -> None:
    """Guard against XPU CUDA shim breaking Torch Dynamo during AOT compile.

    Before the fix, ``_torch_cuda_wrapper`` assigned
    ``torch.cuda.current_stream = torch.xpu.current_stream`` (same function object).
    On the first AOT/profile run, Dynamo builds its in-graph handler table and
    registers ``torch.cuda.current_stream`` and ``torch.xpu.current_stream``
    separately; duplicate identity triggers::

        AssertionError: Handler already registered for <function current_stream ...>

    That surfaced as EngineCore failing in ``profile_run`` / ``_get_handlers()``.
    The fix uses distinct shim callables so both can be registered.

    This test replays the post-init state (wrapper applied, patches left on
    ``torch.cuda``) and checks that Dynamo's real ``_get_handlers()`` succeeds.
    """
    # Same entry point as XPUModelRunner.__init__ (patches persist after exit).
    with _torch_cuda_wrapper():
        pass

    # Fresh handler table build, as on first torch.compile / AOT in the worker.
    # Registers torch.cuda.current_stream and torch.xpu.current_stream separately;
    # if they are the same object (pre-fix alias), raises Handler already registered.
    TorchInGraphFunctionVariable._get_handlers.cache_clear()
    TorchInGraphFunctionVariable._get_handlers()
