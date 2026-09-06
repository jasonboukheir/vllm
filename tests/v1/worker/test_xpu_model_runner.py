# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Unit tests for ``vllm.v1.worker.xpu_model_runner`` (XPU worker / CUDA shims)."""

import os
from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.variables.torch import TorchInGraphFunctionVariable

from vllm.v1.worker import xpu_worker
from vllm.v1.worker.xpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.xpu_worker import (
    XPUWorker,
    _configure_kvarn_onednn_determinism,
)


class _EnvResolvedRunnerConfig:
    def __init__(self, cache_dtype: str) -> None:
        self.cache_config = SimpleNamespace(cache_dtype=cache_dtype)
        self.device_config = SimpleNamespace(device_type="xpu")

    @property
    def use_v2_model_runner(self) -> bool:
        return os.environ.get("VLLM_USE_V2_MODEL_RUNNER") == "1"


def _stub_worker_init(worker, vllm_config, *_args, **_kwargs) -> None:
    worker.vllm_config = vllm_config
    worker.cache_config = vllm_config.cache_config
    worker.device_config = vllm_config.device_config
    worker.use_v2_model_runner = vllm_config.use_v2_model_runner


def test_xpu_worker_rejects_kvarn_v2_selected_by_divergent_worker_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    config = _EnvResolvedRunnerConfig("kvarn_k4v4_g128_compact")
    assert not config.use_v2_model_runner

    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr(xpu_worker.Worker, "__init__", _stub_worker_init)
    monkeypatch.setattr(xpu_worker.current_platform, "is_xpu", lambda: True)

    with pytest.raises(
        ValueError,
        match="this worker selected Model Runner V2",
    ):
        XPUWorker(config, 0, 0, "local")  # type: ignore[arg-type]


def test_xpu_worker_leaves_non_kvarn_v2_selection_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    config = _EnvResolvedRunnerConfig("auto")
    monkeypatch.setattr(xpu_worker.Worker, "__init__", _stub_worker_init)
    monkeypatch.setattr(xpu_worker.current_platform, "is_xpu", lambda: True)

    worker = XPUWorker(config, 0, 0, "local")  # type: ignore[arg-type]

    assert worker.use_v2_model_runner


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


def test_configure_kvarn_onednn_determinism_logs_factory_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("KVARN_ONEDNN_DETERMINISTIC", raising=False)
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
            "true",
            "release-default",
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
