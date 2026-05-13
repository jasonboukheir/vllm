# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for WorkspaceManager.

Focused on the lock semantics that turboquant_attn._decode_attention
depends on: a locked workspace that is already large enough must still
hand out tensor views, while a locked workspace that would need to grow
must return None (via try_get_simultaneous) instead of raising — so the
caller can fall back to per-call allocation. Repros the failure mode in
vllm-project/vllm#42544.
"""

import pytest
import torch

from vllm.v1.worker.workspace import WorkspaceManager


@pytest.fixture
def manager() -> WorkspaceManager:
    return WorkspaceManager(device=torch.device("cpu"), num_ubatches=1)


def test_get_simultaneous_grows_when_unlocked(manager: WorkspaceManager) -> None:
    a, b = manager.get_simultaneous(
        ((16,), torch.float32),
        ((8,), torch.float32),
    )
    assert a.shape == (16,)
    assert b.shape == (8,)
    assert a.dtype == torch.float32


def test_get_simultaneous_raises_on_locked_undersized(
    manager: WorkspaceManager,
) -> None:
    manager.lock()
    with pytest.raises(AssertionError, match="Workspace is locked"):
        manager.get_simultaneous(((1024,), torch.float32))


def test_get_simultaneous_succeeds_on_locked_when_fits(
    manager: WorkspaceManager,
) -> None:
    manager.get_simultaneous(((1024,), torch.float32))
    manager.lock()
    (t,) = manager.get_simultaneous(((128,), torch.float32))
    assert t.shape == (128,)


def test_try_get_simultaneous_returns_none_when_locked_undersized(
    manager: WorkspaceManager,
) -> None:
    """The failure mode from issue #42544: workspace locked at 0 bytes,
    decode-time allocation requested. Must not raise."""
    manager.lock()
    result = manager.try_get_simultaneous(
        ((24, 4, 8, 257), torch.float32),
        ((24, 4, 256), torch.bfloat16),
        ((24, 4), torch.float32),
    )
    assert result is None


def test_try_get_simultaneous_returns_views_when_locked_and_fits(
    manager: WorkspaceManager,
) -> None:
    manager.get_simultaneous(
        ((24, 4, 8, 257), torch.float32),
        ((24, 4, 256), torch.bfloat16),
        ((24, 4), torch.float32),
    )
    manager.lock()
    result = manager.try_get_simultaneous(
        ((24, 4, 8, 257), torch.float32),
        ((24, 4, 256), torch.bfloat16),
        ((24, 4), torch.float32),
    )
    assert result is not None
    mid, out, lse = result
    assert mid.shape == (24, 4, 8, 257)
    assert out.shape == (24, 4, 256)
    assert out.dtype == torch.bfloat16
    assert lse.shape == (24, 4)


def test_try_get_simultaneous_grows_when_unlocked(
    manager: WorkspaceManager,
) -> None:
    result = manager.try_get_simultaneous(((1024,), torch.float32))
    assert result is not None
    (t,) = result
    assert t.shape == (1024,)
