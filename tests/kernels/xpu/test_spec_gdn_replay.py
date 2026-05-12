# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay harness for the SYCL gdn_attention spec-decoding work.

Drives Rung 2+ of the test ladder documented in
``vllm/.spec-gdn-progress.md``.

Each input ``.pt`` tuple was produced by the capture hook in
``vllm/_xpu_ops.py`` (``VLLM_XPU_DUMP_SPEC_GDN=<dir>`` plus
``VLLM_XPU_FORCE_FLA_GDN=1``). Schema is documented in
``_spec_gdn_dump_call`` (schema_version=2) — at minimum each payload
contains ``projected_states_qkvz/ba`` (the SYCL kernel's inputs), the
post-call FLA outputs, and the pre/post slices of the kv_cache pool at
the slots referenced by the call.

This harness:
  * Auto-discovers tuples in ``$VLLM_XPU_SPEC_GDN_CAPTURES`` (default
    ``/tmp/spec_gdn_captures``); skips cleanly if the directory is
    empty so CI without captures still passes.
  * Validates schema version + presence of required keys.
  * Reconstructs a *dense* kv_cache pool just large enough to hold the
    referenced slots (the original pool can be thousands of blocks; we
    remap to ``[0, N+1)`` with slot 0 reserved as the FLA NULL_BLOCK_ID
    sentinel) and writes the captured pre-state into it.
  * For non-spec captures, calls ``torch.ops._xpu_C.gdn_attention``
    directly and diffs the resulting outputs + post-state against the
    captured FLA reference (oracle). Tolerance: ``atol=2e-2 rtol=2e-2``
    for bf16 — tightened later if numerics permit.
  * For spec captures, marks the test ``xfail`` with a clear pointer to
    the pending kernel work (Rung 3+ extends the SYCL op signature with
    ``spec_state_indices_tensor`` + ``num_accepted_tokens``).

Run locally with::

    .venv/bin/python -m pytest tests/kernels/xpu/test_spec_gdn_replay.py -v

Setting ``VLLM_XPU_SPEC_GDN_CAPTURES`` is required if captures live
elsewhere.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


# ---------------------------------------------------------------------------
# Capture discovery
# ---------------------------------------------------------------------------

_DEFAULT_CAPTURES = "/tmp/spec_gdn_captures"
_SCHEMA_VERSION = 2

_REQUIRED_KEYS = (
    "schema_version",
    "layer_prefix",
    "step",
    "flavor",
    "projected_states_qkvz",
    "projected_states_ba",
    "mixed_qkv",
    "b",
    "a",
    "slot_indices",
    "conv_state_pre",
    "conv_state_post",
    "ssm_state_pre",
    "ssm_state_post",
    "core_attn_out",
    "z",
    "non_spec_query_start_loc",
    "non_spec_state_indices_tensor",
    "num_actual_tokens",
    "num_prefills",
    "num_decodes",
    "layer_config",
    "conv_weight",
    "A_log",
    "dt_bias",
)


def _captures_dir() -> Path:
    return Path(os.environ.get("VLLM_XPU_SPEC_GDN_CAPTURES", _DEFAULT_CAPTURES))


def _discover_tuples() -> list[Path]:
    d = _captures_dir()
    if not d.is_dir():
        return []
    return sorted(d.glob("tuple_*.pt"))


_TUPLES = _discover_tuples()


def _split_capture_name(p: Path) -> tuple[str, str]:
    """Parse ``tuple_<sanitized_prefix>_<step:06d>_<flavor>.pt`` into
    ``(prefix, flavor)``. Both parts may contain underscores
    (``language_model_model_layers_0_linear_attn`` and ``spec_K4_min1_max1``);
    the 6-digit step is the unique anchor between them.
    """
    body = p.stem[len("tuple_") :] if p.stem.startswith("tuple_") else p.stem
    tokens = body.split("_")
    for i in range(len(tokens) - 1, -1, -1):
        if len(tokens[i]) == 6 and tokens[i].isdigit():
            return "_".join(tokens[:i]), "_".join(tokens[i + 1 :])
    return body, ""


def _layer_prefix_from_name(p: Path) -> str:
    return _split_capture_name(p)[0]


def _flavor_from_name(p: Path) -> str:
    return _split_capture_name(p)[1]


def _pair_mixed_tuples() -> list[tuple[Path, Path]]:
    """Pair every ``non_spec`` capture with the lowest-numbered ``spec`` capture
    from the same layer prefix. The harness can then synthesise a mixed batch
    by concatenating the two payloads and exercising the spec/non-spec split
    that ``_gdn_xpu_spec_sycl_path`` does in production.
    """
    by_layer_ns: dict[str, Path] = {}
    by_layer_spec: dict[str, Path] = {}
    for t in _TUPLES:
        prefix = _layer_prefix_from_name(t)
        flavor = _flavor_from_name(t)
        if flavor == "non_spec":
            by_layer_ns.setdefault(prefix, t)
        elif flavor.startswith("spec_") and not flavor.endswith("_mixed"):
            by_layer_spec.setdefault(prefix, t)
    pairs: list[tuple[Path, Path]] = []
    for prefix, ns in by_layer_ns.items():
        sp = by_layer_spec.get(prefix)
        if sp is not None:
            pairs.append((ns, sp))
    return pairs


_MIXED_PAIRS = _pair_mixed_tuples()


def _xpu_available() -> bool:
    return getattr(torch, "xpu", None) is not None and torch.xpu.is_available()


_XPU_OK = _xpu_available()
_HAS_OP = (
    _XPU_OK
    and hasattr(torch.ops, "_xpu_C")
    and hasattr(torch.ops._xpu_C, "gdn_attention")
)


pytestmark = [
    pytest.mark.skipif(not _TUPLES, reason="No GDN spec captures discovered"),
    pytest.mark.skipif(not _XPU_OK, reason="XPU device not available"),
    pytest.mark.skipif(not _HAS_OP, reason="_xpu_C.gdn_attention op not registered"),
]


# ---------------------------------------------------------------------------
# Pool reconstruction
# ---------------------------------------------------------------------------


def _build_dense_pool(payload, device):
    """Materialise a dense kv_cache pool sized to the slots actually
    referenced by this call. Returns ``(conv_pool, ssm_pool, idx_remap,
    pool_size)`` where ``idx_remap[old_slot] = new_slot`` and slot 0 is
    reserved as the FLA NULL_BLOCK_ID sentinel.
    """
    slots = payload["slot_indices"].to(torch.long)
    if slots.numel() == 0:
        return None, None, None, 0

    real_slots = sorted({int(s) for s in slots.tolist()} - {0})
    pool_size = len(real_slots) + 1  # +1 for the slot-0 null sentinel

    max_slot = int(slots.max().item()) + 1
    remap = torch.zeros(max_slot, dtype=torch.long, device=device)
    for dense_idx, src_slot in enumerate(real_slots, start=1):
        remap[src_slot] = dense_idx

    conv_pre = payload["conv_state_pre"].to(device)
    ssm_pre = payload["ssm_state_pre"].to(device)

    conv_pool = torch.zeros(
        (pool_size,) + tuple(conv_pre.shape[1:]),
        dtype=conv_pre.dtype,
        device=device,
    )
    ssm_pool = torch.zeros(
        (pool_size,) + tuple(ssm_pre.shape[1:]),
        dtype=ssm_pre.dtype,
        device=device,
    )

    slots_dense = remap[slots.to(remap.device)]
    keep = slots_dense > 0
    if keep.any():
        conv_pool[slots_dense[keep]] = conv_pre[keep]
        ssm_pool[slots_dense[keep]] = ssm_pre[keep]

    return conv_pool, ssm_pool, remap, pool_size


def _remap_index_tensor(t, remap):
    if t is None:
        return None
    # The kernel treats slot ids <= 0 as NULL_BLOCK_ID; map them to the
    # dense pool's slot-0 sentinel instead of letting negative indices
    # wrap to the end of `remap`.
    t_long = t.to(torch.long).to(remap.device)
    out = torch.zeros_like(t_long)
    mask = t_long > 0
    out[mask] = remap[t_long[mask]]
    return out


def _build_unified_pool(non_spec_payload, spec_payload, device):
    """Materialise a single conv/ssm pool covering the slots referenced by both
    payloads. Returns ``(conv_pool, ssm_pool, ns_remap, spec_remap)``: the
    remap tensors map raw slots from each payload into the unified pool.

    Slot 0 is reserved as the FLA NULL_BLOCK_ID sentinel; remaining slots are
    assigned dense indices in the unified pool, with non-spec slots placed
    before spec slots so the two domains are visibly disjoint.
    """
    ns_slots_t = non_spec_payload["slot_indices"].to(torch.long)
    sp_slots_t = spec_payload["slot_indices"].to(torch.long)

    ns_real = sorted({int(s) for s in ns_slots_t.tolist()} - {0})
    sp_real = sorted({int(s) for s in sp_slots_t.tolist()} - {0})

    # Allocate non-overlapping dense indices, even if the same slot id happens
    # to appear in both captures (production avoids this collision; the
    # harness must handle it because both captures were taken in isolation).
    pool_size = 1 + len(ns_real) + len(sp_real)

    def _make_remap(raw_slots, dense_start, src_slot_max):
        max_slot = src_slot_max + 1
        remap = torch.zeros(max_slot, dtype=torch.long, device=device)
        for i, src in enumerate(raw_slots):
            remap[src] = dense_start + i
        return remap

    ns_max = int(ns_slots_t.max().item()) if ns_slots_t.numel() else 0
    sp_max = int(sp_slots_t.max().item()) if sp_slots_t.numel() else 0
    ns_remap = _make_remap(ns_real, 1, ns_max)
    sp_remap = _make_remap(sp_real, 1 + len(ns_real), sp_max)

    conv_pre_ns = non_spec_payload["conv_state_pre"]
    ssm_pre_ns = non_spec_payload["ssm_state_pre"]
    conv_pre_sp = spec_payload["conv_state_pre"]
    ssm_pre_sp = spec_payload["ssm_state_pre"]

    conv_pool = torch.zeros(
        (pool_size,) + tuple(conv_pre_ns.shape[1:]),
        dtype=conv_pre_ns.dtype,
        device=device,
    )
    ssm_pool = torch.zeros(
        (pool_size,) + tuple(ssm_pre_ns.shape[1:]),
        dtype=ssm_pre_ns.dtype,
        device=device,
    )

    if ns_slots_t.numel():
        ns_dense = ns_remap[ns_slots_t.to(device)]
        keep_ns = ns_dense > 0
        if keep_ns.any():
            conv_pool[ns_dense[keep_ns]] = conv_pre_ns.to(device)[keep_ns]
            ssm_pool[ns_dense[keep_ns]] = ssm_pre_ns.to(device)[keep_ns]
    if sp_slots_t.numel():
        sp_dense = sp_remap[sp_slots_t.to(device)]
        keep_sp = sp_dense > 0
        if keep_sp.any():
            conv_pool[sp_dense[keep_sp]] = conv_pre_sp.to(device)[keep_sp]
            ssm_pool[sp_dense[keep_sp]] = ssm_pre_sp.to(device)[keep_sp]

    return conv_pool, ssm_pool, ns_remap, sp_remap


# ---------------------------------------------------------------------------
# qkvz / ba layout permutation (Rung 8: gqa_interleaved=True coverage)
# ---------------------------------------------------------------------------


def _qkvz_false_to_true(qkvz_false: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Repack a non-interleaved ``qkvz`` projection (gqa_interleaved=False) into
    the per-k_head interleaved layout the kernel expects under
    ``reorder_input=False``.

    False layout: ``[Q_block | K_block | V_block | Z_block]``.
    True layout:  ``[k_head_0:(q,k,v,z) | k_head_1:(q,k,v,z) | ...]``.

    Mirrors the inverse of the ``ReorderInput`` index math in
    ``causal_conv1d.hpp`` (lines ~194-225).
    """
    n = qkvz_false.size(0)
    H_k = cfg["num_k_heads"] // cfg["tp_size"]
    H_v = cfg["num_v_heads"] // cfg["tp_size"]
    d_k = cfg["head_k_dim"]
    d_v = cfg["head_v_dim"]
    v_per_k = H_v // H_k
    Q = H_k * d_k
    K = H_k * d_k
    V = H_v * d_v
    q = qkvz_false[:, :Q].reshape(n, H_k, d_k)
    k = qkvz_false[:, Q : Q + K].reshape(n, H_k, d_k)
    v = qkvz_false[:, Q + K : Q + K + V].reshape(n, H_k, v_per_k * d_v)
    z = qkvz_false[:, Q + K + V :].reshape(n, H_k, v_per_k * d_v)
    return torch.cat([q, k, v, z], dim=-1).reshape(n, -1).contiguous()


def _ba_false_to_true(ba_false: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Repack ``[b | a]`` (each ``num_v_heads`` wide) into the per-k_head
    layout ``[kh0_b | kh0_a | kh1_b | kh1_a | ...]`` the kernel reads when
    ``reorder_input=False`` (see causal_conv1d.hpp:131-143).
    """
    n = ba_false.size(0)
    H_k = cfg["num_k_heads"] // cfg["tp_size"]
    H_v = cfg["num_v_heads"] // cfg["tp_size"]
    v_per_k = H_v // H_k
    b = ba_false[:, :H_v].reshape(n, H_k, v_per_k)
    a = ba_false[:, H_v:].reshape(n, H_k, v_per_k)
    return torch.cat([b, a], dim=-1).reshape(n, -1).contiguous()


# ---------------------------------------------------------------------------
# SYCL invocation
# ---------------------------------------------------------------------------


def _use_sycl_spec_env() -> bool:
    return os.environ.get("VLLM_XPU_USE_SYCL_SPEC_GDN", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _call_sycl(payload, conv_pool, ssm_pool, remap, device):
    """Drive ``torch.ops._xpu_C.gdn_attention`` from a captured payload.

    Branch by flavor:
    - ``non_spec``: passes the non_spec query_start_loc / state_indices the
      kernel already understood pre-Rung 3.
    - ``spec_*`` (when the kernel supports it): passes the spec_* tensors
      via the Rung 3 kwargs and uses ``spec_query_start_loc`` as the
      sequence boundary table. For now this assumes a *spec-only* batch
      (matches captures from a Qwen3.6 + MTP-K3 run); mixed-batch tuples
      would need the same Python-side split that
      ``_gdn_xpu_spec_sycl_path`` does — flagged xfail in the test below.
    """
    cfg = payload["layer_config"]
    n_actual = int(payload["num_actual_tokens"])
    flavor = payload["flavor"]
    is_spec_capture = flavor != "non_spec"

    qkvz = payload["projected_states_qkvz"].to(device)
    ba = payload["projected_states_ba"].to(device)

    core_attn_out = torch.empty(
        payload["core_attn_out"].shape,
        dtype=payload["core_attn_out"].dtype,
        device=device,
    )
    z = torch.empty(payload["z"].shape, dtype=payload["z"].dtype, device=device)

    conv_w = payload["conv_weight"].to(device)
    if conv_w.dim() == 3:
        conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))
    conv_b = payload["conv_bias"]
    if conv_b is not None:
        conv_b = conv_b.to(device)

    has_initial_state = payload.get("has_initial_state")
    if has_initial_state is not None:
        has_initial_state = has_initial_state.to(device)

    common = dict(
        conv_state=conv_pool,
        ssm_state=ssm_pool,
        conv_weights=conv_w,
        conv_bias=conv_b,
        activation=cfg["activation"],
        A_log=payload["A_log"].to(device),
        dt_bias=payload["dt_bias"].to(device),
        tp_size=cfg["tp_size"],
        reorder_input=not cfg["gqa_interleaved_layout"],
    )

    if is_spec_capture:
        spec_qsl = payload["spec_query_start_loc"]
        if spec_qsl is None:
            pytest.skip("spec capture missing spec_query_start_loc — re-capture")
        spec_qsl = spec_qsl.to(device).contiguous()
        spec_idx_dense = (
            _remap_index_tensor(payload["spec_state_indices_tensor"], remap)
            .to(torch.int32)
            .contiguous()
        )
        num_acc = payload["num_accepted_tokens"]
        if num_acc is None:
            pytest.skip("spec capture missing num_accepted_tokens — re-capture")
        num_acc = num_acc.to(device).to(torch.int32).contiguous()
        num_spec_decodes = int(payload["num_spec_decodes"])
        # Sentinel non_spec slot lookup: passed for argument validation only;
        # the kernel ignores it under IS_SPEC.
        non_spec_idx_sentinel = spec_idx_dense[:, 0].contiguous()

        torch.ops._xpu_C.gdn_attention(
            core_attn_out[:n_actual],
            z[:n_actual],
            qkvz[:n_actual].contiguous(),
            ba[:n_actual].contiguous(),
            cfg["num_k_heads"],
            cfg["num_v_heads"],
            cfg["head_k_dim"],
            cfg["head_v_dim"],
            num_prefills=0,
            num_decodes=num_spec_decodes,
            has_initial_state=None,
            non_spec_query_start_loc=spec_qsl[: num_spec_decodes + 1].contiguous(),
            non_spec_state_indices_tensor=non_spec_idx_sentinel,
            num_actual_tokens=n_actual,
            spec_state_indices_tensor=spec_idx_dense,
            num_accepted_tokens=num_acc,
            **common,
        )
    else:
        non_spec_qsl = payload["non_spec_query_start_loc"]
        if non_spec_qsl is not None:
            non_spec_qsl = non_spec_qsl.to(device)
        non_spec_idx = _remap_index_tensor(
            payload["non_spec_state_indices_tensor"], remap
        )
        torch.ops._xpu_C.gdn_attention(
            core_attn_out[:n_actual],
            z[:n_actual],
            qkvz[:n_actual],
            ba[:n_actual],
            cfg["num_k_heads"],
            cfg["num_v_heads"],
            cfg["head_k_dim"],
            cfg["head_v_dim"],
            num_prefills=int(payload["num_prefills"]),
            num_decodes=int(payload["num_decodes"]),
            has_initial_state=has_initial_state,
            non_spec_query_start_loc=non_spec_qsl,
            non_spec_state_indices_tensor=non_spec_idx,
            num_actual_tokens=n_actual,
            **common,
        )

    return core_attn_out, z, conv_pool, ssm_pool


def _compute_fla_spec_oracle(payload, device):
    """Reproduce the FLA Triton spec path on captured inputs.

    Replaces the captured ``payload["core_attn_out"]`` / ``payload["ssm_state_post"]``
    as the oracle for spec captures. Cross-check (tick 10) showed those captured
    outputs are NOT byte-reproducible by re-running the same FLA kernel on the
    same inputs — there's hidden non-determinism (autotune cache, reduction
    ordering, or unreplicated state) at capture time. Driving the oracle inline
    per replay gives a stable target the SYCL kernel can be diffed against.

    Mirrors ``GatedDeltaNetAttention._forward_core`` (gdn_linear_attn.py:830-943)
    for the spec branch only:
      1. ``causal_conv1d_update`` on captured ``mixed_qkv`` writing into a fresh
         conv pool.
      2. Flat split into q/k/v (matches ``_gdn_xpu_spec_python_path``'s
         post-rearrange concat order).
      3. ``fused_sigmoid_gating_delta_rule_update`` writing into a fresh ssm pool.

    Returns ``(core_attn_out, conv_pool_post, ssm_pool_post, remap)`` with conv
    pool in the kv_cache layout (matches what SYCL produces, so the test can
    diff slot-by-slot directly).
    """
    from vllm.model_executor.layers.fla.ops import (
        fused_sigmoid_gating_delta_rule_update,
    )
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )

    cfg = payload["layer_config"]
    n_actual = int(payload["num_actual_tokens"])
    num_v_heads = cfg["num_v_heads"]
    num_k_heads = cfg["num_k_heads"]
    head_k_dim = cfg["head_k_dim"]
    head_v_dim = cfg["head_v_dim"]
    key_dim = cfg["key_dim"]
    activation = cfg["activation"]

    conv_pool, ssm_pool, remap, _pool_size = _build_dense_pool(payload, device)
    if conv_pool is None:
        return None, None, None, None

    # FLA expects conv state in (..., dim, width-1) layout. The kv_cache stores
    # it as (..., width-1, dim) when is_conv_state_dim_first() is False — that's
    # the layout we capture and that SYCL consumes. Transpose into FLA's layout
    # for the call, then transpose back so the post-state diffs against SYCL's
    # in-place update slot-by-slot.
    dim_first = bool(payload.get("is_conv_state_dim_first") or False)
    conv_pool_for_fla = (
        conv_pool if dim_first else conv_pool.transpose(-1, -2).contiguous()
    )

    mixed_qkv = payload["mixed_qkv"][:n_actual].to(device).clone()
    conv_w = payload["conv_weight"].to(device)
    if conv_w.dim() == 3:
        conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))
    conv_b = payload["conv_bias"]
    if conv_b is not None:
        conv_b = conv_b.to(device)

    spec_idx_dense = (
        _remap_index_tensor(payload["spec_state_indices_tensor"], remap)
        .to(torch.int32)
        .contiguous()
    )
    num_acc = payload["num_accepted_tokens"].to(device).to(torch.int32).contiguous()
    spec_qsl = payload["spec_query_start_loc"].to(device).to(torch.int32).contiguous()
    num_spec_decodes = int(payload["num_spec_decodes"])

    mqp = causal_conv1d_update(
        mixed_qkv,
        conv_pool_for_fla,
        conv_w,
        conv_b,
        activation,
        conv_state_indices=spec_idx_dense[:, 0][:num_spec_decodes].contiguous(),
        num_accepted_tokens=num_acc,
        query_start_loc=spec_qsl,
        max_query_len=spec_idx_dense.size(-1),
        validate_data=False,
    )

    conv_pool_post = (
        conv_pool_for_fla
        if dim_first
        else conv_pool_for_fla.transpose(-1, -2).contiguous()
    )

    q = mqp[:, :key_dim].view(n_actual, num_k_heads, head_k_dim)
    k = mqp[:, key_dim : 2 * key_dim].view(n_actual, num_k_heads, head_k_dim)
    v = mqp[:, 2 * key_dim :].view(n_actual, num_v_heads, head_v_dim)

    fla_core, _ = fused_sigmoid_gating_delta_rule_update(
        A_log=payload["A_log"].to(device),
        a=payload["a"][:n_actual].to(device).unsqueeze(0).contiguous(),
        b=payload["b"][:n_actual].to(device).unsqueeze(0).contiguous(),
        dt_bias=payload["dt_bias"].to(device),
        q=q.unsqueeze(0).contiguous(),
        k=k.unsqueeze(0).contiguous(),
        v=v.unsqueeze(0).contiguous(),
        initial_state=ssm_pool,
        inplace_final_state=True,
        cu_seqlens=spec_qsl[: num_spec_decodes + 1].contiguous(),
        ssm_state_indices=spec_idx_dense,
        num_accepted_tokens=num_acc,
        use_qk_l2norm_in_kernel=True,
    )

    return fla_core.squeeze(0), conv_pool_post, ssm_pool, remap


def _diff(actual, expected, atol, rtol, label):
    actual_cpu = actual.detach().to("cpu")
    expected_cpu = expected.detach().to("cpu")
    assert actual_cpu.shape == expected_cpu.shape, (
        f"{label}: shape mismatch — sycl={tuple(actual_cpu.shape)} "
        f"fla={tuple(expected_cpu.shape)}"
    )
    if actual_cpu.dtype != expected_cpu.dtype:
        actual_cpu = actual_cpu.to(expected_cpu.dtype)
    if not torch.allclose(actual_cpu, expected_cpu, atol=atol, rtol=rtol):
        delta = (actual_cpu.float() - expected_cpu.float()).abs()
        pytest.fail(
            f"{label}: SYCL diverges from FLA oracle.\n"
            f"  max abs diff: {delta.max().item():.4e}\n"
            f"  mean abs diff: {delta.mean().item():.4e}\n"
            f"  shape: {tuple(actual_cpu.shape)}, "
            f"dtype: {actual_cpu.dtype}\n"
            f"  tolerance: atol={atol} rtol={rtol}"
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(params=_TUPLES, ids=lambda p: p.name)
def payload(request):
    p = torch.load(request.param, map_location="cpu", weights_only=False)
    p["__path__"] = str(request.param)
    return p


def test_payload_schema(payload):
    """Loaded payload has the expected schema_version and required keys."""
    assert payload["schema_version"] == _SCHEMA_VERSION, (
        f"{payload['__path__']}: schema_version "
        f"{payload['schema_version']} != expected {_SCHEMA_VERSION}. "
        "Re-capture with the current vllm/_xpu_ops.py."
    )
    missing = [k for k in _REQUIRED_KEYS if k not in payload]
    assert not missing, f"{payload['__path__']}: missing keys {missing}"

    # Sanity: shapes line up
    n_actual = int(payload["num_actual_tokens"])
    assert payload["projected_states_qkvz"].shape[0] >= n_actual
    assert payload["projected_states_ba"].shape[0] >= n_actual
    assert payload["core_attn_out"].shape[0] >= n_actual

    # If we captured slot data, sizes match between pre/post and slot count
    slots = payload["slot_indices"]
    assert payload["conv_state_pre"].shape[0] == slots.numel()
    assert payload["conv_state_post"].shape[0] == slots.numel()
    assert payload["ssm_state_pre"].shape[0] == slots.numel()
    assert payload["ssm_state_post"].shape[0] == slots.numel()


def test_sycl_matches_fla(payload):
    """Run the SYCL gdn_attention op on the captured inputs and diff
    against the captured FLA-oracle outputs.

    Behavior by flavor (Rung 3+):
    - ``non_spec``: always exercises the cross-backend SYCL-vs-FLA diff.
    - ``spec_*`` with ``VLLM_XPU_USE_SYCL_SPEC_GDN`` set: invokes the
      SYCL kernel via the new spec kwargs and diffs at the captured
      tolerance.
    - ``spec_*`` without the env flag: ``xfail`` so CI without the
      rebuilt kernel doesn't regress.
    - ``spec_*_mixed`` (mixed batch): skipped here; Rung 7's
      ``test_sycl_mixed_batch_matches_per_subset`` synthesises the
      mixed-batch path from a non_spec + spec capture pair instead
      of trying to drive a single ambiguous SYCL call.
    """
    flavor: str = payload["flavor"]
    is_spec_capture = flavor != "non_spec"
    is_mixed = flavor.endswith("_mixed")

    if is_mixed:
        pytest.skip(
            "Mixed-batch flavour covered by Rung 7's "
            "test_sycl_mixed_batch_matches_per_subset"
        )
    if is_spec_capture and not _use_sycl_spec_env():
        pytest.xfail(
            "Spec capture requires VLLM_XPU_USE_SYCL_SPEC_GDN=1 (and a "
            "vllm-xpu-kernels build with Rung 3 spec args); rerun with "
            "the env var set after rebuilding. "
            f"flavor={flavor}"
        )

    device = torch.device("xpu")
    conv_pool, ssm_pool, remap, _pool_size = _build_dense_pool(payload, device)
    if conv_pool is None:
        pytest.skip("Capture has no referenced kv_cache slots — nothing to call")

    core_attn_out, z, conv_pool_post, ssm_pool_post = _call_sycl(
        payload, conv_pool, ssm_pool, remap, device
    )

    atol, rtol = 2e-2, 2e-2
    n_actual = int(payload["num_actual_tokens"])

    # z is a slice of the qkvz projection (no recurrence) — the captured value
    # is byte-stable and remains a valid oracle for both spec and non_spec.
    _diff(z[:n_actual], payload["z"][:n_actual], atol, rtol, label="z")

    slots = payload["slot_indices"].to(torch.long)
    slots_dense = remap[slots.to(remap.device)]
    keep = slots_dense > 0

    if is_spec_capture:
        # Spec captures: drive the FLA recurrence inline on the captured inputs
        # (a fresh pool each call) and diff SYCL against that, instead of the
        # captured payload outputs. Tick 10 found the captured FLA outputs are
        # not byte-reproducible across runs — pyref_hpp ≡ FLA-recurrence on the
        # captured inputs but both differ from payload['core_attn_out'] /
        # payload['ssm_state_post'] by ~108 cells.
        oracle_core, oracle_conv_post, oracle_ssm_post, _ = _compute_fla_spec_oracle(
            payload, device
        )
        if oracle_core is None:
            pytest.skip("FLA oracle could not be built — no referenced slots")

        _diff(
            core_attn_out[:n_actual],
            oracle_core[:n_actual],
            atol,
            rtol,
            label="core_attn_out (vs inline FLA oracle)",
        )
        if keep.any():
            _diff(
                conv_pool_post[slots_dense[keep]],
                oracle_conv_post[slots_dense[keep]],
                atol,
                rtol,
                label="conv_state_post (vs inline FLA oracle)",
            )
            _diff(
                ssm_pool_post[slots_dense[keep]],
                oracle_ssm_post[slots_dense[keep]],
                atol,
                rtol,
                label="ssm_state_post (vs inline FLA oracle)",
            )
    else:
        # Non-spec captures: still diff against the captured oracle. Reproducing
        # the FLA prefill path inline needs attn_metadata fields (chunk_indices,
        # chunk_offsets) that the dump doesn't carry yet — separate tick.
        _diff(
            core_attn_out[:n_actual],
            payload["core_attn_out"][:n_actual],
            atol,
            rtol,
            label="core_attn_out",
        )
        if keep.any():
            _diff(
                conv_pool_post[slots_dense[keep]],
                payload["conv_state_post"][keep.to("cpu")],
                atol,
                rtol,
                label="conv_state_post",
            )
            _diff(
                ssm_pool_post[slots_dense[keep]],
                payload["ssm_state_post"][keep.to("cpu")],
                atol,
                rtol,
                label="ssm_state_post",
            )


# ---------------------------------------------------------------------------
# Rung 7: synthetic mixed-batch coverage
# ---------------------------------------------------------------------------


def _layer_configs_compatible(a: dict, b: dict) -> bool:
    keys = (
        "num_k_heads",
        "num_v_heads",
        "head_k_dim",
        "head_v_dim",
        "key_dim",
        "value_dim",
        "tp_size",
        "gqa_interleaved_layout",
        "activation",
    )
    return all(a.get(k) == b.get(k) for k in keys)


@pytest.fixture(
    params=_MIXED_PAIRS, ids=lambda pair: f"{pair[0].name}__plus__{pair[1].name}"
)
def mixed_pair(request):
    ns_path, sp_path = request.param
    ns = torch.load(ns_path, map_location="cpu", weights_only=False)
    sp = torch.load(sp_path, map_location="cpu", weights_only=False)
    ns["__path__"] = str(ns_path)
    sp["__path__"] = str(sp_path)
    return ns, sp


@pytest.mark.skipif(
    not _MIXED_PAIRS,
    reason="No same-layer non_spec+spec pairs to synthesise a mixed batch",
)
def test_sycl_mixed_batch_matches_per_subset(mixed_pair):
    """Synthesise a mixed batch by concatenating a non_spec and a spec capture
    from the same layer, then drive the spec/non-spec split that
    ``_gdn_xpu_spec_sycl_path`` does in production. Each subset must yield the
    same per-token output it would have on its own — verifying both the
    ``index_select`` partition and the ``index_copy_`` scatter.
    """
    if not _use_sycl_spec_env():
        pytest.xfail(
            "Mixed-batch path needs VLLM_XPU_USE_SYCL_SPEC_GDN=1 (and a "
            "vllm-xpu-kernels build with the spec kwargs)"
        )

    ns, sp = mixed_pair
    if not _layer_configs_compatible(ns["layer_config"], sp["layer_config"]):
        pytest.skip("Paired captures have incompatible layer configs")

    cfg = ns["layer_config"]
    device = torch.device("xpu")

    n_ns = int(ns["num_actual_tokens"])
    n_sp = int(sp["num_actual_tokens"])
    if n_ns == 0 or n_sp == 0:
        pytest.skip("One side of the pair is empty — nothing to synthesise")

    n_total = n_ns + n_sp

    # Inputs: non_spec slice first, spec slice second.
    qkvz_total = torch.cat(
        [
            ns["projected_states_qkvz"][:n_ns].to(device),
            sp["projected_states_qkvz"][:n_sp].to(device),
        ],
        dim=0,
    ).contiguous()
    ba_total = torch.cat(
        [
            ns["projected_states_ba"][:n_ns].to(device),
            sp["projected_states_ba"][:n_sp].to(device),
        ],
        dim=0,
    ).contiguous()

    non_spec_token_indx = torch.arange(0, n_ns, device=device, dtype=torch.long)
    spec_token_indx = torch.arange(n_ns, n_total, device=device, dtype=torch.long)

    # has_initial_state: only the non_spec subset uses it.
    has_initial_state = ns.get("has_initial_state")
    if has_initial_state is not None:
        has_initial_state = has_initial_state.to(device)

    conv_pool, ssm_pool, ns_remap, sp_remap = _build_unified_pool(ns, sp, device)
    if conv_pool is None:
        pytest.skip("Pair has no referenced kv_cache slots")

    # Build per-subset slot tensors mapped into the unified pool.
    non_spec_idx_dense = _remap_index_tensor(
        ns["non_spec_state_indices_tensor"], ns_remap
    )
    spec_idx_dense = (
        _remap_index_tensor(sp["spec_state_indices_tensor"], sp_remap)
        .to(torch.int32)
        .contiguous()
    )

    non_spec_qsl = ns["non_spec_query_start_loc"]
    if non_spec_qsl is not None:
        non_spec_qsl = non_spec_qsl.to(device)
    spec_qsl = sp["spec_query_start_loc"]
    if spec_qsl is None:
        pytest.skip("Spec capture missing spec_query_start_loc")
    spec_qsl = spec_qsl.to(device).contiguous()

    num_acc = sp["num_accepted_tokens"]
    if num_acc is None:
        pytest.skip("Spec capture missing num_accepted_tokens")
    num_acc = num_acc.to(device).to(torch.int32).contiguous()

    num_prefills = int(ns["num_prefills"])
    num_decodes = int(ns["num_decodes"])
    num_spec_decodes = int(sp["num_spec_decodes"])

    out_per_token = (
        cfg["num_v_heads"] // cfg["tp_size"],
        cfg["head_v_dim"],
    )
    dtype_core = ns["core_attn_out"].dtype
    dtype_z = ns["z"].dtype

    # Output buffers for the merged batch — pre-fill with NaN so a missing
    # scatter is loud rather than silently masquerading as a near-match.
    core_out = torch.full(
        (n_total,) + out_per_token, float("nan"), dtype=dtype_core, device=device
    )
    z_out = torch.full(
        (n_total,) + out_per_token, float("nan"), dtype=dtype_z, device=device
    )

    conv_w = ns["conv_weight"].to(device)
    if conv_w.dim() == 3:
        conv_w = conv_w.view(conv_w.size(0), conv_w.size(2))
    conv_b = ns["conv_bias"]
    if conv_b is not None:
        conv_b = conv_b.to(device)

    common = dict(
        conv_state=conv_pool,
        ssm_state=ssm_pool,
        conv_weights=conv_w,
        conv_bias=conv_b,
        activation=cfg["activation"],
        A_log=ns["A_log"].to(device),
        dt_bias=ns["dt_bias"].to(device),
        tp_size=cfg["tp_size"],
        reorder_input=not cfg["gqa_interleaved_layout"],
    )

    # === Replicate the mixed-batch path of _gdn_xpu_spec_sycl_path ===

    # Split.
    ns_qkvz = qkvz_total.index_select(0, non_spec_token_indx).contiguous()
    ns_ba = ba_total.index_select(0, non_spec_token_indx).contiguous()
    sp_qkvz = qkvz_total.index_select(0, spec_token_indx).contiguous()
    sp_ba = ba_total.index_select(0, spec_token_indx).contiguous()

    ns_core = torch.empty((n_ns,) + out_per_token, dtype=dtype_core, device=device)
    ns_z = torch.empty_like(ns_core)
    sp_core = torch.empty((n_sp,) + out_per_token, dtype=dtype_core, device=device)
    sp_z = torch.empty_like(sp_core)

    # Non-spec subset call.
    torch.ops._xpu_C.gdn_attention(
        ns_core,
        ns_z,
        ns_qkvz,
        ns_ba,
        cfg["num_k_heads"],
        cfg["num_v_heads"],
        cfg["head_k_dim"],
        cfg["head_v_dim"],
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_qsl,
        non_spec_state_indices_tensor=non_spec_idx_dense,
        num_actual_tokens=n_ns,
        **common,
    )

    # Spec subset call. Sentinel non_spec slot lookup is required for arg
    # validation; the kernel ignores it under IS_SPEC.
    torch.ops._xpu_C.gdn_attention(
        sp_core,
        sp_z,
        sp_qkvz,
        sp_ba,
        cfg["num_k_heads"],
        cfg["num_v_heads"],
        cfg["head_k_dim"],
        cfg["head_v_dim"],
        num_prefills=0,
        num_decodes=num_spec_decodes,
        has_initial_state=None,
        non_spec_query_start_loc=spec_qsl[: num_spec_decodes + 1].contiguous(),
        non_spec_state_indices_tensor=spec_idx_dense[:, 0].contiguous(),
        num_actual_tokens=n_sp,
        spec_state_indices_tensor=spec_idx_dense,
        num_accepted_tokens=num_acc,
        **common,
    )

    # Scatter back.
    core_out.index_copy_(0, non_spec_token_indx, ns_core)
    core_out.index_copy_(0, spec_token_indx, sp_core)
    z_out.index_copy_(0, non_spec_token_indx, ns_z)
    z_out.index_copy_(0, spec_token_indx, sp_z)

    assert not torch.isnan(core_out).any(), "scatter left NaNs in core_attn_out"
    assert not torch.isnan(z_out).any(), "scatter left NaNs in z"

    # === Diff each subset against its own reference ===
    atol, rtol = 2e-2, 2e-2

    # Non-spec subset: captured payload outputs are byte-stable for prefill.
    _diff(
        core_out[:n_ns],
        ns["core_attn_out"][:n_ns],
        atol,
        rtol,
        label="mixed.non_spec.core_attn_out",
    )
    _diff(z_out[:n_ns], ns["z"][:n_ns], atol, rtol, label="mixed.non_spec.z")

    # Spec subset: drive the inline FLA oracle on the spec capture in
    # isolation (separate scratch pool) — same approach as test_sycl_matches_fla.
    oracle_core, oracle_conv_post, oracle_ssm_post, sp_oracle_remap = (
        _compute_fla_spec_oracle(sp, device)
    )
    if oracle_core is not None:
        _diff(
            core_out[n_ns:],
            oracle_core[:n_sp],
            atol,
            rtol,
            label="mixed.spec.core_attn_out (vs inline FLA oracle)",
        )
        _diff(
            z_out[n_ns:],
            sp["z"][:n_sp],
            atol,
            rtol,
            label="mixed.spec.z",
        )

        # Slot-state diffs against the FLA oracle, slot-by-slot.
        sp_slots = sp["slot_indices"].to(torch.long)
        sp_dense_unified = sp_remap[sp_slots.to(device)]
        sp_dense_oracle = sp_oracle_remap[sp_slots.to(device)]
        keep = (sp_dense_unified > 0) & (sp_dense_oracle > 0)
        if keep.any():
            _diff(
                conv_pool[sp_dense_unified[keep]],
                oracle_conv_post[sp_dense_oracle[keep]],
                atol,
                rtol,
                label="mixed.spec.conv_state_post (vs inline FLA oracle)",
            )
            _diff(
                ssm_pool[sp_dense_unified[keep]],
                oracle_ssm_post[sp_dense_oracle[keep]],
                atol,
                rtol,
                label="mixed.spec.ssm_state_post (vs inline FLA oracle)",
            )

    # Non-spec slot states diff against the captured payload.
    ns_slots = ns["slot_indices"].to(torch.long)
    ns_dense = ns_remap[ns_slots.to(device)]
    keep_ns = ns_dense > 0
    if keep_ns.any():
        _diff(
            conv_pool[ns_dense[keep_ns]],
            ns["conv_state_post"][keep_ns.to("cpu")],
            atol,
            rtol,
            label="mixed.non_spec.conv_state_post",
        )
        _diff(
            ssm_pool[ns_dense[keep_ns]],
            ns["ssm_state_post"][keep_ns.to("cpu")],
            atol,
            rtol,
            label="mixed.non_spec.ssm_state_post",
        )


# ---------------------------------------------------------------------------
# Rung 8: gqa_interleaved=True (reorder_input=False) coverage via synthesis
# ---------------------------------------------------------------------------


def test_sycl_reorder_input_false_equivalence(payload):
    """All captures use ``gqa_interleaved_layout=False`` (no Qwen3-Next traffic
    in the dump set). Cover the ``reorder_input=False`` kernel path by
    repacking the captured ``qkvz`` and ``ba`` projections into the per-k_head
    interleaved layout the kernel expects under ``reorder_input=False``, and
    verifying the kernel's output matches the ``reorder_input=True`` path on
    the original layout.

    Tests only the kernel's input-rearrange logic; numerics are inherited from
    ``test_sycl_matches_fla`` against the FLA oracle.
    """
    cfg = payload["layer_config"]
    if cfg["gqa_interleaved_layout"]:
        pytest.skip("Capture is already gqa_interleaved=True — covered elsewhere")

    flavor: str = payload["flavor"]
    is_spec_capture = flavor != "non_spec"
    is_mixed = flavor.endswith("_mixed")
    if is_mixed:
        pytest.skip("Mixed-batch flavour covered by Rung 7 test")
    if is_spec_capture and not _use_sycl_spec_env():
        pytest.xfail(
            "Spec capture requires VLLM_XPU_USE_SYCL_SPEC_GDN=1 to drive the "
            "spec kwargs; rerun with the env set."
        )

    H_k = cfg["num_k_heads"] // cfg["tp_size"]
    H_v = cfg["num_v_heads"] // cfg["tp_size"]
    if H_v % H_k != 0:
        pytest.skip("num_v_heads not divisible by num_k_heads — invalid GQA")

    device = torch.device("xpu")

    # Pass A: original layout, reorder_input=True. Reuses _call_sycl exactly.
    conv_pool_a, ssm_pool_a, remap_a, _ = _build_dense_pool(payload, device)
    if conv_pool_a is None:
        pytest.skip("Capture has no referenced slots")
    core_a, z_a, conv_post_a, ssm_post_a = _call_sycl(
        payload, conv_pool_a, ssm_pool_a, remap_a, device
    )

    # Pass B: permuted layout, reorder_input=False. Patch the cfg in a shallow
    # copy so the kernel call inside _call_sycl flips the flag without
    # mutating the on-disk payload.
    payload_b = dict(payload)
    payload_b["projected_states_qkvz"] = _qkvz_false_to_true(
        payload["projected_states_qkvz"], cfg
    )
    payload_b["projected_states_ba"] = _ba_false_to_true(
        payload["projected_states_ba"], cfg
    )
    cfg_b = dict(cfg)
    cfg_b["gqa_interleaved_layout"] = True
    payload_b["layer_config"] = cfg_b

    conv_pool_b, ssm_pool_b, remap_b, _ = _build_dense_pool(payload_b, device)
    core_b, z_b, conv_post_b, ssm_post_b = _call_sycl(
        payload_b, conv_pool_b, ssm_pool_b, remap_b, device
    )

    n_actual = int(payload["num_actual_tokens"])
    atol, rtol = 2e-2, 2e-2

    _diff(
        core_b[:n_actual],
        core_a[:n_actual],
        atol,
        rtol,
        label="reorder_input=False.core_attn_out vs reorder_input=True",
    )
    _diff(
        z_b[:n_actual],
        z_a[:n_actual],
        atol,
        rtol,
        label="reorder_input=False.z vs reorder_input=True",
    )

    slots = payload["slot_indices"].to(torch.long)
    dense_a = remap_a[slots.to(device)]
    dense_b = remap_b[slots.to(device)]
    keep = (dense_a > 0) & (dense_b > 0)
    if keep.any():
        _diff(
            conv_post_b[dense_b[keep]],
            conv_post_a[dense_a[keep]],
            atol,
            rtol,
            label="reorder_input=False.conv_state_post vs reorder_input=True",
        )
        _diff(
            ssm_post_b[dense_b[keep]],
            ssm_post_a[dense_a[keep]],
            atol,
            rtol,
            label="reorder_input=False.ssm_state_post vs reorder_input=True",
        )
