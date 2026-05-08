# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch
from vllm_xpu_kernels.flash_attn_interface import flash_attn_varlen_func

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

if TYPE_CHECKING:

    def register_fake(fn):
        return lambda name: fn
else:
    try:
        from torch.library import register_fake
    except ImportError:
        from torch.library import impl_abstract as register_fake

if hasattr(torch.ops._xpu_C, "fp8_gemm"):

    @register_fake("_xpu_C::fp8_gemm")
    def _fp8_gemm_fake(
        q_input: torch.Tensor,
        q_weight: torch.Tensor,
        out_dtype: torch.dtype,
        input_scales: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = q_input.view(-1, q_input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=out_dtype, device=q_input.device)


if hasattr(torch.ops._xpu_C, "fp8_gemm_w8a16"):

    @register_fake("_xpu_C::fp8_gemm_w8a16")
    def _fp8_gemm_w8a16_fake(
        input: torch.Tensor,
        q_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = input.view(-1, input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=input.dtype, device=input.device)


if hasattr(torch.ops._xpu_C, "int4_gemm_w4a8"):

    @register_fake("_xpu_C::int4_gemm_w4a8")
    def _int4_gemm_w4a8_fake(
        input: torch.Tensor,
        input_scales: torch.Tensor,
        input_zero_points: torch.Tensor,
        q_weight: torch.Tensor,
        weight_scale: torch.Tensor,
        weight_zp: torch.Tensor,
        group_size: int,
        g_idx: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = input.view(-1, input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=torch.float16, device=input.device)


if hasattr(torch.ops._xpu_C, "int4_gemm_w4a16"):

    @register_fake("_xpu_C::int4_gemm_w4a16")
    def _int4_gemm_w4a16_fake(
        input: torch.Tensor,
        q_weight: torch.Tensor,
        bias: torch.Tensor | None,
        weight_scale: torch.Tensor,
        qzeros: torch.Tensor,
        group_size: int,
        group_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        input_2d = input.view(-1, input.shape[-1])
        M = input_2d.size(0)
        N = q_weight.size(1)
        return torch.empty((M, N), dtype=input.dtype, device=input.device)


def _gdn_xpu_spec_python_path(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
    layer,
) -> None:
    """Spec-decode fallback for XPU GDN: route through the FLA Triton
    flow used by forward_cuda.

    The fused SYCL gdn_attention kernel has no spec-aware path
    (per-candidate state evolution + rollback on rejection driven by
    num_accepted_tokens). The custom op is opaque to torch.compile, so
    branching on attn_metadata.spec_sequence_masks at runtime here is
    safe — it is not baked into the compiled graph the way a check in
    forward_xpu's body would be.

    Mirrors the input prep in GatedDeltaNetAttention.forward_cuda
    (gdn_linear_attn.py:543-564) and then defers to
    GatedDeltaNetAttention._forward_core which contains the spec-aware
    Python branching using FLA Triton kernels with IS_SPEC_DECODING.
    """
    from einops import rearrange

    if layer.gqa_interleaved_layout:
        # Qwen3-Next: unpack the interleaved GQA layout
        query, key, value, z_split, b, a = layer.fix_query_key_value_ordering(
            projected_states_qkvz, projected_states_ba
        )
        query, key, value = map(
            lambda x: rearrange(x, "l p d -> l (p d)"), (query, key, value)
        )
        mixed_qkv = torch.cat((query, key, value), dim=-1)
    else:
        # Qwen3.5: weights are already in [q, k, v, z] and [b, a] order
        qkv_size = (layer.key_dim * 2 + layer.value_dim) // layer.tp_size
        z_size = layer.value_dim // layer.tp_size
        mixed_qkv, z_split = projected_states_qkvz.split([qkv_size, z_size], dim=-1)
        z_split = z_split.reshape(z_split.size(0), -1, layer.head_v_dim)
        b, a = projected_states_ba.chunk(2, dim=-1)
        b = b.contiguous()
        a = a.contiguous()

    # Surface z to the caller's z buffer so the subsequent norm in
    # forward_xpu sees the correct gating tensor. Shapes match because
    # forward_xpu allocated z = empty_like(core_attn_out) which is
    # [num_tokens, num_v_heads / tp_size, head_v_dim] — same as z_split.
    z.copy_(z_split)

    dump_dir = _spec_gdn_dump_dir()
    if dump_dir is not None and _spec_gdn_can_dump():
        _spec_gdn_dump_call(
            dump_dir,
            layer,
            mixed_qkv,
            b,
            a,
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
        )
        return

    # _forward_core writes into core_attn_out. It reads attn_metadata
    # from forward_context, so spec_sequence_masks / num_accepted_tokens
    # are already in scope.
    layer._forward_core(
        mixed_qkv=mixed_qkv,
        b=b,
        a=a,
        core_attn_out=core_attn_out,
    )


# ---------------------------------------------------------------------------
# Spec-aware SYCL gdn_attention capture harness (Rung 1 of the test ladder
# documented in vllm/.spec-gdn-progress.md).
#
# When VLLM_XPU_DUMP_SPEC_GDN=<dir> is set in addition to VLLM_XPU_FORCE_FLA_GDN=1,
# every spec-or-non-spec call that reaches _gdn_xpu_spec_python_path is
# serialized to <dir>/tuple_<sanitized-prefix>_<step>_<flavor>.pt as a dict
# containing the inputs to the SYCL kernel, the FLA-path outputs, and the
# pre/post slices of the kv_cache pool indexed by the call. The replay harness
# (Rung 2) consumes these dumps to drive a standalone pytest that diffs the
# SYCL spec path against the FLA oracle.
#
# The capture path never alters numerical behaviour: it only takes snapshots
# around the existing layer._forward_core call. Disabled (zero overhead) when
# VLLM_XPU_DUMP_SPEC_GDN is unset.
# ---------------------------------------------------------------------------

_VLLM_XPU_DUMP_SPEC_GDN_INIT = False
_VLLM_XPU_DUMP_SPEC_GDN: str | None = None
_VLLM_XPU_DUMP_SPEC_GDN_MAX: int = 0
_SPEC_GDN_DUMP_COUNT: int = 0
_SPEC_GDN_STEP_COUNTER: dict[str, int] = {}


def _spec_gdn_dump_dir() -> str | None:
    global _VLLM_XPU_DUMP_SPEC_GDN_INIT
    global _VLLM_XPU_DUMP_SPEC_GDN
    global _VLLM_XPU_DUMP_SPEC_GDN_MAX
    if _VLLM_XPU_DUMP_SPEC_GDN_INIT:
        return _VLLM_XPU_DUMP_SPEC_GDN
    import os

    path = os.environ.get("VLLM_XPU_DUMP_SPEC_GDN", "").strip()
    if path:
        os.makedirs(path, exist_ok=True)
        _VLLM_XPU_DUMP_SPEC_GDN = path
        try:
            _VLLM_XPU_DUMP_SPEC_GDN_MAX = int(
                os.environ.get("VLLM_XPU_DUMP_SPEC_GDN_MAX", "200")
            )
        except ValueError:
            _VLLM_XPU_DUMP_SPEC_GDN_MAX = 200
        logger.warning(
            "VLLM_XPU_DUMP_SPEC_GDN enabled — capturing GDN spec tuples to %s "
            "(cap=%d). Disable for production runs.",
            path,
            _VLLM_XPU_DUMP_SPEC_GDN_MAX,
        )
    else:
        _VLLM_XPU_DUMP_SPEC_GDN = None
    _VLLM_XPU_DUMP_SPEC_GDN_INIT = True
    return _VLLM_XPU_DUMP_SPEC_GDN


def _spec_gdn_can_dump() -> bool:
    return _SPEC_GDN_DUMP_COUNT < _VLLM_XPU_DUMP_SPEC_GDN_MAX


def _spec_gdn_step(prefix: str) -> int:
    n = _SPEC_GDN_STEP_COUNTER.get(prefix, 0)
    _SPEC_GDN_STEP_COUNTER[prefix] = n + 1
    return n


def _spec_gdn_sanitize(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name)


def _spec_gdn_flavor(attn_md) -> str:
    masks = getattr(attn_md, "spec_sequence_masks", None)
    if masks is None:
        return "non_spec"
    nat = getattr(attn_md, "num_accepted_tokens", None)
    spec_idx = getattr(attn_md, "spec_state_indices_tensor", None)
    if nat is None or spec_idx is None or spec_idx.numel() == 0:
        return "spec_unknown"
    K = spec_idx.size(-1)
    nat_cpu = nat.detach().to("cpu").tolist()
    if not nat_cpu:
        return "spec_empty"
    mn = min(nat_cpu)
    mx = max(nat_cpu)
    has_non_spec = (
        getattr(attn_md, "num_prefills", 0) + getattr(attn_md, "num_decodes", 0)
    ) > 0
    base = f"spec_K{K}_min{mn}_max{mx}"
    return base + ("_mixed" if has_non_spec else "")


_LAYER_CONFIG_KEYS = (
    "num_k_heads",
    "num_v_heads",
    "head_k_dim",
    "head_v_dim",
    "key_dim",
    "value_dim",
    "tp_size",
    "gqa_interleaved_layout",
    "activation",
    "prefix",
)


def _spec_gdn_layer_config(layer) -> dict:
    return {k: getattr(layer, k, None) for k in _LAYER_CONFIG_KEYS}


def _opt_to_cpu(t):
    if t is None:
        return None
    if isinstance(t, torch.Tensor):
        return t.detach().to("cpu").clone()
    return t


def _spec_gdn_dump_call(
    dump_dir: str,
    layer,
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
) -> None:
    """Snapshot inputs around layer._forward_core, run it, snapshot outputs,
    serialize to a .pt tuple. See module-level docstring for layout.
    """
    global _SPEC_GDN_DUMP_COUNT
    import os

    from vllm.forward_context import get_forward_context

    forward_context = get_forward_context()
    attn_md_raw = forward_context.attn_metadata
    if attn_md_raw is None:
        # V1 profile run — _forward_core warms up prefill kernels; nothing to
        # capture because there is no real attn_metadata.
        layer._forward_core(
            mixed_qkv=mixed_qkv, b=b, a=a, core_attn_out=core_attn_out
        )
        return

    assert isinstance(attn_md_raw, dict)
    attn_md = attn_md_raw[layer.prefix]

    spec_idx = getattr(attn_md, "spec_state_indices_tensor", None)
    non_spec_idx = getattr(attn_md, "non_spec_state_indices_tensor", None)

    slot_parts = []
    if spec_idx is not None and spec_idx.numel() > 0:
        slot_parts.append(spec_idx.flatten())
    if non_spec_idx is not None and non_spec_idx.numel() > 0:
        slot_parts.append(non_spec_idx.flatten())
    if slot_parts:
        all_slots = torch.cat(slot_parts).unique()
        all_slots = all_slots[all_slots >= 0]
    else:
        all_slots = torch.empty(0, dtype=torch.long, device=core_attn_out.device)

    conv_pool = layer.kv_cache[0]
    ssm_pool = layer.kv_cache[1]
    if all_slots.numel():
        conv_state_pre = conv_pool[all_slots].detach().clone()
        ssm_state_pre = ssm_pool[all_slots].detach().clone()
    else:
        conv_state_pre = conv_pool[:0].detach().clone()
        ssm_state_pre = ssm_pool[:0].detach().clone()

    # The actual FLA work — must run inline so kv_cache mutations persist for
    # the rest of the forward pass.
    layer._forward_core(mixed_qkv=mixed_qkv, b=b, a=a, core_attn_out=core_attn_out)

    if all_slots.numel():
        conv_state_post = conv_pool[all_slots].detach().clone()
        ssm_state_post = ssm_pool[all_slots].detach().clone()
    else:
        conv_state_post = conv_pool[:0].detach().clone()
        ssm_state_post = ssm_pool[:0].detach().clone()

    flavor = _spec_gdn_flavor(attn_md)
    prefix = getattr(layer, "prefix", "unknown")
    step = _spec_gdn_step(prefix)
    sanitized = _spec_gdn_sanitize(prefix)

    try:
        from vllm.model_executor.layers.mamba.mamba_utils import (
            is_conv_state_dim_first,
        )

        conv_dim_first = bool(is_conv_state_dim_first())
    except Exception:
        conv_dim_first = None

    bias = layer.conv1d.bias
    payload = {
        # schema_version 2: added projected_states_qkvz/ba so the SYCL
        # custom op can be driven directly from the payload (the SYCL
        # kernel consumes pre-rearrange projected states, not mixed_qkv).
        "schema_version": 2,
        "layer_prefix": prefix,
        "step": step,
        "flavor": flavor,
        # SYCL kernel inputs (pre-rearrange, before _gdn_xpu_spec_python_path
        # splits/concats them).
        "projected_states_qkvz": projected_states_qkvz.detach().to("cpu").clone(),
        "projected_states_ba": projected_states_ba.detach().to("cpu").clone(),
        # FLA-pipeline inputs (post-rearrange) — kept so a future test can
        # drive layer._forward_core directly from the payload if needed.
        "mixed_qkv": mixed_qkv.detach().to("cpu").clone(),
        "b": b.detach().to("cpu").clone(),
        "a": a.detach().to("cpu").clone(),
        # state slot snapshots (only the slots actually referenced)
        "slot_indices": all_slots.detach().to("cpu").clone(),
        "conv_state_pre": conv_state_pre.to("cpu"),
        "conv_state_post": conv_state_post.to("cpu"),
        "ssm_state_pre": ssm_state_pre.to("cpu"),
        "ssm_state_post": ssm_state_post.to("cpu"),
        # FLA outputs (post-call, in place)
        "core_attn_out": core_attn_out.detach().to("cpu").clone(),
        "z": z.detach().to("cpu").clone(),
        # attn_metadata fields needed to drive the SYCL kernel
        "spec_sequence_masks": _opt_to_cpu(
            getattr(attn_md, "spec_sequence_masks", None)
        ),
        "spec_query_start_loc": _opt_to_cpu(
            getattr(attn_md, "spec_query_start_loc", None)
        ),
        "non_spec_query_start_loc": _opt_to_cpu(
            getattr(attn_md, "non_spec_query_start_loc", None)
        ),
        "spec_token_indx": _opt_to_cpu(getattr(attn_md, "spec_token_indx", None)),
        "non_spec_token_indx": _opt_to_cpu(
            getattr(attn_md, "non_spec_token_indx", None)
        ),
        "spec_state_indices_tensor": _opt_to_cpu(spec_idx),
        "non_spec_state_indices_tensor": _opt_to_cpu(non_spec_idx),
        "num_accepted_tokens": _opt_to_cpu(
            getattr(attn_md, "num_accepted_tokens", None)
        ),
        "has_initial_state": _opt_to_cpu(
            getattr(attn_md, "has_initial_state", None)
        ),
        "num_actual_tokens": int(getattr(attn_md, "num_actual_tokens", 0) or 0),
        "num_prefills": int(getattr(attn_md, "num_prefills", 0) or 0),
        "num_decodes": int(getattr(attn_md, "num_decodes", 0) or 0),
        "num_spec_decodes": int(getattr(attn_md, "num_spec_decodes", 0) or 0),
        # layer config + weights / bias snapshots (small relative to states)
        "layer_config": _spec_gdn_layer_config(layer),
        "conv_weight": layer.conv1d.weight.detach().to("cpu").clone(),
        "conv_bias": (bias.detach().to("cpu").clone() if bias is not None else None),
        "A_log": layer.A_log.detach().to("cpu").clone(),
        "dt_bias": layer.dt_bias.detach().to("cpu").clone(),
        "is_conv_state_dim_first": conv_dim_first,
    }

    fname = f"tuple_{sanitized}_{step:06d}_{flavor}.pt"
    torch.save(payload, os.path.join(dump_dir, fname))
    _SPEC_GDN_DUMP_COUNT += 1


_VLLM_XPU_FORCE_FLA_GDN = None


def _force_fla_gdn() -> bool:
    """One-shot lookup of VLLM_XPU_FORCE_FLA_GDN env var.

    When set, every GDN call (spec AND non-spec) is routed through the
    FLA Triton flow used by forward_cuda. Useful for byte-equality
    validation: comparing baseline+force_fla against mtp-k3+force_fla
    isolates spec-path correctness from cross-backend bf16 drift
    between SYCL and FLA Triton.
    """
    global _VLLM_XPU_FORCE_FLA_GDN
    if _VLLM_XPU_FORCE_FLA_GDN is None:
        import os

        _VLLM_XPU_FORCE_FLA_GDN = os.environ.get(
            "VLLM_XPU_FORCE_FLA_GDN", ""
        ).lower() in ("1", "true", "yes", "on")
    return _VLLM_XPU_FORCE_FLA_GDN


_VLLM_XPU_USE_SYCL_SPEC_GDN: bool | None = None
_SYCL_SPEC_GDN_OP_VALIDATED = False


def _use_sycl_spec_gdn() -> bool:
    """One-shot lookup of VLLM_XPU_USE_SYCL_SPEC_GDN env var.

    Tri-state:
      - unset / ``0`` (default): spec batches go to the FLA Triton oracle.
      - ``auto``: try SYCL spec path; on missing op or schema mismatch,
        log once and fall back to FLA. For staged rollout / forward-compat.
      - ``1`` / ``true``: SYCL spec path is mandatory. If the op is
        missing (kernel built with ``GDN_KERNELS_ENABLED=OFF``) or its
        schema lacks the spec args (older kernel), fail loudly at
        startup with a build-fix hint. No silent fallback — the whole
        point of this opt-in is to actually exercise SYCL.

    ``VLLM_XPU_FORCE_FLA_GDN=1`` still wins regardless.
    """
    global _VLLM_XPU_USE_SYCL_SPEC_GDN
    if _VLLM_XPU_USE_SYCL_SPEC_GDN is None:
        import os

        raw = os.environ.get("VLLM_XPU_USE_SYCL_SPEC_GDN", "").lower()
        _VLLM_XPU_USE_SYCL_SPEC_GDN = raw in ("1", "true", "yes", "on", "auto")
    return _VLLM_XPU_USE_SYCL_SPEC_GDN


def _sycl_spec_gdn_strict() -> bool:
    """True iff VLLM_XPU_USE_SYCL_SPEC_GDN is a hard opt-in (not 'auto')."""
    import os

    return os.environ.get("VLLM_XPU_USE_SYCL_SPEC_GDN", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _validate_sycl_spec_gdn_op() -> None:
    """Confirm torch.ops._xpu_C.gdn_attention exists and accepts the spec
    kwargs. Called lazily on first use of the SYCL spec path. Raises
    RuntimeError with a build-fix hint when strict mode is set; under
    'auto' the caller catches and falls back."""
    global _SYCL_SPEC_GDN_OP_VALIDATED
    if _SYCL_SPEC_GDN_OP_VALIDATED:
        return
    if not hasattr(torch.ops._xpu_C, "gdn_attention"):
        raise RuntimeError(
            "torch.ops._xpu_C.gdn_attention is not registered. The "
            "vllm-xpu-kernels build was produced with "
            "GDN_KERNELS_ENABLED=OFF. Rebuild with "
            "GDN_KERNELS_ENABLED=ON, e.g. "
            "`GDN_KERNELS_ENABLED=ON vllm-xpu-build` from the flake."
        )
    schema = str(torch.ops._xpu_C.gdn_attention.default._schema)
    if "spec_state_indices_tensor" not in schema:
        raise RuntimeError(
            "torch.ops._xpu_C.gdn_attention is missing the spec args "
            "(spec_state_indices_tensor / num_accepted_tokens). The "
            "vllm-xpu-kernels build is older than the spec-decoding "
            "patch. Rebuild from the current vllm-xpu-kernels HEAD."
        )
    _SYCL_SPEC_GDN_OP_VALIDATED = True


def _gdn_xpu_spec_sycl_path(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
    layer,
    attn_metadata,
) -> None:
    """Dispatch the SYCL gdn_attention op for a spec batch.

    Mirrors the spec/non-spec split that ``_forward_core`` does for the
    FLA Triton path (gdn_linear_attn.py:815-1006): when both kinds of
    sequences are present in the batch, we issue **two** kernel calls
    (one per subset) and scatter the outputs back into the caller's
    ``core_attn_out`` / ``z`` buffers using the captured
    ``spec_token_indx`` / ``non_spec_token_indx`` indices.

    Pre-condition: ``VLLM_XPU_FORCE_FLA_GDN`` is unset and the SYCL op
    accepts the new ``spec_state_indices_tensor`` / ``num_accepted_tokens``
    kwargs (validated by the caller via try/except).
    """
    spec_token_indx = attn_metadata.spec_token_indx
    non_spec_token_indx = attn_metadata.non_spec_token_indx
    spec_query_start_loc = attn_metadata.spec_query_start_loc
    non_spec_query_start_loc = attn_metadata.non_spec_query_start_loc
    spec_state_indices_tensor = attn_metadata.spec_state_indices_tensor
    non_spec_state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
    num_accepted_tokens = attn_metadata.num_accepted_tokens
    has_initial_state = attn_metadata.has_initial_state
    num_actual_tokens = int(attn_metadata.num_actual_tokens)
    num_prefills = int(attn_metadata.num_prefills)
    num_decodes = int(attn_metadata.num_decodes)
    num_spec_decodes = int(attn_metadata.num_spec_decodes or 0)

    has_non_spec = (num_prefills + num_decodes) > 0

    core_attn_out = core_attn_out[:num_actual_tokens]
    z = z[:num_actual_tokens]
    qkvz = projected_states_qkvz[:num_actual_tokens]
    ba = projected_states_ba[:num_actual_tokens]

    conv_weights = layer.conv1d.weight.view(
        layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
    )

    common_kwargs = dict(
        conv_state=layer.kv_cache[0],
        ssm_state=layer.kv_cache[1],
        conv_weights=conv_weights,
        conv_bias=layer.conv1d.bias,
        activation=layer.activation,
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        tp_size=layer.tp_size,
        reorder_input=not layer.gqa_interleaved_layout,
    )

    if not has_non_spec:
        # Spec-only batch: single kernel call against the spec slot ring.
        torch.ops._xpu_C.gdn_attention(
            core_attn_out,
            z,
            qkvz.contiguous(),
            ba.contiguous(),
            layer.num_k_heads,
            layer.num_v_heads,
            layer.head_k_dim,
            layer.head_v_dim,
            num_prefills=0,
            num_decodes=num_spec_decodes,
            has_initial_state=None,
            non_spec_query_start_loc=spec_query_start_loc[
                : num_spec_decodes + 1
            ].contiguous(),
            non_spec_state_indices_tensor=spec_state_indices_tensor[:, 0]
            .contiguous(),
            num_actual_tokens=num_actual_tokens,
            spec_state_indices_tensor=spec_state_indices_tensor,
            num_accepted_tokens=num_accepted_tokens,
            **common_kwargs,
        )
        return

    # Mixed batch: split, run twice, scatter.
    spec_qkvz = qkvz.index_select(0, spec_token_indx).contiguous()
    spec_ba = ba.index_select(0, spec_token_indx).contiguous()
    non_spec_qkvz = qkvz.index_select(0, non_spec_token_indx).contiguous()
    non_spec_ba = ba.index_select(0, non_spec_token_indx).contiguous()

    n_spec = spec_qkvz.size(0)
    n_non_spec = non_spec_qkvz.size(0)

    out_shape_per_token = (
        layer.num_v_heads // layer.tp_size,
        layer.head_v_dim,
    )
    spec_core = torch.empty(
        (n_spec,) + out_shape_per_token,
        dtype=core_attn_out.dtype,
        device=core_attn_out.device,
    )
    spec_z = torch.empty_like(spec_core)
    non_spec_core = torch.empty(
        (n_non_spec,) + out_shape_per_token,
        dtype=core_attn_out.dtype,
        device=core_attn_out.device,
    )
    non_spec_z = torch.empty_like(non_spec_core)

    # Non-spec subset.
    torch.ops._xpu_C.gdn_attention(
        non_spec_core,
        non_spec_z,
        non_spec_qkvz,
        non_spec_ba,
        layer.num_k_heads,
        layer.num_v_heads,
        layer.head_k_dim,
        layer.head_v_dim,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        has_initial_state=has_initial_state,
        non_spec_query_start_loc=non_spec_query_start_loc,
        non_spec_state_indices_tensor=non_spec_state_indices_tensor,
        num_actual_tokens=n_non_spec,
        **common_kwargs,
    )

    # Spec subset.
    torch.ops._xpu_C.gdn_attention(
        spec_core,
        spec_z,
        spec_qkvz,
        spec_ba,
        layer.num_k_heads,
        layer.num_v_heads,
        layer.head_k_dim,
        layer.head_v_dim,
        num_prefills=0,
        num_decodes=num_spec_decodes,
        has_initial_state=None,
        non_spec_query_start_loc=spec_query_start_loc[
            : num_spec_decodes + 1
        ].contiguous(),
        non_spec_state_indices_tensor=spec_state_indices_tensor[:, 0]
        .contiguous(),
        num_actual_tokens=n_spec,
        spec_state_indices_tensor=spec_state_indices_tensor,
        num_accepted_tokens=num_accepted_tokens,
        **common_kwargs,
    )

    core_attn_out.index_copy_(0, non_spec_token_indx, non_spec_core)
    core_attn_out.index_copy_(0, spec_token_indx, spec_core)
    z.index_copy_(0, non_spec_token_indx, non_spec_z)
    z.index_copy_(0, spec_token_indx, spec_z)


def _gdn_attention_core_xpu_impl(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
    layer_name: str,
) -> None:
    """Custom op wrapping the XPU SYCL GDN kernel for torch.compile."""
    from vllm.forward_context import get_forward_context
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    forward_context = get_forward_context()
    self = forward_context.no_compile_layers[layer_name]
    attn_metadata_raw = forward_context.attn_metadata

    if attn_metadata_raw is None:
        return

    assert isinstance(attn_metadata_raw, dict)
    attn_metadata = attn_metadata_raw[self.prefix]
    assert isinstance(attn_metadata, GDNAttentionMetadata)

    is_spec_batch = attn_metadata.spec_sequence_masks is not None  # type: ignore[attr-defined]

    if is_spec_batch and _use_sycl_spec_gdn() and not _force_fla_gdn():
        # Spec batches dispatch to the SYCL kernel via the Python-side
        # spec/non-spec split. Strict mode (=1) raises on op/schema
        # mismatch — explicit opt-in shouldn't silently degrade.
        # 'auto' mode catches build-availability errors and falls back
        # to FLA for staged rollout; runtime errors from the kernel
        # itself always propagate so we don't paper over real bugs.
        if _sycl_spec_gdn_strict():
            _validate_sycl_spec_gdn_op()
            _gdn_xpu_spec_sycl_path(
                core_attn_out,
                z,
                projected_states_qkvz,
                projected_states_ba,
                self,
                attn_metadata,
            )
            return
        try:
            _validate_sycl_spec_gdn_op()
        except RuntimeError as e:
            logger.warning_once(
                "SYCL gdn_attention spec path unavailable (%s); "
                "falling back to FLA Triton. Set "
                "VLLM_XPU_USE_SYCL_SPEC_GDN=1 to make this fatal.",
                e,
            )
        else:
            _gdn_xpu_spec_sycl_path(
                core_attn_out,
                z,
                projected_states_qkvz,
                projected_states_ba,
                self,
                attn_metadata,
            )
            return

    if is_spec_batch or _force_fla_gdn():
        # See _gdn_xpu_spec_python_path docstring. The custom op is
        # opaque to torch.compile, so this runtime branch survives
        # graph capture; the equivalent check in forward_xpu's body
        # would be specialized at trace time when attn_metadata is None.
        # When VLLM_XPU_FORCE_FLA_GDN=1, the same path is used even for
        # non-spec batches, for verification.
        _gdn_xpu_spec_python_path(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self,
        )
        return

    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )

    torch.ops._xpu_C.gdn_attention(
        core_attn_out,
        z,
        projected_states_qkvz,
        projected_states_ba,
        self.num_k_heads,
        self.num_v_heads,
        self.head_k_dim,
        self.head_v_dim,
        conv_state=self.kv_cache[0],
        ssm_state=self.kv_cache[1],
        conv_weights=conv_weights,
        conv_bias=self.conv1d.bias,
        activation=self.activation,
        A_log=self.A_log,
        dt_bias=self.dt_bias,
        num_prefills=attn_metadata.num_prefills,  # type: ignore[attr-defined]
        num_decodes=attn_metadata.num_decodes,  # type: ignore[attr-defined]
        has_initial_state=attn_metadata.has_initial_state,  # type: ignore[attr-defined]
        non_spec_query_start_loc=attn_metadata.non_spec_query_start_loc,  # type: ignore[attr-defined]
        non_spec_state_indices_tensor=attn_metadata.non_spec_state_indices_tensor,  # type: ignore[attr-defined]
        num_actual_tokens=attn_metadata.num_actual_tokens,  # type: ignore[attr-defined]
        tp_size=self.tp_size,
        reorder_input=not self.gqa_interleaved_layout,
    )


def _gdn_attention_core_xpu_fake(
    core_attn_out: torch.Tensor,
    z: torch.Tensor,
    projected_states_qkvz: torch.Tensor,
    projected_states_ba: torch.Tensor,
    layer_name: str,
) -> None:
    return


def _xpu_ops_deepseek_scaling_rope_impl(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    offsets: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert key is not None
    return torch.ops._xpu_C.deepseek_scaling_rope(
        positions, query, key, offsets, cos_sin_cache, rotary_dim, is_neox_style
    )


def _xpu_ops_deepseek_scaling_rope_fake(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor | None,
    offsets: torch.Tensor | None,
    cos_sin_cache: torch.Tensor | None,
    rotary_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return query, key


def _topk_topp_sample_impl(
    random_sampled: torch.Tensor,
    logits_to_return: torch.Tensor | None,
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    logprobs_mode: str,
    seeds: torch.Tensor | None,
    lambda_: float = 1.0,
) -> None:
    torch.ops._xpu_C.topk_topp_sampler(
        random_sampled, logits_to_return, logits, k, p, logprobs_mode, seeds, lambda_
    )
    return


def _topk_topp_sample_fake(
    random_sampled: torch.Tensor,
    logits_to_return: torch.Tensor | None,
    logits: torch.Tensor,
    k: torch.Tensor | None,
    p: torch.Tensor | None,
    logprobs_mode: str,
    seeds: torch.Tensor | None,
    lambda_: float = 1.0,
) -> None:
    return


def _xpu_mxfp8_quantize_impl(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    MXFP8_BLOCK_SIZE = 32
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0
    if dtype is not None:
        assert dtype in (torch.float8_e4m3fn, torch.float8_e5m2), (
            f"Unsupported dtype for xpu_mxfp8_quantize: {dtype}. "
            f"Expected torch.float8_e4m3fn or torch.float8_e5m2."
        )
    else:
        dtype = current_platform.fp8_dtype()

    finfo = torch.finfo(dtype)
    fp8_min = finfo.min
    fp8_max = finfo.max
    eps = 1e-10

    x_q = torch.empty_like(x, device=x.device, dtype=dtype)
    shape = x.shape[:-1] + (x.shape[-1] // MXFP8_BLOCK_SIZE,)
    x_s = torch.empty(shape, device=x.device, dtype=torch.float32)
    torch.ops._C.per_token_group_fp8_quant(
        x, x_q, x_s, MXFP8_BLOCK_SIZE, eps, fp8_min, fp8_max, True
    )
    x_s = x_s.to(torch.float8_e8m0fnu)
    return x_q, x_s


def _xpu_mxfp8_quantize_fake(
    x: torch.Tensor, dtype: torch.dtype | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    if dtype is None:
        dtype = current_platform.fp8_dtype()

    MXFP8_BLOCK_SIZE = 32

    shape = x.shape[:-1] + (x.shape[-1] // MXFP8_BLOCK_SIZE,)
    x_s = torch.zeros(shape, device=x.device, dtype=torch.float32)

    return x.to(dtype), x_s.to(torch.float8_e8m0fnu)


def _xpu_mxfp4_quantize_impl(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    MXFP4_BLOCK_SIZE = 32
    eps = 1e-10
    assert x.ndim == 2, "input must be 2-D"
    assert x.shape[-1] % MXFP4_BLOCK_SIZE == 0, (
        f"last dimension {x.shape[-1]} must be divisible by group_size "
        f"{MXFP4_BLOCK_SIZE}"
    )
    assert x.is_contiguous(), "input groups must be contiguous"

    M, N = x.shape

    # Packed FP4 output: two nibbles per byte
    x_q = torch.empty(M, N // 2, device=x.device, dtype=torch.uint8)
    x_s = torch.empty(M, N // MXFP4_BLOCK_SIZE, device=x.device, dtype=torch.float32)

    torch.ops._C.per_token_group_quant_mxfp4(x, x_q, x_s, MXFP4_BLOCK_SIZE, eps)

    x_q = x_q.view(torch.float4_e2m1fn_x2)
    x_s = x_s.to(dtype=torch.float8_e8m0fnu, memory_format=torch.preserve_format)
    return x_q, x_s


def _xpu_mxfp4_quantize_fake(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    MXFP4_BLOCK_SIZE = 32
    M, N = x.shape

    # Packed FP4 output: two nibbles per byte
    x_q = torch.empty(M, N // 2, device=x.device, dtype=torch.uint8)
    x_s = torch.empty(M, N // MXFP4_BLOCK_SIZE, device=x.device, dtype=torch.float32)

    x_q = x_q.view(torch.float4_e2m1fn_x2)
    x_s = x_s.to(dtype=torch.float8_e8m0fnu, memory_format=torch.preserve_format)
    return x_q, x_s


# Global flag to ensure ops are registered only once
_OPS_REGISTERED = False


class xpu_ops:
    @staticmethod
    @torch.compile
    def dynamic_per_token_int8_quant_ref(
        input: torch.Tensor, use_sym_quant: bool, bits: int
    ):
        original_sizes = input.size()
        # view is not safe in torch.compile if input is not contiguous
        input = input.reshape(
            -1, original_sizes[-1]
        )  # Flatten except for the last dimension
        qmin = -(2 ** (bits - 1)) if use_sym_quant else 0
        qmax = 2 ** (bits - 1) - 1 if use_sym_quant else 2**bits - 1
        min_val = torch.min(input, dim=-1)[0].to(dtype=torch.float32).unsqueeze(-1)
        max_val = torch.max(input, dim=-1)[0].to(dtype=torch.float32).unsqueeze(-1)
        if use_sym_quant:
            scale = (
                torch.maximum(torch.abs(min_val), torch.abs(max_val)) / qmax
            ).clamp(min=1e-5)
            zero_point = torch.zeros_like(scale).to(dtype=torch.int32)
        else:
            scale = ((max_val - min_val) / qmax).clamp(min=1e-5)
            zero_point = -1 * torch.round(min_val / scale).to(dtype=torch.int32)
        scale = scale.to(dtype=input.dtype)
        quantized = torch.clamp(
            torch.round(input / scale.to(dtype=torch.float32) + zero_point),
            qmin,
            qmax,
        ).to(dtype=torch.int8 if use_sym_quant else torch.uint8)
        return (
            quantized.view(original_sizes),
            scale.view(original_sizes[:-1] + (1,)),
            zero_point.view(original_sizes[:-1] + (1,)),
        )

    @staticmethod
    def flash_attn_varlen_func(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float | None = None,
        causal: bool = False,
        out: torch.Tensor | None = None,
        block_table: torch.Tensor | None = None,
        alibi_slopes: torch.Tensor | None = None,
        window_size: list[int] | None = None,
        softcap: float | None = 0.0,
        seqused_k: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        # passed in qwen vl
        dropout_p: float = 0.0,
        # The following parameters are not used in xpu kernel currently,
        # we keep API compatible to CUDA's.
        scheduler_metadata=None,
        fa_version: int = 2,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        num_splits=0,
        return_softmax_lse: bool | None = False,
        s_aux: torch.Tensor | None = None,
        return_attn_probs: bool | None = False,
    ):
        assert cu_seqlens_k is not None or seqused_k is not None, (
            "cu_seqlens_k or seqused_k must be provided"
        )
        assert cu_seqlens_k is None or seqused_k is None, (
            "cu_seqlens_k and seqused_k cannot be provided at the same time"
        )
        assert block_table is None or seqused_k is not None, (
            "when enable block_table, seqused_k is needed"
        )
        assert block_table is not None or cu_seqlens_k is not None, (
            "when block_table is disabled, cu_seqlens_k is needed"
        )
        if out is None:
            out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
        real_window_size: tuple[int, int]
        if window_size is None:
            real_window_size = (-1, -1)
        else:
            assert len(window_size) == 2
            real_window_size = (window_size[0], window_size[1])  # noqa: F841

        return flash_attn_varlen_func(
            out=out,
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=softmax_scale,
            causal=causal,
            block_table=block_table,
            s_aux=s_aux,
            window_size=real_window_size,
            # alibi_slopes = alibi_slopes,
            # softcap=softcap,
            return_softmax_lse=return_softmax_lse,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
        )

    @staticmethod
    def get_scheduler_metadata(
        batch_size,
        max_seqlen_q,
        max_seqlen_k,
        num_heads_q,
        num_heads_kv,
        headdim,
        cache_seqlens: torch.Tensor,
        qkv_dtype=torch.bfloat16,
        headdim_v=None,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k_new: torch.Tensor | None = None,
        cache_leftpad: torch.Tensor | None = None,
        page_size: int | None = None,
        max_seqlen_k_new=0,
        causal=False,
        window_size=(-1, -1),  # -1 means infinite context window
        has_softcap=False,
        num_splits=0,  # Can be tuned for speed
        pack_gqa=None,  # Can be tuned for speed
        sm_margin=0,  # Can be tuned if some SMs are used for communication
    ) -> None:
        logger.warning_once(
            "get_scheduler_metadata is not implemented for xpu_ops, returning None."
        )
        return None

    @staticmethod
    def indexer_k_quant_and_cache(
        k: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        quant_block_size: int,
        scale_fmt: str | None,
    ) -> None:
        head_dim = k.shape[-1]
        k = k.view(-1, head_dim)  # [total_tokens, head_dim]

        def group_quant_torch(
            x: torch.Tensor,
            group_size: int,
            eps: float = 1e-10,
            dtype: torch.dtype | None = None,
            column_major_scales: bool = False,
            out_q: torch.Tensor | None = None,
            use_ue8m0: bool | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if use_ue8m0 is None:
                # Default fallback - could import is_deep_gemm_e8m0_used if needed
                use_ue8m0 = False

            if dtype is None:
                dtype = current_platform.fp8_dtype()

            # Validate inputs
            assert x.shape[-1] % group_size == 0, (
                f"Last dimension {x.shape[-1]} must be divisible by "
                f"group_size {group_size}"
            )
            assert x.stride(-1) == 1, "Input tensor groups must be contiguous"

            # Prepare output tensor
            if out_q is None:
                x_q = torch.empty_like(x, dtype=dtype)
            else:
                assert out_q.shape == x.shape
                x_q = out_q

            # Reshape input for group processing
            # Original shape: (..., last_dim)
            # Target shape: (..., num_groups, group_size)
            original_shape = x.shape
            num_groups = original_shape[-1] // group_size

            # Reshape to separate groups
            group_shape = original_shape[:-1] + (num_groups, group_size)
            x_grouped = x.view(group_shape)

            # Compute per-group absolute maximum values
            # Shape: (..., num_groups)
            abs_max = torch.amax(torch.abs(x_grouped), dim=-1, keepdim=False)
            abs_max = torch.maximum(
                abs_max, torch.tensor(eps, device=x.device, dtype=x.dtype)
            )

            # Compute scales
            FP8_MAX = torch.finfo(dtype).max
            FP8_MIN = torch.finfo(dtype).min
            scale_raw = abs_max / FP8_MAX

            if use_ue8m0:
                # For UE8M0 format, scales must be powers of 2
                scales = torch.pow(2.0, torch.ceil(torch.log2(scale_raw)))
            else:
                scales = scale_raw

            # Expand scales for broadcasting with grouped data
            # Shape: (..., num_groups, 1)
            scales_expanded = scales.unsqueeze(-1)

            # Quantize the grouped data
            x_scaled = x_grouped / scales_expanded
            x_clamped = torch.clamp(x_scaled, FP8_MIN, FP8_MAX)
            x_quantized = x_clamped.to(dtype)

            # Reshape back to original shape
            x_q.copy_(x_quantized.view(original_shape))

            # Prepare scales tensor in requested format
            if column_major_scales:
                # Column-major: (num_groups,) + batch_dims
                # Transpose the scales to put group dimension first
                scales_shape = (num_groups,) + original_shape[:-1]
                x_s = scales.permute(-1, *range(len(original_shape) - 1))
                x_s = x_s.contiguous().view(scales_shape)
            else:
                # Row-major: batch_dims + (num_groups,)
                x_s = scales.contiguous()

            # Ensure scales are float32
            return x_q, x_s.float()

        k_fp8, k_scale = group_quant_torch(
            k,
            group_size=quant_block_size,
            column_major_scales=False,
            use_ue8m0=(scale_fmt == "ue8m0"),
        )

        k_fp8_bytes = k_fp8.view(-1, head_dim).view(torch.uint8)
        scale_bytes = k_scale.view(torch.uint8).view(-1, 4)
        k = torch.cat(
            [k_fp8_bytes, scale_bytes], dim=-1
        )  # [total_tokens, head_dim + 4]

        slot_mapping = slot_mapping.flatten()
        # kv_cache: [num_block, block_size, head_dim + 4]
        kv_cache.view(-1, kv_cache.shape[-1]).index_copy_(0, slot_mapping, k)

    @staticmethod
    def cp_gather_indexer_k_quant_cache(
        kv_cache: torch.Tensor,
        dst_k: torch.Tensor,
        dst_scale: torch.Tensor,
        block_table: torch.Tensor,
        cu_seq_lens: torch.Tensor,
    ) -> None:
        """
        Args:
            kv_cache: [num_blocks, block_size, cache_stride] - quantized KV cache
                    Layout per block: [k_values, scale_values]
                    - k_values: [block_size * head_dim]
                    - scale_values: [block_size * head_dim * 4 / quant_block_size]
            dst_k: [num_tokens, head_dim] - output tensor for K values
            dst_scale: [num_tokens, head_dim / quant_block_size * 4]
                - output tensor for scale values
            block_table: [batch_size, num_blocks] - block table for indexing
            cu_seq_lens: [batch_size + 1] - cumulative sequence lengths
        """
        batch_size = block_table.size(0)
        num_tokens = dst_k.size(0)
        head_dim = dst_k.size(1)
        cache_block_size = kv_cache.size(1)
        quant_block_size = head_dim * 4 // dst_scale.size(1)

        # For each token, find which batch it belongs to using searchsorted
        token_indices = torch.arange(num_tokens, device=dst_k.device) + 1
        # cu_seq_lens is [batch_size + 1], we need to find which interval each
        # token belongs to
        batch_indices = torch.searchsorted(cu_seq_lens, token_indices) - 1
        batch_indices = torch.clamp(batch_indices, 0, batch_size - 1)

        # Calculate the in-batch sequence index for each token
        inbatch_seq_indices = token_indices - cu_seq_lens[batch_indices]

        # Find which block each token belongs to
        block_indices_in_table = inbatch_seq_indices // cache_block_size
        physical_block_indices = block_table[batch_indices, block_indices_in_table]

        # Calculate the offset within each block
        inblock_offsets = (inbatch_seq_indices - 1) % cache_block_size

        # Calculate strides
        block_stride = kv_cache.stride(0)  # stride for each block

        # Flatten kv_cache for easier indexing
        kv_cache_flat = kv_cache.view(-1)

        # Calculate source offset for K values for all tokens (vectorized)
        src_block_offsets = physical_block_indices * block_stride
        src_k_offsets = src_block_offsets + inblock_offsets * head_dim

        # Gather K values using advanced indexing
        # Create indices for all elements we need to gather
        k_indices = src_k_offsets.unsqueeze(1) + torch.arange(
            head_dim, device=dst_k.device
        )
        dst_k[:] = kv_cache_flat[k_indices]

        # Calculate source offset for scale values (vectorized)
        # Scales are stored after all K values for each block
        scale_size = head_dim * 4 // quant_block_size
        src_scale_offsets = src_block_offsets + head_dim + inblock_offsets * scale_size

        # Gather scale values
        scale_indices = src_scale_offsets.unsqueeze(1) + torch.arange(
            scale_size, device=dst_scale.device
        )
        dst_scale[:] = kv_cache_flat[scale_indices]

    @staticmethod
    def top_k_per_row_prefill(
        logits: torch.Tensor,
        cu_seqlen_ks: torch.Tensor,
        cu_seqlen_ke: torch.Tensor,
        raw_topk_indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        strdide1: int,
        topk_tokens: int,
    ) -> torch.Tensor:
        real_topk = min(topk_tokens, logits.shape[-1])
        topk_indices = logits.topk(real_topk, dim=-1)[1].to(torch.int32)
        topk_indices -= cu_seqlen_ks[:, None]
        mask_lo = topk_indices >= 0
        mask_hi = topk_indices - (cu_seqlen_ke - cu_seqlen_ks)[:, None] < 0
        mask = torch.full_like(
            topk_indices, False, dtype=torch.bool, device=topk_indices.device
        )
        mask = mask_lo & mask_hi
        topk_indices.masked_fill_(~mask, -1)
        raw_topk_indices[: topk_indices.shape[0], : topk_indices.shape[1]] = (
            topk_indices
        )

    @staticmethod
    def top_k_per_row_decode(
        logits: torch.Tensor,
        next_n: int,
        seq_lens: torch.Tensor,
        raw_topk_indices: torch.Tensor,
        num_rows: int,
        stride0: int,
        stride1: int,
        topk_tokens: int,
    ) -> torch.Tensor:
        device = logits.device
        batch_size = seq_lens.size(0)
        # padded query len
        padded_num_tokens = batch_size * next_n
        positions = (
            torch.arange(logits.shape[-1], device=device)
            .unsqueeze(0)
            .expand(batch_size * next_n, -1)
        )
        row_indices = torch.arange(padded_num_tokens, device=device) // next_n
        next_n_offset = torch.arange(padded_num_tokens, device=device) % next_n
        index_end_pos = (seq_lens[row_indices] - next_n + next_n_offset).unsqueeze(1)
        # index_end_pos: [B * N, 1]
        mask = positions <= index_end_pos
        # mask: [B * N, L]
        logits = logits.masked_fill(~mask, float("-inf"))
        real_topk = min(topk_tokens, logits.shape[-1])
        topk_indices = logits.topk(real_topk, dim=-1)[1].to(torch.int32)  # [B * N, K]
        # ensure we don't set indices for the top k
        # that is out of range(masked already)
        # this will happen if context length is shorter than K
        topk_indices[topk_indices > index_end_pos] = -1
        raw_topk_indices[: topk_indices.shape[0], : topk_indices.shape[1]] = (
            topk_indices
        )

    @staticmethod
    def register_ops_once() -> None:
        global _OPS_REGISTERED
        if not _OPS_REGISTERED:
            # register all the custom ops here
            direct_register_custom_op(
                op_name="xpu_ops_deepseek_scaling_rope",
                op_func=_xpu_ops_deepseek_scaling_rope_impl,
                mutates_args=[],
                fake_impl=_xpu_ops_deepseek_scaling_rope_fake,
                dispatch_key=current_platform.dispatch_key,
            )

            direct_register_custom_op(
                op_name="xpu_mxfp8_quantize",
                op_func=_xpu_mxfp8_quantize_impl,
                fake_impl=_xpu_mxfp8_quantize_fake,
            )

            direct_register_custom_op(
                op_name="xpu_mxfp4_quantize",
                op_func=_xpu_mxfp4_quantize_impl,
                fake_impl=_xpu_mxfp4_quantize_fake,
            )

            direct_register_custom_op(
                op_name="gdn_attention_core_xpu",
                op_func=_gdn_attention_core_xpu_impl,
                mutates_args=["core_attn_out", "z"],
                fake_impl=_gdn_attention_core_xpu_fake,
            )

            direct_register_custom_op(
                op_name="xpu_topk_topp_sampler",
                op_func=_topk_topp_sample_impl,
                fake_impl=_topk_topp_sample_fake,
            )

            _OPS_REGISTERED = True


xpu_ops.register_ops_once()
