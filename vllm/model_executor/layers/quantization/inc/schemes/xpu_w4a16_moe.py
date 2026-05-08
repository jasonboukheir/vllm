# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
    FusedMoEMethodBase,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoeWeightScaleSupported
from vllm.model_executor.layers.fused_moe.oracle.int_wna16 import (
    convert_to_wna16_moe_kernel_format,
    make_wna16_moe_kernel,
    select_wna16_moe_backend,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4StaticGroupScale,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.scalar_type import scalar_types

if TYPE_CHECKING:
    from ..resolver import INCLayerConfig


class INCXPUMoEMethod(FusedMoEMethodBase):
    """W4A16 INT4-symmetric MoE on Intel XPU via vllm-xpu-kernels.

    Companion to ``INCXPUW4A16LinearScheme``. Routes through the WNA16 MoE
    oracle, which on XPU returns ``XPUExpertsWNA16`` — a thin wrapper over
    ``vllm_xpu_kernels.fused_moe_interface.xpu_fused_moe(is_int4=True)``.

    Weights are loaded in standard GPTQ MoE format
    ([E, K // pack_factor, 2*N] int32 etc.) so existing GPTQv2 sym int4
    checkpoints (e.g. ``palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4``) load
    without a custom adapter. ``process_weights_after_loading`` then
    repacks into the [E, 2*N, K] uint8 layout the XPU kernel expects.
    """

    def __init__(
        self,
        layer_config: "INCLayerConfig",
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        if layer_config.bits != 4:
            raise NotImplementedError(
                f"INC XPU MoE only supports 4-bit quantization, "
                f"got bits={layer_config.bits}."
            )
        if not layer_config.sym:
            raise NotImplementedError(
                "INC XPU MoE only supports symmetric quantization for now."
            )
        self.weight_bits = layer_config.bits
        self.group_size = layer_config.group_size
        self.sym = layer_config.sym
        self.pack_factor = 32 // layer_config.bits
        self.input_dtype: torch.dtype | None = None

        weight_key = QuantKey(
            scalar_types.uint4b8, kInt4StaticGroupScale, symmetric=True
        )
        self.wna16_moe_backend, self.experts_cls = select_wna16_moe_backend(
            moe, weight_key, self.weight_bits
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        layer.input_dtype = self.input_dtype
        # Drop intermediate_size_full (used by Marlin act-order); not relevant
        # here since INC checkpoints have desc_act=false.
        extra_weight_attrs.pop("intermediate_size_full", None)

        if hidden_size % self.group_size != 0:
            raise ValueError(
                f"INC XPU MoE requires hidden_size ({hidden_size}) to be "
                f"divisible by group_size ({self.group_size})."
            )
        if intermediate_size_per_partition % self.group_size != 0:
            raise ValueError(
                f"INC XPU MoE requires intermediate_size_per_partition "
                f"({intermediate_size_per_partition}) to be divisible by "
                f"group_size ({self.group_size}); check tensor-parallel size."
            )

        scales_size13 = hidden_size // self.group_size
        scales_size2 = intermediate_size_per_partition // self.group_size
        layer.num_groups_w13 = scales_size13
        layer.num_groups_w2 = scales_size2

        strategy = FusedMoeWeightScaleSupported.GROUP.value
        extra_weight_attrs.update({"quant_method": strategy, "is_transposed": True})

        # GPTQ-format storage: [E, K // pack_factor, 2*N] int32 along packed
        # input dim. Same as GPTQMarlinMoEMethod so existing checkpoints load.
        w13_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.pack_factor,
                2 * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_qweight", w13_qweight)
        set_weight_attrs(w13_qweight, extra_weight_attrs)

        w2_qweight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.pack_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_qweight", w2_qweight)
        set_weight_attrs(w2_qweight, extra_weight_attrs)

        w13_scales = torch.nn.Parameter(
            torch.empty(
                num_experts,
                scales_size13,
                2 * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_scales", w13_scales)
        set_weight_attrs(w13_scales, extra_weight_attrs)

        w2_scales = torch.nn.Parameter(
            torch.empty(num_experts, scales_size2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_scales", w2_scales)
        set_weight_attrs(w2_scales, extra_weight_attrs)

        # GPTQ checkpoints carry qzeros / g_idx tensors. Register placeholders
        # so the loader doesn't error; they're discarded in
        # process_weights_after_loading because the XPU kernel handles
        # symmetric int4 zero points internally.
        for name, shape in (
            (
                "w13_qzeros",
                (
                    num_experts,
                    scales_size13,
                    2 * intermediate_size_per_partition // self.pack_factor,
                ),
            ),
            (
                "w2_qzeros",
                (
                    num_experts,
                    scales_size2,
                    hidden_size // self.pack_factor,
                ),
            ),
            ("w13_g_idx", (num_experts, hidden_size)),
            ("w2_g_idx", (num_experts, intermediate_size_per_partition)),
        ):
            param = torch.nn.Parameter(
                torch.empty(shape, dtype=torch.int32), requires_grad=False
            )
            layer.register_parameter(name, param)
            set_weight_attrs(param, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        (
            w13,
            w2,
            w13_scale,
            w2_scale,
            _w13_g_idx,
            _w2_g_idx,
            _w13_g_idx_sort,
            _w2_g_idx_sort,
            _w13_global_scale,
            _w2_global_scale,
            _w13_bias,
            _w2_bias,
        ) = convert_to_wna16_moe_kernel_format(
            backend=self.wna16_moe_backend,
            layer=layer,
            quant_config=self,  # _process_weights_xpu ignores quant_config
            input_dtype=self.input_dtype,
            w13=layer.w13_qweight,
            w2=layer.w2_qweight,
            w13_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w13_g_idx=layer.w13_g_idx,
            w2_g_idx=layer.w2_g_idx,
        )

        replace_parameter(layer, "w13_qweight", w13)
        replace_parameter(layer, "w2_qweight", w2)
        replace_parameter(layer, "w13_scales", w13_scale)
        replace_parameter(layer, "w2_scales", w2_scale)

        self._assert_no_g_idx_reordering(layer)

        empty = torch.empty((0,), dtype=torch.int32, device=w13.device)
        for name in ("w13_qzeros", "w2_qzeros", "w13_g_idx", "w2_g_idx"):
            if hasattr(layer, name):
                replace_parameter(layer, name, empty)

        self._setup_kernel(layer)

    @staticmethod
    def _assert_no_g_idx_reordering(layer: torch.nn.Module) -> None:
        """xpu_fused_moe(is_int4=True) does not implement desc_act / g_idx
        activation reordering. Verify any loaded g_idx is monotonically
        non-decreasing (the desc_act=false pattern) so that the hardcoded
        is_k_full=True passed into make_wna16_moe_kernel below is safe."""
        for name in ("w13_g_idx", "w2_g_idx"):
            if not hasattr(layer, name):
                continue
            g_idx = getattr(layer, name)
            if g_idx.numel() == 0:
                continue
            if (g_idx[..., 1:] < g_idx[..., :-1]).any():
                raise NotImplementedError(
                    f"INC XPU MoE requires desc_act=false GPTQ checkpoints; "
                    f"got non-monotonic {name} indicating activation reordering, "
                    f"which xpu_fused_moe does not support."
                )

    def _setup_kernel(self, layer: torch.nn.Module) -> None:
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        self.moe_kernel = make_wna16_moe_kernel(
            moe_quant_config=self.moe_quant_config,
            moe_config=self.moe,
            experts_cls=self.experts_cls,
            layer=layer,
            is_k_full=True,  # validated above via _assert_no_g_idx_reordering
            w13_g_idx=None,
            w2_g_idx=None,
            w13_g_idx_sort_indices=None,
            w2_g_idx_sort_indices=None,
            routing_tables=layer._maybe_init_expert_routing_tables(),
            shared_experts=layer.shared_experts,
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        return int4_w4a16_moe_quant_config(
            w1_scale=layer.w13_scales,
            w2_scale=layer.w2_scales,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size],
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            hidden_states=x,
            w1=layer.w13_qweight,
            w2=layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            shared_experts_input=shared_experts_input,
        )
