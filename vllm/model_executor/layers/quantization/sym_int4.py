# SPDX-License-Identifier: Apache-2.0

from typing import Any, Dict, List, Optional, Tuple, Callable, Union
from concurrent.futures import ThreadPoolExecutor
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               UnquantizedLinearMethod)
from vllm.model_executor.parameter import (BlockQuantScaleParameter,
                                           ModelWeightParameter,
                                           PerTensorScaleParameter)

from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig, FusedMoEQuantConfig
from vllm.model_executor.layers.fused_moe.router.fused_moe_router import FusedMoERouter
from vllm.model_executor.utils import set_weight_attrs
import torch
from torch.nn import Module
from torch.nn.parameter import Parameter
from vllm.utils.math_utils import round_up

from vllm.envs import VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT, VLLM_QUANTIZE_Q40_LIB
import ctypes
from packaging import version

MIN_IPEX_VERSION = "2.5.0"
QK4_GROUP_SIZE: int = 128
QK4_PACK_FACTOR: int = 8

_QLIB_CACHE = None

def _get_quant_lib():
    """
    Lazy loads the quantization library and sets up argtypes.
    Singleton pattern to avoid reloading the DLL multiple times.
    """
    global _QLIB_CACHE
    if _QLIB_CACHE is not None:
        return _QLIB_CACHE

    try:
        clib = ctypes.CDLL(VLLM_QUANTIZE_Q40_LIB)
    except OSError as e:
        raise RuntimeError(f"Failed to load required quantization lib at {VLLM_QUANTIZE_Q40_LIB}: {e}")

    # Updated argtypes to match C signature:
    # (float *src, int32_t *qweight, ggml_fp16_t *scale, int out_features, int in_features, int block_size)
    clib.quantize_q4_0_to_qweight_and_scale.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,  # [New] block_size argument
    ]
    clib.quantize_q4_0_to_qweight_and_scale.restype = ctypes.c_size_t

    _QLIB_CACHE = clib
    return clib

def ggml_quantize_tensor(weight: torch.Tensor,
                         out_qweight: torch.Tensor,
                         out_scale: torch.Tensor,
                         out_features: int,
                         in_features: int,
                         block_size: int = QK4_GROUP_SIZE,
                         transpose: bool = True):
    """
    Shared implementation for quantizing a tensor using the C library.

    Args:
        transpose: If True (default), transpose the output tensors. The C code
            fills row-wise [out_features, ...], and some callers (Linear) need
            the transposed layout. MoE callers that want the original
            [out_features, ...] layout should pass transpose=False to avoid a
            redundant transpose-then-transpose-back.
    """
    # Assertions
    assert weight.dim() == 2
    # Validate shapes considering packing factor and block size
    assert out_qweight.shape == (out_features, in_features // QK4_PACK_FACTOR)
    assert out_scale.shape == (out_features, in_features // block_size)

    assert weight.dtype == torch.float32
    assert out_qweight.dtype == torch.int32
    assert out_scale.dtype == torch.float16

    assert out_qweight.is_contiguous()
    assert out_scale.is_contiguous()

    # Ctypes casting
    src = ctypes.cast(weight.data.data_ptr(), ctypes.POINTER(ctypes.c_float))
    qweight = ctypes.cast(out_qweight.data.data_ptr(), ctypes.POINTER(ctypes.c_int32))
    scale = ctypes.cast(out_scale.data.data_ptr(), ctypes.POINTER(ctypes.c_uint16))

    clib = _get_quant_lib()

    # Call C function with the new block_size parameter
    clib.quantize_q4_0_to_qweight_and_scale(src, qweight, scale, out_features, in_features, block_size)

    if transpose:
        # Transpose for callers that need column-major layout (e.g. Linear)
        out_qweight = out_qweight.transpose(0, 1).contiguous()
        out_scale = out_scale.transpose(0, 1).contiguous()

    return out_qweight, out_scale

# ==============================================================================
#  Classes
# ==============================================================================

class SymInt4Config(QuantizationConfig):
    """SYM_INT4 quantization config class which uses IPEX kernel behind the scene..."""
    def __init__(self) -> None:
        super().__init__()

    @classmethod
    def get_name(cls) -> str:
        return "sym_int4"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return -1

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SymInt4Config":
        return cls()

    @classmethod
    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        """Get the quantize method to use for the quantized layer.

        Args:
            layer: The layer for the quant method.
            prefix: The full name of the layer in the state dict
        Returns:
            The quantize method. None if the given layer doesn't support quant
            method.
        """
        modules_to_not_convert = ["visual", "vision", "vpm", "resampler"]
        modules_to_convert=["vision_experts"]
        if any(key in prefix for key in modules_to_not_convert) and not any(key in prefix for key in modules_to_convert):
            return UnquantizedLinearMethod()
        if isinstance(layer, LinearBase):
            return SymInt4LinearMethod(self)
        if isinstance(layer, FusedMoE):
            return XPUGPTQInt4LinearMoEMethod(self, layer.moe_config)
        else:
            return None


class SymInt4LinearMethod(LinearMethodBase):
    def __init__(self, quant_config: SymInt4Config):
        self.quant_config = quant_config
        # Ensure lib is loaded on init
        _get_quant_lib()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.orig_dtype = params_dtype

        weight_dtype = params_dtype
        weight = ModelWeightParameter(data=torch.empty(
            output_size_per_partition,
            input_size_per_partition,
            dtype=weight_dtype,
            device="cpu"),
                                      input_dim=1,
                                      output_dim=0,
                                      weight_loader=weight_loader)
        layer.register_parameter("weight", weight)

    def apply(self,
              layer: torch.nn.Module,
              x: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        # The same with the GPTQ's linear method by IPEX
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = layer.ipex_qlinear(reshaped_x)
        if bias is not None:
            out.add_(bias)
        return out.reshape(x.shape[:-1] + (layer.ipex_output_size, ))

    def process_weights_after_loading(self, layer: Module) -> None:
        weight = layer.weight.float()
        out_features = layer.weight.shape[0]
        in_features = layer.weight.shape[1]

        qweight = torch.zeros((out_features, in_features // QK4_PACK_FACTOR), dtype=torch.int32, device=layer.weight.device)
        scale = torch.zeros((out_features, in_features // QK4_GROUP_SIZE), dtype=torch.float16, device=layer.weight.device)

        # Use the extracted global function
        qweight, scale = ggml_quantize_tensor(
            weight, qweight, scale, out_features, in_features, block_size=QK4_GROUP_SIZE
        )
        
        qweight = qweight.to("xpu")
        scale = scale.to("xpu")

        # Use qweight to replace weight...
        layer.weight = Parameter(qweight, requires_grad=False)
        # qweight_scale
        layer.weight_scale = Parameter(scale, requires_grad=False)

        try:
            import intel_extension_for_pytorch as ipex
            if version.parse(ipex.__version__) < version.parse(MIN_IPEX_VERSION):
                raise ImportError(
                    f"intel_extension_for_pytorch version is wrong. "
                    f"Current: {ipex.__version__}, Required: >={MIN_IPEX_VERSION}")
        except ImportError as err:
            raise ImportError(
                "Please install "
                f"intel_extension_for_pytorch>={MIN_IPEX_VERSION} via "
                f"`pip install intel_extension_for_pytorch>={MIN_IPEX_VERSION}`"
                " to use IPEX-AWQ linear method.") from err

        lowp_mode = ipex.quantization.WoqLowpMode.INT8
        weight_dtype = ipex.quantization.WoqWeightDtype.INT4
        act_quant_mode = ipex.quantization.WoqActQuantMode.PER_BATCH_IC_BLOCK
        qconfig = ipex.quantization.get_weight_only_quant_qconfig_mapping(
            weight_dtype=weight_dtype,
            lowp_mode=lowp_mode,
            act_quant_mode=act_quant_mode,
            group_size=QK4_GROUP_SIZE,
        )
        layer.ipex_output_size = layer.weight.shape[-1]
        g_idx = None
        layer.ipex_qlinear = ipex.llm.quantization.woq_linear. \
            IPEXWeightOnlyQuantizedLinear.from_weight(
            layer.weight,     # weight should be on xpu...
            layer.weight_scale,
            torch.tensor([8], device=layer.weight.device, dtype=torch.int8),
            layer.weight.size(0),
            layer.ipex_output_size,
            qconfig=qconfig,
            g_idx=g_idx,
            bias=None,
            group_size=QK4_GROUP_SIZE,
            # For GPTQ layout
            quant_method=0
        )


class XPUGPTQInt4LinearMoEMethod(FusedMoEMethodBase):
    def __init__(
        self,
        quant_config: SymInt4Config,
        moe: "FusedMoEConfig",
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.moe_config = moe
        # Ensure lib is loaded
        _get_quant_lib()

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return None

    def create_weights(self, layer: Module, num_experts: int, hidden_size: int,
                       intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):
        # Just normally loads the weights, obey VLLM_OFFLOAD_WEIGHTS_BEFORE_QUANT...
        layer.intermediate_size_per_partition = intermediate_size_per_partition
        layer.hidden_size = hidden_size
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None

        tp_size = get_tensor_model_parallel_world_size()
        if tp_size == 4:
            intermediate_size_per_partition = round_up(intermediate_size_per_partition, 256)
        elif tp_size == 8:
            if self.moe_config.hidden_dim == 2048:
                # For qwen3-30b-a3b
                intermediate_size_per_partition = round_up(intermediate_size_per_partition, 128)
            elif self.moe_config.hidden_dim == 4096:
                # For qwen3-235b-a22b
                intermediate_size_per_partition = round_up(intermediate_size_per_partition, 256)
            else:
                raise ValueError("Unsupported hidden_dim")
        elif tp_size == 16:
            # For qwen3-235b
            assert self.moe_config.hidden_dim == 4096, f"Currently TP_SIZE=16 only supports qwen3-235b-a22b"
            intermediate_size_per_partition = round_up(intermediate_size_per_partition, 128)
        layer.d_ff = intermediate_size_per_partition
        # w13 shape: [d_ff * 2, d_model]
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype,
            device="cpu"),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # w2 shape: [d_model, d_ff]
        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype,
            device="cpu"),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        import intel_extension_for_pytorch as ipex
        E = layer.num_experts
        d_model = layer.hidden_size
        d_ff = layer.d_ff

        assert d_model % QK4_PACK_FACTOR == 0 and d_ff % QK4_PACK_FACTOR == 0, "INT4 packing requires feature dims % 8 == 0"
        assert d_model % QK4_GROUP_SIZE == 0 and d_ff % QK4_GROUP_SIZE == 0, f"group_size={QK4_GROUP_SIZE} requires dims % {QK4_GROUP_SIZE} == 0"

        # Allocating CPU tensors
        w13_qweight = torch.empty(E, 2 * d_ff, d_model // QK4_PACK_FACTOR, dtype=torch.int32, device="cpu")
        w2_qweight  = torch.empty(E, d_model,    d_ff // QK4_PACK_FACTOR, dtype=torch.int32, device="cpu")
        w13_scales  = torch.empty(E, 2 * d_ff, d_model // QK4_GROUP_SIZE, dtype=torch.float16, device="cpu")
        w2_scales   = torch.empty(E, d_model,    d_ff // QK4_GROUP_SIZE, dtype=torch.float16, device="cpu")

        # Quantize per expert (parallelized across experts)
        num_loop = getattr(layer, "local_num_experts", E)

        def _quantize_expert(e):
            # Ensure fp32 contiguous
            w13_e = layer.w13_weight[e].float().contiguous().to("cpu")
            w2_e  = layer.w2_weight[e].float().contiguous().to("cpu")

            # --- w13 --- (transpose=False: C output is already [out, in//8])
            q13_buf = torch.zeros((2 * d_ff, d_model // QK4_PACK_FACTOR), dtype=torch.int32, device="cpu")
            s13_buf = torch.zeros((2 * d_ff, d_model // QK4_GROUP_SIZE), dtype=torch.float16, device="cpu")
            q13, s13 = ggml_quantize_tensor(
                w13_e, q13_buf, s13_buf, 2 * d_ff, d_model,
                block_size=QK4_GROUP_SIZE, transpose=False,
            )
            w13_qweight[e].copy_(q13)
            w13_scales[e].copy_(s13)

            # --- w2 ---
            q2_buf = torch.zeros((d_model, d_ff // QK4_PACK_FACTOR), dtype=torch.int32, device="cpu")
            s2_buf = torch.zeros((d_model, d_ff // QK4_GROUP_SIZE), dtype=torch.float16, device="cpu")
            q2, s2 = ggml_quantize_tensor(
                w2_e, q2_buf, s2_buf, d_model, d_ff,
                block_size=QK4_GROUP_SIZE, transpose=False,
            )
            w2_qweight[e].copy_(q2)
            w2_scales[e].copy_(s2)

        with ThreadPoolExecutor() as executor:
            list(executor.map(_quantize_expert, range(num_loop)))

        # Move to XPU
        w13_qweight = w13_qweight.to("xpu")
        w2_qweight  = w2_qweight.to("xpu")
        w13_scales  = w13_scales.to("xpu")
        w2_scales   = w2_scales.to("xpu")

        # Override parameters
        layer.w13_weight = torch.nn.Parameter(w13_qweight, requires_grad=False)
        layer.w2_weight  = torch.nn.Parameter(w2_qweight,  requires_grad=False)
        layer.w13_scales = torch.nn.Parameter(w13_scales, requires_grad=False)
        layer.w2_scales  = torch.nn.Parameter(w2_scales,  requires_grad=False)

        layer.ipex_fusion = ipex.llm.modules.GatedMLPMOE(
            layer.w13_weight,                      # [E, 2*d_ff, d_model//8]
            layer.w2_weight,                       # [E, d_model, d_ff//8]
            w1_scale_inv=layer.w13_scales,          # [E, 2*d_ff, d_model//64]
            w2_scale_inv=layer.w2_scales,           # [E, d_model, d_ff//64]
            is_int4=True
        )

    def apply(
        self,
        layer: FusedMoE,
        router: FusedMoERouter,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        res = layer.ipex_fusion(
            x,
            layer.use_grouped_topk,
            layer.top_k,
            router_logits,
            layer.renormalize,
            topk_group=layer.topk_group,
            num_expert_group=layer.num_expert_group,
            custom_routing_function=layer.custom_routing_function,
            scoring_func=layer.scoring_func,
        )
        return res
