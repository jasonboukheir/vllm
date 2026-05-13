# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from fractions import Fraction
from typing import TYPE_CHECKING, Any

import regex as re
import torch
from torch.nn.parameter import Parameter

from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import RoutedExperts
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
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization import (
    QuantizationConfig,
    QuantizationMethods,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kInt4StaticGroupScale,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

if TYPE_CHECKING:
    from vllm.model_executor.models.utils import WeightsMapper

logger = init_logger(__name__)


class INCConfig(QuantizationConfig):
    """Config class for Intel Neural Compressor (INC).
    Repo: https://github.com/intel/neural-compressor
    """

    SUPPORTED_BITS = {2, 3, 4, 8}
    SUPPORTED_DTYPES = {"int"}
    SUPPORTED_FORMATS = {"auto_round:auto_gptq", "auto_round:auto_awq"}
    SUPPORTED_BACKENDS = {
        "auto",
        "gptq",
        "gptq:marlin",
        "awq",
        "awq:marlin",
        "marlin",
    }

    def __init__(
        self,
        weight_bits: int,
        group_size: int,
        sym: bool = True,
        packing_format: str = "auto_round:auto_gptq",
        block_name_to_quantize: str | list[str] | None = None,
        extra_config: dict[str, Any] | None = None,
        data_type: str = "int",
        backend: str = "auto",
    ) -> None:
        super().__init__()
        if weight_bits not in self.SUPPORTED_BITS:
            raise ValueError(
                f"Unsupported weight_bits: {weight_bits}, "
                f"currently only support {self.SUPPORTED_BITS}."
            )
        if data_type not in self.SUPPORTED_DTYPES:
            raise ValueError(
                f"Unsupported data_type: {data_type},"
                f" currently only support  {self.SUPPORTED_DTYPES}."
            )
        if packing_format not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported packing_format: {packing_format}, "
                f"currently only support {self.SUPPORTED_FORMATS}."
            )
        if backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(
                f"Unsupported backend: {backend},  "
                f"currently only support {self.SUPPORTED_BACKENDS}."
            )

        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.packing_format = packing_format
        self.block_name_to_quantize = (
            block_name_to_quantize.split(",")
            if isinstance(block_name_to_quantize, str)
            else block_name_to_quantize
        )
        self.extra_config = extra_config
        self.data_type = data_type
        self.backend = backend
        self.pack_factor = Fraction(32, weight_bits)

    def __repr__(self) -> str:
        return (
            f"INCConfig(weight_bits={self.weight_bits}, "
            f"group_size={self.group_size}, sym={self.sym})"
        )

    @classmethod
    def get_name(cls) -> QuantizationMethods:
        return "inc"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return ["quantization_config.json"]

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "INCConfig":
        # GPTQModel checkpoints carry per-module overrides under `dynamic`
        # rather than `extra_config`. Translate `-:<regex>` (skip) and
        # `+:<regex>` / bare (override) entries into INC's extra_config
        # format so get_layer_config() handles them via its existing regex
        # path. For "skip", we set bits=16 (= unquantized in check_quantized).
        #
        # GPTQModel matches its `dynamic` keys with `re.match` (anchored at
        # start), but INC's get_layer_config uses `re.search` and only treats
        # a key as a regex when it contains regex special characters. We
        # therefore anchor every translated key with `^` and append `.*` for
        # bare prefixes so they reach the regex matcher. Override values can
        # be a dict (config), True/None (= use defaults), or False (= skip).
        skip_cfg = {"bits": 16, "group_size": -1, "sym": True}
        extra_config = cls.get_from_keys_or(config, ["extra_config"], None)
        dynamic = cls.get_from_keys_or(config, ["dynamic"], None)
        if dynamic:
            extra_config = dict(extra_config) if extra_config else {}
            for raw_pattern, override in dynamic.items():
                if raw_pattern.startswith("-:"):
                    pattern = raw_pattern.removeprefix("-:")
                    cfg: dict[str, Any] = dict(skip_cfg)
                else:
                    pattern = raw_pattern.removeprefix("+:")
                    if override is False:
                        cfg = dict(skip_cfg)
                    elif override is None or override is True:
                        cfg = {}
                    elif isinstance(override, dict):
                        cfg = override
                    else:
                        raise ValueError(
                            f"INC `dynamic` override for {raw_pattern!r} must "
                            f"be a dict, bool, or None; got {type(override).__name__}."
                        )
                anchored = pattern if pattern.startswith("^") else f"^{pattern}"
                if not any(c in r"*+?^$()[]{}|\\" for c in anchored.removeprefix("^")):
                    anchored = f"{anchored}.*"
                extra_config[anchored] = cfg

        return cls(
            weight_bits=cls.get_from_keys(config, ["bits"]),
            group_size=cls.get_from_keys(config, ["group_size"]),
            sym=cls.get_from_keys(config, ["sym"]),
            packing_format=cls.get_from_keys_or(
                config, ["packing_format"], "auto_round:auto_gptq"
            ),
            block_name_to_quantize=cls.get_from_keys_or(
                config, ["block_name_to_quantize", "to_quant_block_names"], None
            ),
            extra_config=extra_config,
            data_type=cls.get_from_keys_or(config, ["data_type"], "int"),
            backend=cls.get_from_keys_or(config, ["backend", "vllm_backend"], "auto"),
        )

    def get_layer_config(self, layer, layer_name: str):
        def get_config(name: str, quantized: bool = True):
            if not self.extra_config:
                return (
                    self.weight_bits if quantized else 16,
                    self.group_size if quantized else -1,
                    self.sym if quantized else True,
                )

            # exact match first
            if name in self.extra_config:
                cfg = self.extra_config[name]
                return (
                    cfg.get("bits", self.weight_bits if quantized else 16),
                    cfg.get("group_size", self.group_size if quantized else -1),
                    cfg.get("sym", self.sym if quantized else True),
                )

            REGEX_SPECIAL_CHARS = set(r"*+?^$()[]{}|\\")
            for pattern, cfg in self.extra_config.items():
                if not isinstance(pattern, str) or not any(
                    c in REGEX_SPECIAL_CHARS for c in pattern
                ):
                    continue

                try:
                    if re.search(re.compile(pattern), name) is not None:
                        return (
                            cfg.get("bits", self.weight_bits if quantized else 16),
                            cfg.get("group_size", self.group_size if quantized else -1),
                            cfg.get("sym", self.sym if quantized else True),
                        )
                except re.error:
                    # Invalid regex, ignore.
                    continue

            return (
                self.weight_bits if quantized else 16,
                self.group_size if quantized else -1,
                self.sym if quantized else True,
            )

        # 1. Exact match from config
        if self.extra_config and layer_name in self.extra_config:
            return get_config(layer_name)

        # 2. Determine whether layer should be quantized
        quantized = not isinstance(layer, ParallelLMHead)
        if self.block_name_to_quantize:
            quantized = any(
                layer_name.startswith(name) for name in self.block_name_to_quantize
            )

        # 3. Handle fused MoE
        if self.extra_config and "fusedmoe" in layer.__class__.__name__.lower():
            moe_configs = [
                get_config(name, quantized)
                for name in self.extra_config
                if name.startswith(layer_name)
            ]
            if moe_configs:
                if len(set(moe_configs)) == 1:
                    return moe_configs[0]
                raise ValueError(
                    f"Fused MoE layer '{layer_name}' requires "
                    f"consistent quant config for all sub-layers"
                )

        # 4. Handle fused QKV or other patterns
        if self.extra_config:
            for fusion_key, sub_keys in self.packed_modules_mapping.items():
                if fusion_key in layer_name and layer_name.count(fusion_key) == 1:
                    sub_names = [
                        layer_name.replace(fusion_key, sub_key) for sub_key in sub_keys
                    ]
                    sub_configs = [get_config(name, quantized) for name in sub_names]
                    if len(set(sub_configs)) == 1:
                        return sub_configs[0]
                    raise ValueError(
                        f"Fused module '{layer_name}' requires "
                        f"consistent quant config for {sub_names}"
                    )

        # 5. Fallback or try a regular expression match
        return get_config(layer_name, quantized)

    def check_quantized(self, weight_bits: int) -> bool:
        return weight_bits < 16

    def apply_vllm_mapper(self, hf_to_vllm_mapper: "WeightsMapper"):
        if self.block_name_to_quantize is not None:
            self.block_name_to_quantize = hf_to_vllm_mapper.apply_list(
                self.block_name_to_quantize
            )
        if self.extra_config is not None:
            self.extra_config = hf_to_vllm_mapper.apply_dict(self.extra_config)

    def apply_awq_quant_layer(self, layer, prefix: str, backend: str = "auto"):
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_marlin_supported,
            check_moe_marlin_supports_layer,
        )

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            weight_bits,
            group_size,
            sym,
        )
        if backend == "auto" or "marlin" in backend:
            AWQ_TYPE_MAP = {
                4: scalar_types.uint4,
                8: scalar_types.uint8,
            }
            use_marlin = (weight_bits in AWQ_TYPE_MAP) and check_marlin_supported(
                AWQ_TYPE_MAP[weight_bits], group_size, not sym
            )

            if isinstance(layer, RoutedExperts):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size
                )

        else:
            use_marlin = False
        if use_marlin:
            from vllm.model_executor.layers.quantization.awq_marlin import (
                AWQMarlinConfig,
                AWQMarlinLinearMethod,
                AWQMarlinMoEMethod,
            )

            quant_args_marlin = AWQMarlinConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
                lm_head_quantized=False,
                full_config={},
                modules_to_not_convert=[],
            )
        else:
            from vllm.model_executor.layers.quantization.awq import (
                AWQConfig,
                AWQLinearMethod,
            )

            quant_args = AWQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                zero_point=not sym,
            )

        if isinstance(layer, RoutedExperts):
            if use_marlin:
                return AWQMarlinMoEMethod(quant_args_marlin, layer.moe_config)
            from vllm.model_executor.layers.quantization.moe_wna16 import MoeWNA16Config

            config = {
                "quant_method": "awq",
                "bits": weight_bits,
                "group_size": group_size,
                "zero_point": not sym,
                "lm_head": False,
            }
            return MoeWNA16Config.from_config(config).get_quant_method(layer, prefix)

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if use_marlin:
                return AWQMarlinLinearMethod(quant_args_marlin)
            else:
                return AWQLinearMethod(quant_args)
        return None

    def apply_gptq_quant_layer(self, layer, prefix: str, backend: str = "auto"):
        from vllm.model_executor.layers.quantization.utils.marlin_utils import (
            check_marlin_supported,
            check_moe_marlin_supports_layer,
        )

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        logger.debug(
            "[%s] Type: %s, Bits: %s, Group Size: %s, Sym: %s",
            prefix,
            layer.__class__.__name__,
            weight_bits,
            group_size,
            sym,
        )
        if backend == "auto" or "marlin" in backend:
            GPTQ_TYPE_MAP = {
                (4, True): scalar_types.uint4b8,
                (8, True): scalar_types.uint8b128,
            }
            use_marlin = (weight_bits, sym) in GPTQ_TYPE_MAP and check_marlin_supported(
                GPTQ_TYPE_MAP[(weight_bits, sym)], group_size, has_zp=not sym
            )
            if isinstance(layer, RoutedExperts):
                use_marlin = use_marlin and check_moe_marlin_supports_layer(
                    layer, group_size
                )
        else:
            use_marlin = False
        if use_marlin:
            from vllm.model_executor.layers.quantization.gptq_marlin import (
                GPTQMarlinConfig,
                GPTQMarlinLinearMethod,
                GPTQMarlinMoEMethod,
            )

            quant_args_marlin = GPTQMarlinConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                is_sym=sym,
                lm_head_quantized=False,
                desc_act=False,
                dynamic={},
                full_config={},
            )
        else:
            from vllm.model_executor.layers.quantization.gptq import (
                GPTQConfig,
                GPTQLinearMethod,
            )

            quant_args = GPTQConfig(
                weight_bits=weight_bits,
                group_size=group_size,
                lm_head_quantized=False,
                desc_act=False,
                dynamic={},
            )

        if isinstance(layer, RoutedExperts):
            if use_marlin:
                return GPTQMarlinMoEMethod(quant_args_marlin, layer.moe_config)
            else:
                from vllm.model_executor.layers.quantization.moe_wna16 import (
                    MoeWNA16Config,
                )

                config = {
                    "quant_method": "gptq",
                    "bits": weight_bits,
                    "group_size": group_size,
                    "sym": sym,
                    "lm_head": False,
                }
                return MoeWNA16Config.from_config(config).get_quant_method(
                    layer, prefix
                )

        if isinstance(layer, (LinearBase, ParallelLMHead)):
            if use_marlin:
                return GPTQMarlinLinearMethod(quant_args_marlin)
            else:
                return GPTQLinearMethod(quant_args)

        return None

    def apply_xpu_w4a16_quant_layer(self, layer, prefix: str):
        from vllm.model_executor.layers.fused_moe import FusedMoE

        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)

        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        if weight_bits != 4:
            raise NotImplementedError(
                f"INC on XPU only supports 4-bit quantization, "
                f"got weight_bits={weight_bits}."
            )
        if not sym:
            raise NotImplementedError(
                "INC W4A16 on XPU only supports symmetric quantization for now."
            )
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return INCXPULinearMethod(
                weight_bits=weight_bits,
                group_size=group_size,
                sym=sym,
            )
        if isinstance(layer, FusedMoE):
            return INCXPUMoEMethod(
                weight_bits=weight_bits,
                group_size=group_size,
                sym=sym,
                moe=layer.moe_config,
            )
        return None

    def apply_cpu_w4a16_quant_layer(self, layer, prefix: str):
        weight_bits, group_size, sym = self.get_layer_config(layer, prefix)
        if not self.check_quantized(weight_bits):
            if isinstance(layer, (LinearBase, ParallelLMHead)):
                return UnquantizedLinearMethod()
            else:
                return None

        if weight_bits != 4:
            raise NotImplementedError(
                f"INC on CPU only supports 4-bit quantization, "
                f"got weight_bits={weight_bits}."
            )
        if not sym:
            raise NotImplementedError(
                "INC W4A16 on CPU only supports symmetric quantization for now."
            )
        if isinstance(layer, (LinearBase, ParallelLMHead)):
            return self.apply_gptq_quant_layer(layer, prefix)
        # FusedMoE on CPU is not yet supported by INC; XPU has its own
        # INCXPUMoEMethod path above.
        return None

    def get_quant_method(self, layer: torch.nn.Module, prefix: str):
        if prefix and self.extra_config:
            for layer_name in self.extra_config:
                if (
                    layer_name == prefix or layer_name == f"model.{prefix}"
                ) and self.extra_config[layer_name].get("bits", 16) >= 16:
                    return UnquantizedLinearMethod()
        if current_platform.is_xpu():
            return self.apply_xpu_w4a16_quant_layer(layer, prefix)
        is_gptq = "gptq" in self.packing_format or "gptq" in self.backend
        if current_platform.is_cpu() and is_gptq:
            return self.apply_cpu_w4a16_quant_layer(layer, prefix)
        if is_gptq:
            return self.apply_gptq_quant_layer(layer, prefix)
        if "awq" in self.packing_format or "awq" in self.backend:
            return self.apply_awq_quant_layer(layer, prefix)

        raise NotImplementedError(
            f"Unsupported quantization configuration for layer '{prefix}'. "
            f"Platform: CPU={current_platform.is_cpu()}. "
            f"Platform: XPU={current_platform.is_xpu()}. "
            f"Format: {self.packing_format}, Backend: {self.backend}."
        )

    @classmethod
    def override_quantization_method(
        cls, hf_quant_cfg, user_quant, hf_config=None
    ) -> "QuantizationMethods | None":
        """Override the `auto-round` method to `inc`.

        On XPU, also claim vanilla GPTQ sym int4 desc_act=false checkpoints:
        gptq_marlin / awq_marlin gate on CUDA/CPU, and the moe_wna16 fallback
        runs Triton kernels that don't execute on Intel GPUs. INC routes the
        linear path through INCXPULinearMethod (oneDNN int4_gemm_w4a16) and
        the MoE path through INCXPUMoEMethod (vllm-xpu-kernels
        xpu_fused_moe(is_int4=True)) — the only working W4A16 path on XPU.
        """
        quant_method = hf_quant_cfg.get("quant_method", None)
        if quant_method == "auto-round":
            return cls.get_name()
        # On XPU, also claim vanilla GPTQ sym int4 desc_act=false. Honor an
        # explicit user `--quantization` flag: only take over when the user
        # didn't ask for a specific method or asked for `inc` directly, so
        # `--quantization gptq_marlin` still routes through the marlin path
        # for users who want it (it would then no-op on XPU; that's the
        # user's call, not ours).
        if (
            current_platform.is_xpu()
            and quant_method == "gptq"
            and user_quant in (None, "inc")
        ):
            bits = hf_quant_cfg.get("bits")
            sym = hf_quant_cfg.get("sym", True)
            desc_act = hf_quant_cfg.get("desc_act", False)
            if bits == 4 and sym is True and not desc_act:
                return cls.get_name()
        return None


class INCXPULinearMethod(LinearMethodBase):
    """XPU linear method for INC w4a16 GPTQ quantization (symmetric only).

    Repacks GPTQ weights from [in_packed, out] to oneDNN [out, in_packed]
    layout and calls torch.ops._xpu_C.int4_gemm_w4a16.

    GPTQ format: qweight [in_packed, out] with sequential nibble order.

    Note: Asymmetric quantization (sym=false) is not for now.

    FIXME(yiliu30): Refine the implementation to reuse XPUwNa16LinearKernel.
    """

    def __init__(self, weight_bits: int, group_size: int, sym: bool):
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.pack_factor = 32 // weight_bits

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del output_size  # Unused.
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")
        scales_and_zp_size = input_size_per_partition // self.group_size

        # GPTQ: qweight [in // pack_factor, out] packed along input dim
        qweight = PackedvLLMParameter(
            data=torch.empty(
                input_size_per_partition // self.pack_factor,
                output_size_per_partition,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=0,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )
        # scales: [num_groups, out] params_dtype
        scales = GroupQuantScaleParameter(
            data=torch.empty(
                scales_and_zp_size,
                output_size_per_partition,
                dtype=params_dtype,
            ),
            input_dim=0,
            output_dim=1,
            weight_loader=weight_loader,
        )
        # qzeros: [num_groups, out // pack_factor] int32
        qzeros = PackedvLLMParameter(
            data=torch.empty(
                scales_and_zp_size,
                output_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=0,
            output_dim=1,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )

        layer.register_parameter("qweight", qweight)
        layer.register_parameter("scales", scales)
        layer.register_parameter("qzeros", qzeros)

        # GPTQ checkpoints may include g_idx for activation reordering.
        # Register it so the weight loader doesn't error on unexpected keys.
        g_idx = RowvLLMParameter(
            data=torch.tensor(
                [i // self.group_size for i in range(input_size_per_partition)],
                dtype=torch.int32,
            ),
            input_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("g_idx", g_idx)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Repack GPTQ weights into kernel-ready NT layout."""
        device = layer.qweight.data.device

        # oneDNN int4 kernel requires strides[0]==1 ("NT format"), but GPTQ
        # checkpoint is [K_packed, N] contiguous with strides (N, 1).
        # Two transposes are needed — neither alone can achieve this:
        #   1. .t().contiguous() → [N, K_packed] contiguous in memory
        #   2. .t()              → [K_packed, N] view with strides (1, K_packed)
        # The result has the same logical shape but strides[0]==1 as required.
        qweight_ct = layer.qweight.data.t().contiguous()
        layer.qweight = Parameter(qweight_ct.t(), requires_grad=False)

        # Scales: [num_groups, out] — no change needed
        layer.scales = Parameter(layer.scales.data, requires_grad=False)

        # Symmetric: GPTQ v1 stores qzeros=7, effective zp = 7+1 = 8
        # Kernel expects int8 scalar = 8
        layer.qzeros = Parameter(
            torch.tensor([8], dtype=torch.int8, device=device),
            requires_grad=False,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # qweight is already in NT layout [K_packed, N] (strides (1, K_packed))
        # from process_weights_after_loading — pass directly to kernel.
        out_shape = x.shape[:-1] + (layer.qweight.shape[1],)
        reshaped_x = x.reshape(-1, x.shape[-1])
        out = torch.ops._xpu_C.int4_gemm_w4a16(
            reshaped_x,
            layer.qweight,
            bias,
            layer.scales,
            layer.qzeros,
            self.group_size,
            None,  # g_idx not needed: desc_act is always False for INC models
        )
        return out.reshape(out_shape)


class INCXPUMoEMethod(FusedMoEMethodBase):
    """W4A16 INT4-symmetric MoE on Intel XPU via vllm-xpu-kernels.

    Companion to ``INCXPULinearMethod``. Routes through the WNA16 MoE
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
        weight_bits: int,
        group_size: int,
        sym: bool,
        moe: FusedMoEConfig,
    ) -> None:
        super().__init__(moe)
        if weight_bits != 4:
            raise NotImplementedError(
                f"INC XPU MoE only supports 4-bit quantization, "
                f"got weight_bits={weight_bits}."
            )
        if not sym:
            raise NotImplementedError(
                "INC XPU MoE only supports symmetric quantization for now."
            )
        self.weight_bits = weight_bits
        self.group_size = group_size
        self.sym = sym
        self.pack_factor = 32 // weight_bits
        self.input_dtype: torch.dtype | None = None

        weight_key = QuantKey(
            scalar_types.uint4b8, kInt4StaticGroupScale, symmetric=True
        )
        self.wna16_moe_backend, self.experts_cls = select_wna16_moe_backend(
            moe, weight_key, weight_bits
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
