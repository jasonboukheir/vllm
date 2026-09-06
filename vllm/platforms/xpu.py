# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
from typing import TYPE_CHECKING, Any

import torch

# import custom ops, trigger op registration
import vllm_xpu_kernels._C  # noqa
import vllm_xpu_kernels._moe_C  # noqa
import vllm_xpu_kernels._xpu_C  # noqa

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from .interface import DeviceCapability, Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.config.kernel import IrOpPriorityConfig
    from vllm.v1.attention.selector import AttentionSelectorConfig
else:
    VllmConfig = None

logger = init_logger(__name__)

# These switches were useful while qualifying the beta implementation, but are
# deliberately not part of the release contract.  Fail closed rather than
# silently running a different cache implementation when an old launch
# environment is reused.
_RETIRED_KVARN_ENV_VARS = (
    "KVARN_QUANT_SLIDING",
    "KVARN_FLUSH_INDEX_MATERIALIZATION",
    "KVARN_FLUSH_WRITER",
    "KVARN_SINKHORN_SOURCE",
    "KVARN_FORWARD_POOL_ENSURE",
    "KVARN_QLEN1_INLINE_PLAN",
    "KVARN_METADATA_LIFECYCLE",
    "KVARN_NATIVE_XPU_LAYER",
    "KVARN_FAST_FLUSH",
    "KVARN_DUMP_TILES",
    "KVARN_FUSED_VERIFY",
    "KVARN_FUSED_VERIFY_MAXQ",
    "KVARN_FUSED_VERIFY_MIN_BLOCKS",
    "KVARN_FUSED_DECODE",
    "KVARN_RTN_QUANTILE",
    "KVARN_SINKHORN_ITERS",
    "KVARN_SINK_TOKENS",
    "KVARN_PREFILL_FP16_WINDOW_BLOCKS",
    "KVARN_DECODE_FP16_WINDOW_BLOCKS",
    "KVARN_DECODE_FP16_LOW_WATER_BLOCKS",
    "KVARN_DECODE_FLUSH_SCOPE",
    "KVARN_NATIVE_XPU",
    "KVARN_SPLIT_K",
    "KVARN_NUM_KV_SPLITS",
    "KVARN_SHARED_VERIFY",
    "KVARN_ONEDNN_DETERMINISTIC",
    "VLLM_KVARN_DEFER_PREFILL_FLUSH",
)


def _check_kvarn_beta_unsupported_config(
    vllm_config: "VllmConfig", cudagraph_none: object
) -> None:
    """Fail clearly for combinations outside the XPU KVarN beta contract."""
    cache_dtype = vllm_config.cache_config.cache_dtype
    if not isinstance(cache_dtype, str) or not cache_dtype.startswith("kvarn_"):
        return

    retired = [name for name in _RETIRED_KVARN_ENV_VARS if name in os.environ]
    retired.extend(name for name in os.environ if name.startswith("KVARN_NATIVE_XPU_"))
    if retired:
        raise ValueError(
            "retired KVarN experiment environment override(s): "
            + ", ".join(retired)
            + "; remove them and use the qualified release defaults"
        )

    if vllm_config.speculative_config is not None:
        raise ValueError(
            "XPU KVarN beta does not support speculative decoding/MTP; "
            "disable it or use --kv-cache-dtype=auto"
        )
    if vllm_config.use_v2_model_runner:
        raise ValueError(
            "XPU KVarN beta requires Model Runner V1; unset "
            "VLLM_USE_V2_MODEL_RUNNER=1 or use --kv-cache-dtype=auto"
        )
    if vllm_config.compilation_config.cudagraph_mode != cudagraph_none:
        raise ValueError(
            "XPU KVarN beta does not support graph mode; use --enforce-eager "
            "or set cudagraph_mode=NONE, or use --kv-cache-dtype=auto"
        )
    if vllm_config.cache_config.enable_prefix_caching:
        raise ValueError(
            "XPU KVarN beta does not support prefix caching; disable it "
            "or use --kv-cache-dtype=auto"
        )
    multimodal_config = getattr(vllm_config.model_config, "multimodal_config", None)
    language_model_only = bool(getattr(multimodal_config, "language_model_only", False))
    if vllm_config.model_config.is_multimodal_model and not language_model_only:
        hf_config = getattr(vllm_config.model_config, "hf_config", None)
        if getattr(hf_config, "model_type", None) != "qwen3_5":
            raise ValueError(
                "XPU KVarN beta vision/multimodal support is limited to Qwen3.5 "
                "image inputs; use --language-model-only or --kv-cache-dtype=auto"
            )
        if multimodal_config.get_limit_per_prompt("video") != 0:
            raise ValueError(
                "XPU KVarN beta vision requires video inputs disabled; "
                "use --limit-mm-per-prompt '{\"video\":0}' or --kv-cache-dtype=auto"
            )


def get_mem_info_wrapper(
    device: int | str | torch.device | None = None,
) -> tuple[int, int]:
    """
    Get memory info for a device, compatible with torch.accelerator.get_memory_info API.

    Args:
        device: Device specification. Can be:
            - None: Use current XPU device
            - int: Device index
            - str: Device string (e.g., "xpu:0", "xpu")
            - torch.device: Device object

    Returns:
        Tuple[int, int]: (free_memory, total_memory) in bytes
    """
    # Handle None - use current device
    if device is None:
        device = torch.xpu.current_device()

    # Handle torch.device objects
    elif isinstance(device, torch.device):
        if device.type != "xpu":
            raise RuntimeError(f"Expected 'xpu' device, got '{device.type}'")
        # If device index is not specified, use current device
        device = (
            device.index if device.index is not None else torch.xpu.current_device()
        )

    # Handle string device specifications (e.g., "xpu:0", "xpu")
    elif isinstance(device, str):
        if not device.startswith("xpu"):
            raise RuntimeError(f"Expected 'xpu' device string, got '{device}'")
        # Parse device string
        parts = device.split(":")
        if len(parts) == 1:
            # "xpu" -> use current device
            device = torch.xpu.current_device()
        elif len(parts) == 2:
            # "xpu:0" -> use index 0
            try:
                device = int(parts[1])
            except ValueError as err:
                raise RuntimeError(
                    f"Invalid device index: '{device}', expected integer after ':'"
                ) from err
        else:
            raise RuntimeError(f"Invalid device string format: '{device}'")

    # At this point, device should be an int
    if isinstance(device, int):
        # bounds check
        device_count = torch.xpu.device_count()
        if not (0 <= device < device_count):
            raise ValueError(
                f"Invalid device index {device}, must be in range [0, {device_count})"
            )

    elif not isinstance(device, int):
        raise TypeError(
            f"device must be int, str, torch.device, or None, got {type(device)}"
        )

    # Call the underlying C++ implementation
    free, total = torch.ops._C_cache_ops.getMemoryInfo(device)

    return free, total


torch.accelerator.get_memory_info = get_mem_info_wrapper


class XPUPlatform(Platform):
    _enum = PlatformEnum.XPU
    device_name: str = "xpu"
    device_type: str = "xpu"
    dispatch_key: str = "XPU"
    # Intel XPU's device key is "GPU" for Ray.
    # see https://github.com/ray-project/ray/blob/6a5eb5865eeb9ccf058a79b44f107e327e360673/python/ray/_private/accelerators/intel_gpu.py#L20 # noqa: E501
    ray_device_key: str = "GPU"
    dist_backend: str = "xccl"  # xccl only
    device_control_env_var: str = "ZE_AFFINITY_MASK"
    _kvarn_request_stable_xe2_validated = False
    supported_quantization: list[str] = [
        "awq",
        "gptq",
        "auto_awq",
        "auto_gptq",
        "inc",
        "fp8",
        "deepseek_v4_fp8",
        "mxfp4",
        "mxfp8",
        "fp8_per_tensor",
        "fp8_per_block",
        "online",
        "gpt_oss_mxfp4",
        "modelopt",
        "compressed-tensors",
    ]

    @classmethod
    def import_kernels(cls) -> None:
        # Do not import vllm._C
        with contextlib.suppress(ImportError):
            import vllm._moe_C  # noqa: F401

    @classmethod
    def check_runner_kv_caches_multi_layer(cls) -> None:
        pass

    @classmethod
    def set_additional_forward_context(
        cls,
        *,
        attn_metadata: Any,
        vllm_config: "VllmConfig",
        dp_metadata: Any = None,
        num_tokens: int | None = None,
        num_tokens_across_dp: Any = None,
        cudagraph_runtime_mode: Any = None,
        ubatch_slices: Any = None,
        is_padding: torch.Tensor | None = None,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        from vllm.model_executor.determinism.request_stable_linear import (
            XPU_KVARN_CANONICAL_LINEAR_ROWS,
            XPU_KVARN_REQUEST_SLICES_KEY,
            XPUKvarnRequestSlices,
            use_xpu_kvarn_request_stable_context,
        )

        if not use_xpu_kvarn_request_stable_context(vllm_config):
            return {}
        if not cls._kvarn_request_stable_xe2_validated:
            if not torch.ops._xpu_C.is_xe2_arch():
                raise RuntimeError(
                    "the frozen XPU KVarN request-stable profile requires an Xe2 device"
                )
            cls._kvarn_request_stable_xe2_validated = True
        if attn_metadata is None or attn_metadata == {}:
            # Profiling and warmup passes can omit attention metadata.
            return {}
        if (
            getattr(cudagraph_runtime_mode, "name", None) != "NONE"
            or dp_metadata is not None
            or num_tokens_across_dp is not None
            or ubatch_slices is not None
            or is_padding is not None
            or not isinstance(attn_metadata, dict)
        ):
            raise RuntimeError(
                "the frozen XPU KVarN request-stable profile requires an eager, "
                "unpadded, non-ubatched forward"
            )

        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata
        from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

        static_context = vllm_config.compilation_config.static_forward_context
        qwen_gdn_names = {
            name
            for name, layer in static_context.items()
            if getattr(layer, "mamba_type", None)
            is MambaAttentionBackendEnum.QWEN_GDN_ATTN
        }
        if not qwen_gdn_names:
            raise RuntimeError(
                "the frozen XPU KVarN profile has no registered Qwen GDN layers"
            )
        missing_names = qwen_gdn_names.difference(attn_metadata)
        if missing_names:
            raise RuntimeError(
                "the frozen XPU KVarN forward is missing Qwen GDN metadata for "
                + ", ".join(sorted(missing_names))
            )

        if any(
            not isinstance(attn_metadata[name], GDNAttentionMetadata)
            for name in qwen_gdn_names
        ):
            raise RuntimeError(
                "the frozen XPU KVarN profile received non-GDN Qwen metadata"
            )
        metadata_by_id = {
            id(attn_metadata[name]): attn_metadata[name] for name in qwen_gdn_names
        }

        expected_slices: tuple[tuple[int, int, int, bool], ...] | None = None
        expected_metadata_signature: tuple[int, ...] | None = None
        for metadata in metadata_by_id.values():
            disallowed_spec_fields = (
                metadata.spec_query_start_loc,
                metadata.spec_state_indices_tensor,
                metadata.spec_sequence_masks,
                metadata.spec_token_indx,
                metadata.non_spec_token_indx,
                metadata.num_accepted_tokens,
            )
            if (
                metadata.num_spec_decodes != 0
                or metadata.num_spec_decode_tokens != 0
                or any(value is not None for value in disallowed_spec_fields)
            ):
                raise RuntimeError(
                    "XPU KVarN request-stable operators require ordinary no-spec "
                    "GDN metadata"
                )

            boundaries = metadata.non_spec_query_start_loc_cpu
            positions = metadata.non_spec_num_computed_tokens_cpu
            phases = metadata.non_spec_is_prefilling_cpu
            num_requests = metadata.num_decodes + metadata.num_prefills
            if boundaries is None or len(boundaries) != num_requests + 1:
                raise RuntimeError(
                    "XPU KVarN GDN metadata is missing complete CPU request boundaries"
                )
            if positions is None or len(positions) != num_requests:
                raise RuntimeError(
                    "XPU KVarN GDN metadata is missing CPU request start positions"
                )
            if any(position < 0 for position in positions):
                raise RuntimeError(
                    "XPU KVarN GDN request start positions must be non-negative"
                )
            if phases is None or len(phases) != num_requests:
                raise RuntimeError(
                    "XPU KVarN GDN metadata is missing CPU request phases"
                )
            if boundaries[0] != 0 or any(
                stop <= start for start, stop in zip(boundaries, boundaries[1:])
            ):
                raise RuntimeError(
                    "XPU KVarN GDN CPU request boundaries must be positive and "
                    "contiguous from row 0"
                )
            if (
                boundaries[-1] != metadata.num_actual_tokens
                or boundaries[-1] != num_tokens
                or metadata.num_decode_tokens + metadata.num_prefill_tokens
                != metadata.num_actual_tokens
            ):
                raise RuntimeError(
                    "XPU KVarN GDN request boundaries do not cover the model rows"
                )
            if (
                boundaries[metadata.num_decodes] != metadata.num_decode_tokens
                or boundaries[-1] - boundaries[metadata.num_decodes]
                != metadata.num_prefill_tokens
            ):
                raise RuntimeError(
                    "XPU KVarN GDN request boundaries disagree with decode/prefill "
                    "token counts"
                )

            request_slices = tuple(
                (start, stop, position, is_prefill)
                for start, stop, position, is_prefill in zip(
                    boundaries, boundaries[1:], positions, phases
                )
            )
            metadata_signature = (
                metadata.num_decodes,
                metadata.num_prefills,
                metadata.num_decode_tokens,
                metadata.num_prefill_tokens,
                metadata.num_actual_tokens,
            )
            for start, stop, position, is_prefill in request_slices:
                span = stop - start
                if (
                    is_prefill or span > 1
                ) and position % XPU_KVARN_CANONICAL_LINEAR_ROWS != 0:
                    raise RuntimeError(
                        "XPU KVarN multi-row projection did not start on the "
                        "canonical 64-row grid"
                    )
            if expected_slices is None:
                expected_slices = XPUKvarnRequestSlices(request_slices)
                expected_metadata_signature = metadata_signature
            elif metadata_signature != expected_metadata_signature:
                raise RuntimeError("XPU KVarN GDN layers disagree on dispatch counts")
            elif request_slices != expected_slices:
                raise RuntimeError(
                    "XPU KVarN GDN layers disagree on packed request metadata"
                )

        assert expected_slices is not None
        return {XPU_KVARN_REQUEST_SLICES_KEY: expected_slices}

    @classmethod
    def get_attn_backend_cls(
        cls,
        selected_backend: "AttentionBackendEnum",
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        # TurboQuant KV cache: route directly to TQ backend
        kv_cache_dtype = attn_selector_config.kv_cache_dtype
        if kv_cache_dtype is not None and kv_cache_dtype.startswith("turboquant_"):
            logger.info_once("Using TurboQuant attention backend.")
            return AttentionBackendEnum.TURBOQUANT.get_path()
        if (
            kv_cache_dtype is not None
            and kv_cache_dtype.startswith("kvarn_")
            and not attn_selector_config.use_mla
        ):
            logger.info_once("Using KVarN attention backend on XPU.")
            return AttentionBackendEnum.KVARN.get_path()

        dtype = attn_selector_config.dtype
        if attn_selector_config.use_sparse:
            logger.info_once("Using XPU MLA Sparse backend.")
            return AttentionBackendEnum.XPU_MLA_SPARSE.get_path()
        if attn_selector_config.use_mla:
            logger.info_once("Using Triton MLA backend on V1 engine.")
            return AttentionBackendEnum.TRITON_MLA.get_path()
        if selected_backend == AttentionBackendEnum.TRITON_ATTN:
            logger.info_once("Using Triton backend.")
            return AttentionBackendEnum.TRITON_ATTN.get_path()
        elif attn_selector_config.use_mm_prefix:
            # Flash Attention on XPU has no FA4 kernel, so it cannot apply the
            # multimodal prefix-LM bidirectional mask. Honor an explicit Flash
            # Attention request (for text-only workloads); otherwise fall back
            # to Triton Attention, which supports mm_prefix.
            if selected_backend == AttentionBackendEnum.FLASH_ATTN:
                logger.warning_once(
                    "Using Flash Attention on XPU for a multimodal prefix-LM "
                    "model because it was explicitly requested. The prefix-LM "
                    "bidirectional mask cannot be applied, so image/video "
                    "inputs will produce incorrect results; only use this for "
                    "text-only workloads."
                )
                return AttentionBackendEnum.FLASH_ATTN.get_path()
            logger.warning_once(
                "Flash Attention on XPU does not support multimodal prefix-LM "
                "attention. Falling back to Triton Attention backend."
            )
            return AttentionBackendEnum.TRITON_ATTN.get_path()
        elif dtype == torch.float32:
            logger.warning_once(
                "Flash Attention on XPU does not support float32 dtype. "
                "Falling back to Triton Attention backend."
            )
            return AttentionBackendEnum.TRITON_ATTN.get_path()
        elif selected_backend == AttentionBackendEnum.FLASH_ATTN:
            logger.info_once("Using Flash Attention backend.")
            return AttentionBackendEnum.FLASH_ATTN.get_path()
        elif selected_backend:
            raise ValueError(
                f"Invalid attention backend for {cls.device_name}, "
                f"with use_mla: {attn_selector_config.use_mla}"
            )

        logger.info_once("Using Flash Attention backend.")
        return AttentionBackendEnum.FLASH_ATTN.get_path()

    @classmethod
    def get_supported_vit_attn_backends(cls) -> list["AttentionBackendEnum"]:
        return [
            AttentionBackendEnum.FLASH_ATTN,
            AttentionBackendEnum.TRITON_ATTN,
            AttentionBackendEnum.TORCH_SDPA,
        ]

    @classmethod
    def get_vit_attn_backend(
        cls,
        head_size: int,
        dtype: torch.dtype,
        backend: "AttentionBackendEnum | None" = None,
    ) -> "AttentionBackendEnum":
        if dtype == torch.float32:
            logger.warning_once(
                "Flash Attention on XPU does not support float32 dtype. "
                "Falling back to Triton Attention backend for vit attention."
            )
            return AttentionBackendEnum.TRITON_ATTN

        if backend is not None:
            assert backend in cls.get_supported_vit_attn_backends(), (
                f"Backend {backend} is not supported for vit attention. "
                f"Supported backends are: "
                f"{cls.get_supported_vit_attn_backends()}."
            )
            logger.info_once(f"Using backend {backend} for vit attention")
            return backend

        logger.info_once(
            f"Using backend {AttentionBackendEnum.FLASH_ATTN} for vit attention"
        )
        return AttentionBackendEnum.FLASH_ATTN

    @classmethod
    def set_device(cls, device: torch.device) -> None:
        """
        Set the device for the current platform.
        """
        torch.xpu.set_device(device)

    @classmethod
    def manual_seed_all(cls, seed: int) -> None:
        torch.xpu.manual_seed_all(seed)

    @classmethod
    def get_device_capability(
        cls,
        device_id: int = 0,
    ) -> DeviceCapability | None:
        # capacity format differs from cuda's and will cause unexpected
        # failure, so use None directly
        return None

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return torch.xpu.get_device_name(device_id)

    @classmethod
    def get_punica_wrapper(cls) -> str:
        xpu_use_triton_kernel = os.getenv("XPU_USE_TRITON_KERNEL", "0") == "1"
        if not xpu_use_triton_kernel:
            return "vllm.lora.punica_wrapper.punica_xpu.PunicaWrapperXPU"
        else:
            return "vllm.lora.punica_wrapper.punica_gpu.PunicaWrapperGPU"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        device_props = torch.xpu.get_device_properties(device_id)
        return device_props.total_memory

    @classmethod
    def inference_mode(cls):
        return torch.no_grad()

    @classmethod
    def get_static_graph_wrapper_cls(cls) -> str:
        return "vllm.compilation.cuda_graph.CUDAGraphWrapper"

    @classmethod
    def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
        # lazy import to avoid circular import
        from vllm.config import CUDAGraphMode

        _check_kvarn_beta_unsupported_config(vllm_config, CUDAGraphMode.NONE)

        compilation_config = vllm_config.compilation_config
        if compilation_config.compile_sizes is None:
            compilation_config.compile_sizes = []

        # lazy import to avoid circular import
        from vllm.utils.torch_utils import supports_xpu_graph

        if not supports_xpu_graph():
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            logger.warning_once(
                "XPU Graph is not supported in the current PyTorch version, "
                "disabling cudagraph_mode."
            )
        elif not envs.VLLM_XPU_ENABLE_XPU_GRAPH:
            compilation_config.cudagraph_mode = CUDAGraphMode.NONE
            logger.warning_once(
                "XPU Graph is disabled by environment variable, "
                "please set VLLM_XPU_ENABLE_XPU_GRAPH=1 to enable it."
            )
        else:
            logger.warning_once(
                "XPU Graph support is experimental and currently only supports "
                "single-GPU execution."
            )

        # Disable fusion passes not yet supported on XPU.
        from vllm.config.compilation import CompilationMode

        pass_config = compilation_config.pass_config
        fusion_passes_to_disable = {
            "fuse_gemm_comms": "Async TP",
            "fuse_allreduce_rms": "AllReduce + RMSNorm fusion",
            "fuse_attn_quant": "Attention + quant fusion",
            "fuse_act_padding": "Activation + padding fusion",
            "fuse_rope_kvcache": "RoPE + KV cache fusion",
            "fuse_rope_kvcache_cat_mla": "RoPE + KV cache + MLA fusion",
        }
        if compilation_config.mode != CompilationMode.NONE:
            for flag, feature_name in fusion_passes_to_disable.items():
                if getattr(pass_config, flag):
                    logger.warning_once(
                        "Feature %r is not yet supported on XPU and will be disabled.",
                        feature_name,
                    )
                    setattr(pass_config, flag, False)

        # UVA-offloaded weights are host USM allocations, which Inductor's
        # static Triton launcher rejects ("Pointer argument doesn't reference
        # XPU device memory"). Fall back to Triton's own launcher. Remove once
        # the released torch contains pytorch/pytorch#188240, which relaxes
        # that check to any memory type known by the driver.
        offload_config = vllm_config.offload_config
        uva_offloading = offload_config.offload_backend == "uva" or (
            offload_config.offload_backend == "auto"
            and offload_config.prefetch.offload_group_size == 0
            and offload_config.uva.cpu_offload_gb > 0
        )
        if (
            uva_offloading
            and not envs.VLLM_WEIGHT_OFFLOADING_DISABLE_UVA
            and compilation_config.mode != CompilationMode.NONE
        ):
            compilation_config.inductor_compile_config.setdefault(
                "use_static_cuda_launcher", False
            )
            logger.info_once(
                "Disabling Inductor's static Triton launcher because UVA "
                "weight offloading is enabled."
            )

        # check and update parallel config
        parallel_config = vllm_config.parallel_config
        # Only override worker_cls if it's still the default "auto"
        # This allows custom workers (like vllm-omni workers) to be used on XPU
        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.xpu_worker.XPUWorker"
        if vllm_config.kv_transfer_config is not None:
            vllm_config.kv_transfer_config.enable_permute_local_kv = True

        # In some cases, the internal memory type cache can misdetect GPU
        # memory as host memory, also leading to invalid memory access.
        # This cache can be disabled by setting UCX_MEMTYPE_CACHE=n.
        # ref. https://openucx.readthedocs.io/en/master/faq.html
        os.environ["UCX_MEMTYPE_CACHE"] = "n"

        # spawn is the only supported multiprocessing method on XPU
        if "VLLM_WORKER_MULTIPROC_METHOD" not in os.environ:
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        # XPU requires graceful shutdown to allow oneCCL/Level Zero resources
        # to be properly released. Without this, subsequent server startups on
        # the same devices may hang during CCL initialization.
        if vllm_config.shutdown_timeout == 0:
            vllm_config.shutdown_timeout = 5
            logger.info(
                "XPU platform: set server shutdown_timeout=%d.",
                vllm_config.shutdown_timeout,
            )

        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        scheduler_config = vllm_config.scheduler_config
        cache_dtype = getattr(cache_config, "cache_dtype", None)
        if (
            model_config is not None
            and isinstance(cache_dtype, str)
            and cache_dtype.startswith("kvarn_")
            and not cache_dtype.startswith("kvarn_mla")
            and not getattr(model_config, "use_mla", False)
        ):
            from vllm.model_executor.layers.quantization.kvarn.config import (
                KVarNConfig,
            )

            head_size = model_config.get_head_size()
            if head_size not in (128, 256, 512):
                raise ValueError(
                    f"{cache_dtype} requires head_dim in (128, 256, 512), but "
                    f"this model has head_dim={head_size}; use a different "
                    "--kv-cache-dtype for this model."
                )

            skip_layers = cache_config.kv_cache_dtype_skip_layers
            if "sliding_window" not in skip_layers:
                skip_layers.append("sliding_window")

            kvarn_config = KVarNConfig.from_cache_dtype(cache_dtype, head_size)
            weight_bytes = kvarn_config.estimate_weight_bytes(
                model_config.model,
                tensor_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
            )
            supported = kvarn_config.max_supported_seqs(
                total_gpu_bytes=cls.get_device_total_memory(),
                num_kv_heads=model_config.get_num_kv_heads(vllm_config.parallel_config),
                num_layers=KVarNConfig.num_kvarn_layers(
                    model_config, vllm_config.parallel_config
                ),
                max_num_batched_tokens=scheduler_config.max_num_batched_tokens,
                gpu_memory_utilization=cache_config.gpu_memory_utilization,
                weight_bytes=weight_bytes,
            )
            if scheduler_config.max_num_seqs > supported:
                logger.warning(
                    "KVarN (%s): capping max_num_seqs %d -> %d so the XPU "
                    "fp16 tail pool fits its memory budget.",
                    cache_dtype,
                    scheduler_config.max_num_seqs,
                    supported,
                )
                scheduler_config.max_num_seqs = supported

    @classmethod
    def update_block_size_for_backend(cls, vllm_config: "VllmConfig") -> None:
        super().update_block_size_for_backend(vllm_config)
        from vllm.config.vllm import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import (
            AttentionLayerBase,
        )
        from vllm.utils.math_utils import cdiv

        cache_config = vllm_config.cache_config
        # special fix for GDN since kernel only supports block size dividable by 64
        attn_layers = get_layers_from_vllm_config(
            vllm_config,
            AttentionLayerBase,  # type: ignore[type-abstract]
        )

        kernel_block_size = None
        for layer in attn_layers.values():
            b = layer.get_attn_backend()
            if b.get_name() in ("GDN_ATTN", "QWEN_GDN_ATTN"):
                kernel_block_size = 64
                break

        if kernel_block_size is None:
            return
        new_block_size = (
            cdiv(cache_config.block_size, kernel_block_size) * kernel_block_size
        )
        if new_block_size == cache_config.block_size:
            return

        if cache_config.mamba_cache_mode == "align":
            cache_config.mamba_block_size = new_block_size
        original_mamba_page_size_padded = cache_config.mamba_page_size_padded
        if cache_config.mamba_page_size_padded is not None:
            attn_page_size_1_token = (
                cache_config.mamba_page_size_padded // cache_config.block_size
            )
            cache_config.mamba_page_size_padded = (
                new_block_size * attn_page_size_1_token
            )
        cache_config.block_size = new_block_size
        logger.info(
            "[XPU]Setting attention block size to %d tokens to ensure multiple of %d, "
            "set mamba_page_size_padded to %d bytes accordingly, before was %d bytes.",
            new_block_size,
            kernel_block_size,
            cache_config.mamba_page_size_padded,
            original_mamba_page_size_padded,
        )

    @classmethod
    def support_hybrid_kv_cache(cls) -> bool:
        return True

    @classmethod
    def support_static_graph_mode(cls) -> bool:
        return True

    @classmethod
    def is_pin_memory_available(cls):
        return True

    @classmethod
    def get_current_memory_usage(
        cls, device: torch.types.Device | None = None
    ) -> float:
        torch.xpu.empty_cache()
        torch.xpu.reset_peak_memory_stats(device)
        return torch.xpu.max_memory_allocated(device)

    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        return torch.float8_e4m3fn

    @classmethod
    def is_data_center_gpu(cls) -> bool:
        device_name = cls.get_device_name().lower()
        return device_name.count("data center gpu") > 0

    @classmethod
    def get_device_communicator_cls(cls) -> str:
        if not torch.distributed.is_xccl_available():
            # Supports xccl with PyTorch versions >= 2.8.0.dev for XPU platform
            logger.warning(
                "xccl is not enabled in this torch build, communication"
                " is not available."
            )
        return "vllm.distributed.device_communicators.xpu_communicator.XpuCommunicator"  # noqa

    @classmethod
    def supports_fp8(cls) -> bool:
        return True

    @classmethod
    def get_default_ir_op_priority(
        cls, vllm_config: "VllmConfig"
    ) -> "IrOpPriorityConfig":
        from vllm.config.compilation import CompilationMode
        from vllm.config.kernel import IrOpPriorityConfig

        # Native used by default when compiling,
        # use fused kernels where available when no codegen
        cc = vllm_config.compilation_config
        using_inductor = cc.backend == "inductor" and cc.mode != CompilationMode.NONE
        default = ["native"] if using_inductor else ["vllm_c", "native"]

        return IrOpPriorityConfig.with_default(default)

    @classmethod
    def device_count(cls) -> int:
        return torch.xpu.device_count()

    @classmethod
    def check_if_supports_dtype(cls, dtype: torch.dtype):
        if dtype == torch.bfloat16:  # noqa: SIM102
            device_name = cls.get_device_name().lower()
            # client gpu a770
            if device_name.count("a770") > 0:
                raise ValueError(
                    "Intel Arc A770 have bfloat16 accuracy known issue. "
                    "You can use float16 instead by explicitly setting the "
                    "`dtype` flag in CLI, for example: --dtype=half."
                )

    @classmethod
    def opaque_attention_op(cls) -> bool:
        return True

    @classmethod
    def insert_blocks_to_device(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from src_cache to dst_cache on XPU."""
        _src_cache = src_cache[src_block_indices]
        dst_cache[dst_block_indices] = _src_cache.to(dst_cache.device)

    @classmethod
    def swap_out_blocks_to_host(
        cls,
        src_cache: torch.Tensor,
        dst_cache: torch.Tensor,
        src_block_indices: torch.Tensor,
        dst_block_indices: torch.Tensor,
    ) -> None:
        """Copy blocks from XPU to host (CPU)."""
        _src_cache = src_cache[src_block_indices]
        dst_cache[dst_block_indices] = _src_cache.cpu()

    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        return torch.xpu.get_device_properties(device_id).max_compute_units

    @classmethod
    def use_custom_op_collectives(cls) -> bool:
        return True
