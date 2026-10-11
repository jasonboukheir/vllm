# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU prefix storage for the single-device KVarN hybrid serving profile."""

import time

import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.byte_pool import (
    BytePoolOffloadingManager,
    HostPages,
)


class IndependentCPUOffloadingSpec(OffloadingSpec):
    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        if (
            config.parallel.world_size != 1
            or self.blocks_per_chunk != 1
            or config.model.dtype != "kvarn_k4v2_g128_compact"
            or config.enable_kv_cache_events
            or config.canonical_layout
        ):
            raise ValueError(
                "Independent CPU offload requires compact K4V2, one worker, "
                "one block per chunk, and no KV events or canonical layout"
            )
        self.capacity = int(config.extra_config.get("cpu_bytes_to_use", 0))
        if self.capacity <= 0:
            raise ValueError("cpu_bytes_to_use must be positive")
        self._manager = None
        self._worker = None

    def get_manager(self) -> BytePoolOffloadingManager:
        if self._manager is None:
            self._manager = BytePoolOffloadingManager(
                capacity=self.capacity,
                group_bytes={g.group_id: g.page_size_bytes for g in self.config.groups},
                sink_bytes={g.group_id: g.sink_size_bytes for g in self.config.groups},
                tokens_per_block={
                    g.group_id: g.tokens_per_block for g in self.config.groups
                },
            )
        return self._manager

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if self._worker is None:
            self._worker = IndependentCPUOffloadingWorker(
                self.config, kv_caches, self.capacity
            )
        return self._worker


class IndependentCPUOffloadingWorker(OffloadingWorker):
    """Copy byte ranges at a scheduler fence using the existing batched DMA op.

    Transfers complete before returning. This keeps KVarN's mutable FP16
    side buffers ordered with main-cache copies and metadata construction.
    The connector still handles admission, reuse, eviction and cache metrics.
    """

    def __init__(
        self, config: OffloadingConfig, caches: CanonicalKVCaches, capacity: int
    ):
        self.config = config
        self.caches = caches
        self.host = torch.empty(
            capacity, dtype=torch.uint8, device="cpu", pin_memory=True
        )
        self.finished: list[TransferResult] = []
        self.group_bytes = []
        for group, refs in zip(config.groups, caches.group_data_refs):
            size = sum(ref.page_size_bytes for ref in refs)
            if size != group.page_size_bytes:
                raise ValueError(
                    f"CPU/GPU cache byte accounting differs in group {group.group_id}: "
                    f"{group.page_size_bytes} vs {size}"
                )
            self.group_bytes.append(size)

    def _transfer(
        self, job_id: int, gpu: GPULoadStoreSpec, host: HostPages, store: bool
    ) -> bool:
        from vllm.v1.attention.backends.kvarn_attn import KVarNAttentionImpl

        start = time.perf_counter()
        assert len(gpu.block_ids) == len(host.pages)
        assert len(gpu.group_sizes) == len(self.config.groups)
        src_ptrs: list[int] = []
        dst_ptrs: list[int] = []
        sizes: list[int] = []
        offset = 0
        reloads: list[tuple[tuple[str, ...], list[int], dict[int, torch.Tensor]]] = []
        sidecar_bytes = 0
        for i, (group, count, refs) in enumerate(
            zip(self.config.groups, gpu.group_sizes, self.caches.group_data_refs)
        ):
            ids = gpu.block_ids[offset : offset + count].tolist()
            pages = host.pages[offset : offset + count]
            offset += count
            if not count:
                continue
            assert all(page.group_id == group.group_id for page in pages)
            sidecars = (
                KVarNAttentionImpl.prepare_cpu_offload(group.layer_names, ids)
                if store and group.sink_size_bytes
                else {}
            )
            restored_sinks: dict[int, torch.Tensor] = {}
            for block_id, page in zip(ids, pages):
                cursor = page.offset
                for ref in refs:
                    tensor = self.caches.tensors[ref.tensor_idx].tensor
                    assert 0 < block_id < tensor.shape[0]
                    device_ptr = (
                        tensor.data_ptr()
                        + block_id * tensor.stride(0) * tensor.element_size()
                    )
                    host_ptr = self.host.data_ptr() + cursor
                    src_ptrs.append(device_ptr if store else host_ptr)
                    dst_ptrs.append(host_ptr if store else device_ptr)
                    sizes.append(ref.page_size_bytes)
                    cursor += ref.page_size_bytes
                if page.has_sink:
                    assert group.sink_size_bytes > 0
                    assert cursor + group.sink_size_bytes <= page.offset + page.size
                    payload = self.host[cursor + 2 : cursor + group.sink_size_bytes]
                    if store:
                        sidecar = sidecars.get(block_id)
                        self.host[cursor] = int(sidecar is not None)
                        if sidecar is not None:
                            assert payload.numel() == sidecar.numel()
                            payload.copy_(sidecar)
                            sidecar_bytes += payload.numel()
                    elif self.host[cursor].item():
                        restored_sinks[block_id] = payload
                        sidecar_bytes += payload.numel()
            if not store and group.sink_size_bytes:
                reloads.append((group.layer_names, ids, restored_sinks))
        assert offset == len(host.pages)
        if src_ptrs:
            pointer_dtype = torch.uint64 if current_platform.is_xpu() else torch.int64
            ops.swap_blocks_batch(
                torch.tensor(src_ptrs, dtype=pointer_dtype),
                torch.tensor(dst_ptrs, dtype=pointer_dtype),
                torch.tensor(sizes, dtype=pointer_dtype),
            )
            current_platform.current_stream().synchronize()
        for layer_names, ids, sidecars in reloads:
            KVarNAttentionImpl.prepare_cpu_reload(layer_names, ids, sidecars)
        self.finished.append(
            TransferResult(
                job_id=job_id,
                success=True,
                transfer_size=sum(sizes) + sidecar_bytes,
                transfer_time=time.perf_counter() - start,
            )
        )
        return True

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(dst_spec, HostPages)
        return self._transfer(job_id, src_spec, dst_spec, store=True)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        assert isinstance(src_spec, HostPages)
        return self._transfer(job_id, dst_spec, src_spec, store=False)

    def get_finished(self) -> list[TransferResult]:
        finished, self.finished = self.finished, []
        return finished

    def wait(self, job_ids: set[int]) -> None:
        # Every submitted copy is already complete.
        pass
