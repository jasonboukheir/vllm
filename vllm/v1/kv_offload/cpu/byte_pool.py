# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, variable-size host storage for independent KV cache pools."""

import heapq
from collections.abc import Collection, Iterator
from dataclasses import dataclass, field

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.policies.base import ChunkStatus
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy
from vllm.v1.metrics.cache_hit_source import CacheHitSource


@dataclass(frozen=True)
class HostPage:
    offset: int
    size: int
    group_id: int
    has_sink: bool = False


@dataclass
class HostPages(LoadStoreSpec):
    pages: list[HostPage]


@dataclass
class _Entry:
    page: HostPage
    chunk: ChunkStatus


@dataclass
class _RequestAccess:
    owner: object
    generation: int
    keys: dict[OffloadKey, None] = field(default_factory=dict)
    finished: bool = False


class _BytePoolLRUPolicy(LRUCachePolicy):
    def eviction_candidates(self, protected: set[OffloadKey]) -> Iterator[OffloadKey]:
        """Inspect LRU victims without committing a variable-size allocation.

        The inherited evict() commits a fixed number of chunks. A byte arena
        instead needs enough adjacent free space for every offered page. Inspect
        a heap snapshot only when allocation runs out of room, then remove the
        chosen victims once the entire offer fits. Failed offers preserve both
        the arena and upstream's request recency and transfer pinning.
        """
        heap = list(self._heap)
        seen: set[OffloadKey] = set()
        while heap:
            entry = heapq.heappop(heap)
            key = entry[1]
            if self._is_current(entry) and key not in protected and key not in seen:
                seen.add(key)
                yield key

    def remove_allocated_victims(self, keys: list[OffloadKey]) -> None:
        for key in keys:
            self.remove(key)
        self._maybe_compact_heap()


class BytePoolOffloadingManager(OffloadingManager):
    """Use actual group byte sizes rather than padding every page to an SSM state.

    A single bounded arena holds all groups. Pending transfers pin their ranges;
    only completed, unreferenced pages are eligible for LRU eviction.
    """

    ALIGNMENT = 64

    def __init__(
        self,
        capacity: int,
        group_bytes: dict[int, int],
        sink_bytes: dict[int, int] | None = None,
        tokens_per_block: dict[int, int] | None = None,
    ):
        if capacity <= 0:
            raise ValueError("CPU prefix cache capacity must be positive")
        if not group_bytes or any(size <= 0 for size in group_bytes.values()):
            raise ValueError("Every CPU prefix cache group must have a positive size")
        self.capacity = capacity
        self.group_bytes = dict(group_bytes)
        self.sink_bytes = dict(sink_bytes or {})
        self.tokens_per_block = dict(tokens_per_block or {})
        for group, size in self.sink_bytes.items():
            if (
                group not in self.group_bytes
                or size < 0
                or (size and self.tokens_per_block.get(group, 0) <= 0)
            ):
                raise ValueError(
                    "Sink storage requires a valid cache group and block size"
                )
        self.entries: dict[OffloadKey, _Entry] = {}
        self._policy = _BytePoolLRUPolicy(cache_capacity=capacity // self.ALIGNMENT)
        self.free_ranges = [(0, capacity)]
        self._generation = 0

    def _size(self, key: OffloadKey, context: ReqContext) -> tuple[int, bool]:
        group = get_offload_group_idx(key)
        has_sink = (
            bool(self.sink_bytes.get(group))
            and context.get_offload_key_position(key) == self.tokens_per_block[group]
        )
        size = self.group_bytes[group]
        if has_sink:
            size += self.sink_bytes[group]
        return (
            (size + self.ALIGNMENT - 1) // self.ALIGNMENT
        ) * self.ALIGNMENT, has_sink

    @staticmethod
    def _release(ranges: list[tuple[int, int]], page: HostPage) -> None:
        merged: list[tuple[int, int]] = []
        for offset, size in sorted([*ranges, (page.offset, page.size)]):
            if merged and sum(merged[-1]) == offset:
                start, length = merged[-1]
                merged[-1] = (start, length + size)
            else:
                merged.append((offset, size))
        ranges[:] = merged

    @staticmethod
    def _allocate(ranges: list[tuple[int, int]], size: int) -> int | None:
        for i, (offset, length) in enumerate(ranges):
            if length >= size:
                ranges[i : i + 1] = (
                    [(offset + size, length - size)] if length > size else []
                )
                return offset
        return None

    def _record_accesses(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        state = req_context.get_state(_RequestAccess)
        if (
            state is None
            or state.owner is not self
            or state.generation != self._generation
        ):
            state = _RequestAccess(self, self._generation)
            req_context.set_state(state)
        new_keys = list(dict.fromkeys(key for key in keys if key not in state.keys))
        state.keys.update(dict.fromkeys(new_keys))
        if new_keys:
            self._policy.on_request_access(new_keys, req_context)

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        entry = self.entries.get(key)
        if entry is None:
            return LookupResult.MISS
        return LookupResult.HIT_PENDING if entry.chunk.ref_cnt < 0 else LookupResult.HIT

    def get_load_source(
        self, key: OffloadKey, req_context: ReqContext
    ) -> CacheHitSource:
        return CacheHitSource.HOST

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys, req_context)

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> HostPages:
        unique_keys = dict.fromkeys(keys)
        assert all(self.entries[key].chunk.ref_cnt >= 0 for key in unique_keys)
        for key in unique_keys:
            chunk = self.entries[key].chunk
            if chunk.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
            chunk.ref_cnt += 1
        self._record_accesses(keys, req_context)
        return HostPages([self.entries[key].page for key in keys])

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in dict.fromkeys(keys):
            chunk = self.entries[key].chunk
            assert chunk.ref_cnt > 0
            chunk.ref_cnt -= 1
            if chunk.ref_cnt == 0:
                self._policy.mark_evictable(key)

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        # Preserve the caller's group order, also when it repeats a key.
        missing = list(dict.fromkeys(key for key in keys if key not in self.entries))
        sizes = [self._size(key, req_context) for key in missing]
        if sum(size for size, _ in sizes) > self.capacity:
            return None
        protected = set(keys)
        free_ranges = list(self.free_ranges)
        candidates = self._policy.eviction_candidates(protected)
        evicted: list[OffloadKey] = []
        allocated: list[HostPage] = []
        for key, (size, has_sink) in zip(missing, sizes):
            offset = self._allocate(free_ranges, size)
            while offset is None:
                victim_key = next(candidates, None)
                if victim_key is None:
                    return None
                self._release(free_ranges, self.entries[victim_key].page)
                evicted.append(victim_key)
                offset = self._allocate(free_ranges, size)
            allocated.append(
                HostPage(offset, size, get_offload_group_idx(key), has_sink)
            )
        # Failed multi-page offers leave both the cache and free list intact.
        self._policy.remove_allocated_victims(evicted)
        for key in evicted:
            del self.entries[key]
        self.free_ranges = free_ranges
        for key, page in zip(missing, allocated):
            chunk = ChunkStatus(page.offset)
            self.entries[key] = _Entry(page, chunk)
            self._policy.insert(key, chunk, req_context)
        self._record_accesses(keys, req_context)
        return PrepareStoreOutput(missing, HostPages(allocated), evicted)

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        for key in dict.fromkeys(keys):
            entry = self.entries[key]
            assert entry.chunk.ref_cnt == -1
            if not success:
                self._release(self.free_ranges, self.entries.pop(key).page)
                self._policy.remove(key)
                continue
            entry.chunk.ref_cnt = 0
            self._policy.mark_evictable(key)

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._record_accesses((), req_context)
        return RequestOffloadingContext()

    def on_request_finished(self, req_context: ReqContext) -> None:
        state = req_context.get_state(_RequestAccess)
        if (
            state is None
            or state.owner is not self
            or state.generation != self._generation
            or state.finished
        ):
            return
        state.finished = True
        self._policy.on_request_finished([list(state.keys)], set(), set(), req_context)

    def reset_cache(self) -> None:
        # The connector flushes pending transfers before reusing these offsets
        # and suppresses their stale completion callbacks after reset.
        self.entries.clear()
        self._policy.clear()
        self.free_ranges = [(0, self.capacity)]
        self._generation += 1
