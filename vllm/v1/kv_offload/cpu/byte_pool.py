# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, variable-size host storage for independent KV cache pools."""

from collections import OrderedDict
from collections.abc import Collection
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
    # -1 while storing; otherwise the number of outstanding loads.
    references: int = -1


@dataclass
class _RequestAccess:
    owner: object
    generation: int
    keys: dict[OffloadKey, None] = field(default_factory=dict)
    finished: bool = False


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
        self.entries: OrderedDict[OffloadKey, _Entry] = OrderedDict()
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
        state.keys.update(dict.fromkeys(keys))

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        entry = self.entries.get(key)
        if entry is None:
            return LookupResult.MISS
        return LookupResult.HIT_PENDING if entry.references < 0 else LookupResult.HIT

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        # Keep earlier prefix pages newer: losing the first page makes all later
        # attention pages unusable for a prefix hit.
        for key in reversed(list(keys)):
            if key in self.entries:
                self.entries.move_to_end(key)

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> HostPages:
        entries = [self.entries[key] for key in dict.fromkeys(keys)]
        assert all(entry.references >= 0 for entry in entries)
        for entry in entries:
            entry.references += 1
        self._record_accesses(keys, req_context)
        self.touch(keys, req_context)
        return HostPages([self.entries[key].page for key in keys])

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in dict.fromkeys(keys):
            entry = self.entries[key]
            assert entry.references > 0
            entry.references -= 1

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
        candidates = iter(
            (key, entry)
            for key, entry in self.entries.items()
            if entry.references == 0 and key not in protected
        )
        evicted: list[OffloadKey] = []
        allocated: list[HostPage] = []
        for key, (size, has_sink) in zip(missing, sizes):
            offset = self._allocate(free_ranges, size)
            while offset is None:
                victim = next(candidates, None)
                if victim is None:
                    return None
                victim_key, victim_entry = victim
                self._release(free_ranges, victim_entry.page)
                evicted.append(victim_key)
                offset = self._allocate(free_ranges, size)
            allocated.append(
                HostPage(offset, size, get_offload_group_idx(key), has_sink)
            )
        # Failed multi-page offers leave both the cache and free list intact.
        for key in evicted:
            del self.entries[key]
        self.free_ranges = free_ranges
        for key, page in zip(missing, allocated):
            self.entries[key] = _Entry(page)
        self._record_accesses(keys, req_context)
        self.touch(keys, req_context)
        return PrepareStoreOutput(missing, HostPages(allocated), evicted)

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        for key in dict.fromkeys(keys):
            entry = self.entries[key]
            assert entry.references == -1
            if not success:
                self._release(self.free_ranges, self.entries.pop(key).page)
                continue
            entry.references = 0

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
        keys = list(state.keys)
        positions = {
            key: position
            for key in keys
            if (position := req_context.get_offload_key_position(key)) is not None
        }
        if len(positions) == len(keys):
            keys.sort(key=positions.__getitem__)
        self.touch(keys, req_context)

    def reset_cache(self) -> None:
        # The connector flushes pending transfers before reusing these offsets
        # and suppresses their stale completion callbacks after reset.
        self.entries.clear()
        self.free_ranges = [(0, self.capacity)]
        self._generation += 1
