"""Compare padded and compact KVarN Triton-reader record strides on XPU."""

from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Callable

import torch

from vllm.model_executor.layers.quantization.kvarn.config import KVarNConfig
from vllm.v1.attention.ops.triton_kvarn_decode import (
    _kvarn_build_packed_kv_kernel,
    _kvarn_fused_decode_stage1,
    _kvarn_fused_decode_stage2,
)
from vllm.v1.attention.ops.triton_kvarn_flush import _kvarn_flush_record_kernel

D = 256
G = 128
HK = 4


def percentile(samples: list[float], fraction: float) -> float:
    ordered = sorted(samples)
    return ordered[round((len(ordered) - 1) * fraction)]


def measure(
    launch: Callable[[], None], warmup: int, iterations: int, graph_capture: bool
) -> list[float]:
    """Return XPU device-event durations in microseconds."""
    for _ in range(warmup):
        launch()
    torch.xpu.synchronize()

    runner = launch
    graph = None
    if graph_capture:
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph):
            launch()
        runner = graph.replay
        for _ in range(warmup):
            runner()
        torch.xpu.synchronize()

    samples = []
    for _ in range(iterations):
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        runner()
        end.record()
        torch.xpu.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return samples


def make_fields(records: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(17)
    qk = torch.randint(0, 16, (records, D, G), dtype=torch.uint8, generator=generator)
    qv = torch.randint(0, 16, (records, G, D), dtype=torch.uint8, generator=generator)

    def pack(q: torch.Tensor) -> torch.Tensor:
        return (q[..., 0::2] | (q[..., 1::2] << 4)).contiguous().xpu()

    ones_d = torch.ones((records, D), dtype=torch.float16)
    zeros_d = torch.zeros_like(ones_d)
    ones_g = torch.ones((records, G), dtype=torch.float16)
    zeros_g = torch.zeros_like(ones_g)

    def as_bytes(value: torch.Tensor) -> torch.Tensor:
        return value.contiguous().view(torch.uint8).xpu()

    return (
        pack(qk),
        as_bytes(ones_d),
        as_bytes(zeros_d),
        as_bytes(ones_g),
        pack(qv),
        as_bytes(ones_d),
        as_bytes(ones_g),
        as_bytes(zeros_g),
    )


def make_cache(cache_dtype: str, blocks: int) -> tuple[torch.Tensor, KVarNConfig]:
    cfg = KVarNConfig.from_cache_dtype(cache_dtype, head_dim=D)
    cache = torch.zeros((blocks, HK, cfg.record_bytes), dtype=torch.uint8, device="xpu")
    fields = make_fields(blocks * HK)
    block_ids = torch.arange(blocks, dtype=torch.int64, device="xpu")
    copy_width = 256
    chunks = (cfg.record_bytes + copy_width - 1) // copy_width
    _kvarn_flush_record_kernel[(blocks * HK, chunks)](
        *fields,
        block_ids,
        cache,
        cache.stride(0),
        cache.stride(1),
        K_PACKED_BYTES=cfg.k_packed_bytes,
        V_PACKED_OFFSET=cfg.v_packed_offset,
        V_PACKED_BYTES=cfg.v_packed_bytes,
        K_SCOL_OFFSET=cfg.k_s_col_offset,
        K_ZP_OFFSET=cfg.k_zp_offset,
        K_SROW_OFFSET=cfg.k_s_row_offset,
        V_SCOL_OFFSET=cfg.v_s_col_offset,
        V_SROW_OFFSET=cfg.v_s_row_offset,
        V_ZP_OFFSET=cfg.v_zp_offset,
        RECORD_BYTES=cfg.tile_bytes,
        TILE_BYTES=cfg.record_bytes,
        NUM_HEADS=HK,
        NATIVE_DPAS_LAYOUT=True,
        BLOCK=copy_width,
        num_warps=4,
    )
    return cache, cfg


def benchmark_materialize(
    cache_dtype: str,
    context: int,
    warmup: int,
    iterations: int,
    graph_capture: bool,
) -> dict[str, float | int | str]:
    batch = 4
    blocks = (context + G - 1) // G
    cache, cfg = make_cache(cache_dtype, blocks)
    block_table = torch.arange(blocks, dtype=torch.int32, device="xpu").repeat(batch, 1)
    seq_lens = torch.full((batch,), context, dtype=torch.int32, device="xpu")
    cu = torch.arange(batch + 1, dtype=torch.int32, device="xpu") * context
    block_to_slot = torch.full((blocks,), -1, dtype=torch.int32, device="xpu")
    tail_k = torch.zeros((1, G, HK, D), dtype=torch.float16, device="xpu")
    tail_v = torch.zeros_like(tail_k)
    out_k = torch.empty((batch * context, HK, D), dtype=torch.float16, device="xpu")
    out_v = torch.empty_like(out_k)

    def launch() -> None:
        _kvarn_build_packed_kv_kernel[(batch * blocks, HK)](
            block_table,
            seq_lens,
            cu,
            block_to_slot,
            cache,
            tail_k,
            tail_v,
            out_k,
            out_v,
            block_table.stride(0),
            cache.stride(0),
            cache.stride(1),
            tail_k.stride(0),
            tail_k.stride(1),
            tail_k.stride(2),
            out_k.stride(0),
            out_k.stride(1),
            MAX_BLOCKS_PER_REQ=blocks,
            D=D,
            GROUP=G,
            K_BITS=4,
            V_BITS=4,
            NUM_BLOCKS_LOOKUP=blocks,
            K_PACKED_OFFSET=cfg.k_packed_offset,
            K_S_COL_OFFSET=cfg.k_s_col_offset,
            K_ZP_OFFSET=cfg.k_zp_offset,
            K_S_ROW_OFFSET=cfg.k_s_row_offset,
            V_PACKED_OFFSET=cfg.v_packed_offset,
            V_S_COL_OFFSET=cfg.v_s_col_offset,
            V_S_ROW_OFFSET=cfg.v_s_row_offset,
            V_ZP_OFFSET=cfg.v_zp_offset,
            DPAS_LAYOUT=True,
            num_warps=4,
            num_stages=2,
        )

    samples = measure(launch, warmup, iterations, graph_capture)
    return {
        "reader": "materialize",
        "execution": "xpu_graph" if graph_capture else "eager",
        "cache_dtype": cache_dtype,
        "context": context,
        "record_bytes": cfg.record_bytes,
        "median_us": statistics.median(samples),
        "p95_us": percentile(samples, 0.95),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def benchmark_fused(
    cache_dtype: str,
    context: int,
    warmup: int,
    iterations: int,
    graph_capture: bool,
) -> dict[str, float | int | str]:
    batch, hq, splits = 4, 24, 32
    blocks = (context + G - 1) // G
    cache, cfg = make_cache(cache_dtype, blocks)
    block_table = torch.arange(blocks, dtype=torch.int32, device="xpu").repeat(batch, 1)
    seq_lens = torch.full((batch,), context, dtype=torch.int32, device="xpu")
    block_to_slot = torch.full((blocks,), -1, dtype=torch.int32, device="xpu")
    tail_k = torch.zeros((1, G, HK, D), dtype=torch.float16, device="xpu")
    tail_v = torch.zeros_like(tail_k)
    query = torch.randn((batch, hq, D), dtype=torch.float16, device="xpu")
    mid_o = torch.empty((batch * hq, splits, D), dtype=torch.float32, device="xpu")
    mid_lse = torch.empty((batch * hq, splits), dtype=torch.float32, device="xpu")
    output = torch.empty((batch * hq, D), dtype=torch.float16, device="xpu")

    def launch() -> None:
        _kvarn_fused_decode_stage1[(batch, HK, splits)](
            query,
            seq_lens,
            block_table,
            seq_lens,
            block_to_slot,
            cache,
            tail_k,
            tail_v,
            mid_o,
            mid_lse,
            1 / 16,
            hq * D,
            D,
            block_table.stride(0),
            cache.stride(0),
            cache.stride(1),
            tail_k.stride(0),
            tail_k.stride(1),
            tail_k.stride(2),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_lse.stride(0),
            MAX_BLOCKS_PER_REQ=blocks,
            D=D,
            GROUP=G,
            Q_PER_KV=hq // HK,
            Q_PER_KV_PAD=8,
            NUM_KV_SPLITS=splits,
            HQ=hq,
            SLIDING_WINDOW=0,
            K_BITS=4,
            V_BITS=4,
            NUM_BLOCKS_LOOKUP=blocks,
            K_PACKED_OFFSET=cfg.k_packed_offset,
            K_S_COL_OFFSET=cfg.k_s_col_offset,
            K_ZP_OFFSET=cfg.k_zp_offset,
            K_S_ROW_OFFSET=cfg.k_s_row_offset,
            V_PACKED_OFFSET=cfg.v_packed_offset,
            V_S_COL_OFFSET=cfg.v_s_col_offset,
            V_S_ROW_OFFSET=cfg.v_s_row_offset,
            V_ZP_OFFSET=cfg.v_zp_offset,
            DPAS_LAYOUT=True,
            VQ_INDIRECT=False,
        )
        _kvarn_fused_decode_stage2[(batch * hq,)](
            mid_o,
            mid_lse,
            output,
            mid_o.stride(0),
            mid_o.stride(1),
            mid_lse.stride(0),
            output.stride(0),
            D=D,
            NUM_KV_SPLITS=splits,
            num_warps=2,
        )

    samples = measure(launch, warmup, iterations, graph_capture)
    return {
        "reader": "fused_split_k",
        "execution": "xpu_graph" if graph_capture else "eager",
        "cache_dtype": cache_dtype,
        "context": context,
        "record_bytes": cfg.record_bytes,
        "median_us": statistics.median(samples),
        "p95_us": percentile(samples, 0.95),
        "min_us": min(samples),
        "max_us": max(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="6000,6512")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--execution", choices=("eager", "graph", "both"), default="graph"
    )
    args = parser.parse_args()
    results = []
    for context in (int(value) for value in args.contexts.split(",")):
        for cache_dtype in ("kvarn_k4v4_g128", "kvarn_k4v4_g128_compact"):
            for execution in ("eager", "graph"):
                if args.execution not in (execution, "both"):
                    continue
                for benchmark in (benchmark_materialize, benchmark_fused):
                    results.append(
                        benchmark(
                            cache_dtype,
                            context,
                            args.warmup,
                            args.iterations,
                            graph_capture=execution == "graph",
                        )
                    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
