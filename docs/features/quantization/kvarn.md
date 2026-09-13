# KVarN KV cache on Intel XPU

Use `--kv-cache-dtype kvarn_k4v4_g128_compact` for compact K4V4/G128 storage,
or `--kv-cache-dtype kvarn_k4v2_g128_compact` for compact K4V2/G128 storage.
Native Xe2 DPAS cache layout, ID18 decoder, adaptive splits and the Sinkhorn
writer are selected by the cache dtype. Request-stable model operations retain
their existing model eligibility checks.
Historical `KVARN_*` experiment selectors are retired and rejected at startup;
do not use old factory-ablation instructions to configure a server.

At head dimension 256, a 128-token record for one KV head occupies 35,072
bytes in compact K4V4 and 26,880 bytes in compact K4V2. K4V2 keeps four-bit
keys and packs four two-bit values per byte. This reduces paged attention
storage by 23.36%, allowing about 30.48% more attention tokens within the same
allocation. Weights, recurrent states, activations and resident FP16 windows
are separate allocations. Concurrent requests share the available cache.

Compact K4V2 keeps a minimum of eight full recent-history blocks (1,024
tokens) in FP16 during decode, flushing older blocks when the window exceeds
sixteen blocks. Compact K4V4 retains its four-block high-water mark and
zero-block low-water mark. Both use the existing sixteen-block prefill
reservation, so K4V2 does not allocate a larger FP16 pool. The additional
FP16 reads and changed flush frequency are part of its performance tradeoff.
K4V2 uses sixteen native splits for three decode rows at context extents from
4,096 through 8,192 tokens, including the causal virtual rows used by MTP2.
Other batch/context choices and the thirty-two-split scratch capacity retain
the established B70 policy.

Both formats use lossy quantization. Numerical agreement and task accuracy
depend on the model and workload; neither format is mathematically lossless.
Use the paired release's quality and performance results when choosing one.
The older padded `kvarn_k4v2_g128` format does not provide the compact record's
storage savings. K4V2 requires a paired native library whose width-sensitive
operations expose the `value_bits` argument; old libraries cannot read it.

## Qwen images and bundled MTP

The serving profile uses the Qwen3.5-family W4A16 checkpoint
`RedHatAI/Qwen3.8-27B-INT4`, revision
`bf08f3dbd9a324e53956920aad378a1f1b6dd24a`, with BF16 activations, TP1/PP1,
V1/eager, no prefix cache, up to four scheduler slots, a 2,048-token prefill
budget, and up to two 448x448 images with video disabled. Consult the paired
packaging release for the exact tested cache format, context and runtime.

Add `--speculative-config '{"method":"mtp","num_speculative_tokens":2}'`
for the recommended two-token bundled MTP configuration. One draft token is
also correctness-qualified; MTP remains opt-in. Eligible verification uses
the native packed-cache reader with a separate causal bound per query, without
materializing the entire history. Rejected proposals cannot commit permanent
cache pages. No tuning overrides are required.

The MTP startup guard does not impose a separate context-length ceiling.
Use `--max-model-len` to select the combined input/output context; the model's
position limit and vLLM's profiled KV-cache capacity determine whether it fits.
Passing those startup checks does not replace long-context correctness testing.

Unsupported speculative methods, draft counts and serving combinations fail
the XPU KVarN guard. Do not enable graphs/V2/prefix caching to work around it.
Use `--kv-cache-dtype bfloat16` for an explicit unquantized cache control at a
context length that fits, or omit the speculative config to disable MTP.
`auto` follows checkpoint metadata; this RedHat checkpoint selects FP8.
The bounded synthetic image gate is not a general vision
benchmark or a qualification of other models, video or broader concurrency.

Release results and remaining work:
[KVarN B70 megaissue](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/issues/5).
