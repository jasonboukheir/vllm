# KVarN KV cache on Intel XPU

Use `--kv-cache-dtype kvarn_k4v4_g128_compact` to select the qualified B70
K4V4/G128 profile. Native Xe2 DPAS cache layout, ID18 decoder, adaptive splits,
Sinkhorn writer and request-stable model operations are fixed release defaults.
Historical `KVARN_*` experiment selectors are retired and rejected at startup;
do not use old factory-ablation instructions to configure a server.

## Qwen images and bundled MTP

The xpu-v1.7 release qualifies the AEON Qwen3.5-family W4A16 checkpoint
`jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound`,
revision `6b0622f4354481d5d04577d48ba0db844efc1330`, with BF16 activations,
B1, TP1/PP1, V1/eager, no prefix cache, context up to8192, prefill budget2048,
and up to two448x448 images with video disabled.

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
Use `--kv-cache-dtype auto` for cache rollback, or omit the speculative config
to disable MTP. The bounded synthetic image gate is not a general vision
benchmark or a qualification of other models, video or broader concurrency.

Release results and remaining work:
[KVarN B70 megaissue](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/issues/5).
