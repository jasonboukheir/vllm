# KVarN KV cache

KVarN's scoped Intel XPU profile uses request-stable model operations in
addition to its quantized KV-cache kernels. The stable operations are enabled
by default. Two diagnostic selectors allow a runtime-factory launcher to
measure their costs independently without rebuilding vLLM:

| Child-process environment variable | `1` (default) | `0` |
| ---------------------------------- | ------------- | --- |
| `KVARN_REQUEST_STABLE_PROJECTION_ROWS` | Use canonical request-stable row counts for model and logits projections. | Use ordinary projection dispatch. |
| `KVARN_REQUEST_STABLE_RMSNORM` | Use the request-stable Gemma RMSNorm reduction. | Use ordinary RMSNorm dispatch, including the fused XPU implementation when available. |

Values must be exactly `0` or `1`. Invalid values fail during KVarN worker
startup. An absent value selects `1`, preserving the qualified behavior. Each
selection is cached for the lifetime of the engine process, so changing its
environment after startup has no effect.

The axes are independent. Request metadata remains attached while either axis
is active and is omitted only when both are disabled. This keeps the cache
layout ABI unchanged: these selectors affect model-operation scheduling, not
the KVarN cache writer, reader, or layout.

`KVARN_ONEDNN_DETERMINISTIC` remains a third, independent selector. Disabling
one or both request-stability axes, or oneDNN determinism, is an experimental
performance ablation and requires replay-correctness qualification before it
can replace the defaults.

These switches activate only for the frozen eager, single-XPU Qwen profile.
Other profiles remain fail-closed on their ordinary dispatch paths. The
startup log records each selected value, its source, and profile eligibility
under the `[KVARN_FACTORY]` prefix.
