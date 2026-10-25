---
title: "Run one larger model across Macs as a separate path"
type: implementation
status: proposed
updated: 2026-10-25
sources: ["S02", "S18", "S19"]
---

# Run one larger model across Macs as a separate path

**Scope note 2026-10-25:** [decision 0003](../decisions/0003-build-target.md) rules out building sharded inference; MLX tensor parallelism ([S34](../sources/S34.md)) and EXO ([S18](../sources/S18.md)) cover it. The project's inference work is a per-request scheduler around pinned `mlx-lm` on one machine, with an optional second Mac for prefix-KV transfer. This page stays as the integration guide if a large model is ever needed.

The trainer currently replicates a small model on every worker. Running one model too large for an individual Mac requires sharding and a separate execution graph; it is not obtained by adding workers to local SGD.

## Implement an integration before a new inference engine

1. Inspect a pinned revision of [EXO](../sources/S18.md) and [MLX distributed support](../sources/S02.md). Verify the specific model architecture and network backend in that version. “Mac-optimized Qwen” often means converted or quantized MLX weights; it does not by itself imply ANE execution.
2. Pick the smallest model that exceeds one selected worker's safe memory budget but fits the cluster with headroom. Check model and tokenizer licenses, quantization format, weight size, KV-cache growth, and per-node activation/temporary memory. Do not begin by downloading 100B weights without a capacity plan.
3. First shard a smaller model that also fits on one Mac so correctness and timing can be compared against an unsharded reference. Freeze prompts, tokenizer, sampling settings, precision, and context lengths. Compare logits within a justified tolerance; token equality alone can miss numerical differences, while minor differences can also alter sampled tokens.
4. Prefer testing contiguous layer partitions first on Wi-Fi because this can reduce communication frequency relative to fine-grained tensor collectives. This is a design hypothesis, not a universal speed claim. Record partition sizes and measured compute/transfer balance; mixed memory capacities imply unequal layer counts.
5. Keep KV cache local to its owning layers where the engine allows it. Measure prefill, time to first token, per-token decode latency, throughput at multiple concurrent requests, and peak memory separately. A single autoregressive sequence has serial dependencies even if different nodes hold different layers.
6. Revisit topology, retries, and failure recovery. A failed shard can invalidate request state. Training checkpoint recovery does not restore an in-flight inference request automatically.

## Success criterion

The distributed configuration demonstrably fits and serves the selected model, with documented latency and output checks. A capacity increase may be useful even if it is slower than a smaller model on one Mac. Do not turn that into an unsupported speedup claim.

Connect inference metrics to MacReduce's run/evidence schema, but keep a separate adapter and command namespace. Keep this behind the core trainer milestones in [the roadmap](../project/roadmap.md).
