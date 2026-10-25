---
title: "Implementation order and completion gates"
type: plan
status: active
updated: 2026-10-24
sources: ["S02", "S10", "S18"]
---

# Implementation order and completion gates

Complete a small coherent experiment before expanding every axis. These are engineering estimates, not commitments or measured durations; actual work depends on access to another Mac and a normal terminal with Metal access.

| Order | Work package | Approximate effort | Exit evidence |
|---|---|---|---|
| 1 | [MLX validation](../guides/mlx-validation.md) | 0.5–1 day | Forward/update parity, GPU trace, stable multi-step loss |
| 2 | [Two-Mac Wi-Fi](../guides/wifi.md) | 1–2 days | Three repeated physical runs, topology, effective payload rate, restart case |
| 3 | [Real model and corpus](../guides/model-scaling.md) | 2–4 days | 1M–10M model, disjoint validation, reproducible preprocessing and memory budget |
| 4 | [Profiling validity](../guides/profiling.md) | 1–2 days | Warmup/steady-state split, instrumentation overhead, attribution limits |
| 5 | [Heterogeneity policy](../guides/heterogeneity.md) | 3–5 days | Fixed objective, baseline comparisons, time-to-loss and domain loss |
| 6 | [ANE feasibility spike](../guides/ane.md) | 2–3 days before a stop decision | Small gradients and completed-step benchmark, including host work |
| 7 | [Transport and compression](../guides/transport.md) | 2–4 days if transfer dominates | Bounded-memory transfer and fault injection; quality checks for lossy modes |
| 8 | [Large-model inference](../guides/inference.md) | Separate stretch project | Model fits aggregate capacity, measured TTFT/decode, validated partition |

## First three concrete tasks

1. **Partial, measured 2026-10-24.** The opt-in MLX test passed on this M4 for the default model and a 2-layer config, and a short `mlx-gpu` loopback demo finished. Tolerances were not changed. A GPU trace and the longer gradient audit are still open because `xctrace` was unavailable and that audit is not written yet. See [MLX validation](../guides/mlx-validation.md).
2. **Not started.** Connect a second physical Mac using explicit IP addresses. Keep the model, optimizer, K, and data policy fixed. Capture host identity, transport, and collector availability.
3. **Minimal slice done.** `ModelConfig` is serialized with checkpoints, and model transfer uses a bounded chunked artifact protocol instead of the old 2 MB response ceiling. A real corpus and a 1M-parameter model are still future work. See [model scaling](../guides/model-scaling.md) and [transport](../guides/transport.md).

Choose **one** primary research direction from [the novelty assessment](../research/novelty.md). ANE and M5 work can remain a bounded optional experiment if the Wi-Fi scheduling study is the main contribution. Do not require unavailable M5 hardware for the core deliverable.

## Definition of finished for the course-scale project

A physical two-Mac run is reproducible; comparisons use fixed data and quality targets; failure recovery is demonstrated; implementation and literature limitations are explicit; the report includes unsuccessful cases as well as improvements. Large-model pretraining from scratch is outside this deliverable. Check course expectations before describing this trainer as satisfying a general-purpose MapReduce assignment.
