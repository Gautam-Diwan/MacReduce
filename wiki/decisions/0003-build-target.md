---
title: "Decision 0003 Build a phase-placement scheduler and benchmark, not a fabric"
type: decision
status: active
updated: 2026-10-25
sources: ["S02", "S18", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30", "S31", "S32", "S33", "S34", "S35", "S36"]
---

# Decision 0003 Build a phase-placement scheduler and benchmark, not a fabric

**Decision:** MacReduce's deliverable for 2026-12-04 is a small per-request scheduler for LLM serving on Apple Silicon plus the benchmark that tests it. The scheduler makes three placement decisions per request from online measurements, and the benchmark reports where it beats, ties, or loses to the strongest static configuration. The Wi-Fi local-SGD scheduler of [decision 0001](0001-scope.md) is demoted to a measured prototype and is no longer the research claim. This supersedes the scope in 0001; 0001 remains as history.

**One sentence:** build a scheduler that decides, per request, which engine drafts, how much to speculate, and whether to reuse or recompute prefix KV, and measure whether those decisions beat the best fixed configuration on one M4.

## Context and evidence

The 2026-10-25 source refresh ([log](../log.md)) read primary READMEs and docs for the systems the user named. Inspected scope is on each source card; nothing below was reproduced locally.

- Connecting Macs is first-party. MLX ships ring (TCP), MPI, and JACCL (RDMA over Thunderbolt, macOS 26.2+, Thunderbolt 5) backends, data-parallel `average_gradients`, and tensor-parallel sharded layers ([S02](../sources/S02.md), [S34](../sources/S34.md), [S35](../sources/S35.md)). EXO adds discovery, topology-aware auto-parallel, and pipeline or tensor sharding ([S18](../sources/S18.md); README re-read 2026-10-25, byte-identical to the snapshot; repository not archived, last push 2026-10-13 per API metadata). A community guide already runs stock `mlx_lm lora` data-parallel over Thunderbolt ([S28](../sources/S28.md)).
- ANE training prototypes exist: ANEForge ([S10](../sources/S10.md)), maderix/ANE with GPU-prefill to ANE-decode over IOSurface ([S23](../sources/S23.md)), Orion ([S12](../sources/S12.md)). The maderix README states utilization of about 5–9% of peak and CPU-side weight gradients.
- ANE-prefill plus GPU-decode on one Mac is implemented and benchmarked by at least three projects, with negative results included ([S25](../sources/S25.md), [S26](../sources/S26.md), [S27](../sources/S27.md)). S25 reports that on macOS 26.3 Core ML `compute_units=ALL` ran on the GPU with about 0 W ANE power; S36 confirms the flag is a constraint, not a placement guarantee.
- Concurrent GPU plus ANE serving with per-request routing exists for a classifier workload, with a parity gate and explained routing; the 4.57x and 79.5 ms P99 figures are project-reported on one M4 Max under bursty load, and the README attributes the gain to concurrency, not ANE speed ([S29](../sources/S29.md)).
- Cross-machine prefill/decode disaggregation with KV transfer to a Mac exists twice, with crossover tables that disagree by workload ([S30](../sources/S30.md), [S31](../sources/S31.md)). Memory-tier placement within one Mac exists ([S24](../sources/S24.md)).
- Stock MLX speculative decoding uses a same-device draft model and a fixed draft length, and has open issues about cache trimmability, greedy divergence, and token skipping across model families ([S32](../sources/S32.md), [S33](../sources/S33.md)). ANE Fieldfare adapts draft length from measured acceptance inside its own runtime ([S27](../sources/S27.md)).

**Measured locally (unchanged):** one M4 (Mac16,12), macOS 26.5, MLX 0.32.4; the existing trainer runs a 29,888-parameter model on CPU and MLX GPU over loopback ([state](../project/state.md)). No second Mac, no Thunderbolt 5, no verified ANE execution.

## What not to build

- MLX collectives, hostfiles, or a transport that competes with ring/JACCL.
- Another EXO: discovery, auto-parallel, dashboard, sharded inference.
- Another ANE training stack or ANE kernel compiler.
- Wi-Fi as the research question. The link is a measured parameter of decision D3, nothing more.
- "First Mac cluster", "first ANE X", or a general heterogeneous compute fabric.
- A new inference engine. Use `mlx-lm` at a pinned version as the engine.

## The smallest system

A Python service around pinned `mlx-lm` that serves an OpenAI-style request trace and, per request, chooses:

| Decision | Choices | Signal | Static baseline it must beat |
|---|---|---|---|
| **D1 Drafter engine** | no drafter; draft model on GPU; draft model on CPU stream (concurrent with GPU verify); draft model on ANE via Core ML **only if a placement gate passes** | measured draft step time per engine, target verify time, GPU contention, placement check | best single engine chosen by offline sweep per workload |
| **D2 Speculation depth** | k in {0, 1, 2, 4, 8} per request, updated per step from acceptance | running acceptance rate, draft and verify step costs, output length so far | best fixed k from offline sweep (mlx-lm default is fixed k) |
| **D3 Prefix source** | recompute prefill; load saved prefix KV from local disk; with a second Mac, pull prefix KV from the peer or send the prompt for remote prefill | prefix length, KV bytes, measured disk or link rate, local prefill tok/s, queue depth | always recompute, and always reuse |

The map/reduce shape is the scheduler's structure, not a Hadoop claim: **map** places each request's phases on an engine or machine; **reduce** folds per-request telemetry into the cost tables that drive the next decision. The course assignment is not claimed as satisfied by this.

**Metrics:** TTFT (p50, p95), decode tok/s after first token, requests per minute on the trace, and quality as greedy exact-match against non-speculative output (mandatory) plus mean logprob on sampled runs. Bytes moved and probe overhead are reported. Energy is reported only as a labeled `powermetrics` package estimate or omitted; zero is never written for an unavailable counter ([decision 0002](0002-measurements.md)).

**Benchmark discipline:** A-B-B-A alternation in fresh processes with a stated noise floor, as in [S27](../sources/S27.md); three seeds; pinned `mlx`, `mlx-lm`, and model revisions; greedy equivalence checked before any speed number is kept ([S33](../sources/S33.md) shows why).

## Hypothesis and falsification

**Hypothesis:** on one M4, a scheduler that sets drafter engine, draft depth, and prefix source per request from online measurements achieves lower p50 TTFT and higher decode tok/s on a mixed trace (short chat, long-context, code) than the best single static configuration chosen with hindsight for that trace, without changing greedy outputs.

**Falsified if** the hindsight-best static configuration is within the measured noise floor of the adaptive policy on both TTFT and decode tok/s, or the adaptive policy's probe and bookkeeping overhead exceeds its gain, or greedy outputs differ. A negative result with attributed costs is a valid deliverable.

## What runs where

- **One M4 (available now):** D1 with GPU and CPU drafters; D2 fully; D3 as recompute versus local-disk reuse using `mlx_lm.cache_prompt` files; the entire benchmark harness; the ANE gate (which may fail, and that is a result).
- **Requires a second Mac:** D3's cross-machine arm (KV pull versus prompt send versus local), machine choice under queue depth. Any IP link is acceptable; the link rate is measured and reported. Thunderbolt 5 RDMA is **not** assumed and not required.
- **Not planned:** anything requiring M5, 128 GB machines, or private ANE APIs.

## The six directions

| Direction | Status | Reason in terms of verified sources |
|---|---|---|
| 1 Disaggregated prefill/decode, KV transfer versus recompute | **Narrowed into D3** | Done cross-machine by [S30](../sources/S30.md) and [S31](../sources/S31.md) with disagreeing crossovers; we keep the decision rule, not the system, and need a second Mac for the remote arm |
| 2 ANE draft plus GPU verifier | **Gated option inside D1** | ANE prefill/decode handoffs exist ([S23](../sources/S23.md), [S25](../sources/S25.md), [S26](../sources/S26.md), [S27](../sources/S27.md)); [S26](../sources/S26.md) says a separate tiny drafter is needed and did not test it; ANE placement on macOS 26.x is unverified ([S25](../sources/S25.md), [S36](../sources/S36.md)) |
| 3 MapReduce scheduler picking machine and engine | **Adopted, narrowed to inference** | No source read makes per-request engine plus depth plus prefix-source decisions for autoregressive LLM serving; this is a coverage statement, not a novelty claim, and the required search is in [gaps](../research/gaps.md) |
| 4 Continuous inference-to-training loop | **Demoted to future reduce stage** | LoRA tooling and distributed LoRA exist ([S19](../sources/S19.md), [S28](../sources/S28.md)); the loop is a pipeline without a measurable scheduling question in the time left |
| 5 Cost-aware mixture of specialists across Macs | **Demoted** | Needs several Macs and models we do not have; per-request model routing resembles [S29](../sources/S29.md) |
| 6 Self-learning hardware scheduler as foundation | **Adopted in the narrow form of online cost estimation** | "Self-learning" is limited to measured, updated cost tables; no reinforcement learning claim |

## Alternatives considered

- Continue decision 0001 and make the Wi-Fi scheduler the claim: rejected because [S02](../sources/S02.md), [S28](../sources/S28.md), and [S35](../sources/S35.md) cover connecting Macs for training, and [S04](../sources/S04.md)–[S08](../sources/S08.md) cover the synchronization schedule.
- Build an ANE-first system: rejected as the primary target because every ANE path read is private-API or unverified under Core ML on macOS 26.x, and three prototypes already exist.
- Build a general heterogeneous fabric: rejected; too large, and EXO plus MLX distributed already occupy the connective layer.

## Consequences

- [novelty.md](../research/novelty.md) is rewritten around this target; the old candidates A–D are kept as demoted history.
- [state.md](../project/state.md) records that no scheduler code exists yet; the trainer is a measured prototype whose coordinator and chunked-transfer code may be reused for telemetry and KV-file transfer.
- [roadmap.md](../project/roadmap.md) is replaced with a dated plan to 2026-12-04.
- Two unverified items stay open in [gaps](../research/gaps.md): whether any ANE path executes on this M4 under macOS 26.5, and whether a literature search on adaptive speculation depth and KV reuse policies finds the same decision procedure.

**Revisit when:** the proposal review on 2026-11-06 rejects the framing; the ANE gate passes and changes D1's cost ranking; a second Mac becomes available; or the static-sweep pilot shows no regime where static policies differ, which would remove the hypothesis.
