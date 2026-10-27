---
title: "Decision 0003 Build a phase-placement scheduler and benchmark, not a fabric"
type: decision
status: active
updated: 2026-10-27
sources: ["S02", "S06", "S18", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30", "S31", "S32", "S33", "S34", "S35", "S36"]
---

# Decision 0003 Build a phase-placement scheduler and benchmark, not a fabric

**Decision:** MacReduce's deliverable for 2026-12-04 is a small per-request scheduler for LLM serving on Apple Silicon plus the benchmark that tests it. The scheduler makes three placement decisions per request from online measurements, and the benchmark reports where it beats, ties, or loses to the strongest static configuration. The Wi-Fi local-SGD scheduler of [decision 0001](0001-scope.md) is demoted to a measured prototype and is no longer the research claim. This supersedes the scope in 0001; 0001 remains as history.

**One sentence:** build a scheduler that decides, per request, which engine drafts, how much to speculate, and whether to reuse or recompute prefix KV, and measure whether those decisions beat the best fixed configuration on one M4.

**Course project, amendment 2026-10-27 (evening):** 10-805 open-ended miniproject, lead Gautam Diwan (gdiwan). The project name is MacReduce. The submitted question is heterogeneous federated learning. A client is a GPU, a GPU and Neural Engine combined, or those two devices as separate clients, and every Mac in a run uses the same choice. The payload is a LoRA adapter delta. FedAvg is the communication reduction. A per-step all-reduce is the byte-heavy control. The baseline is one-Mac GPU FedAvg. The proposal is [docs/proposal.md](../../docs/proposal.md). The aggregation checks live in [federated.py](../../src/macreduce/federated.py).

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
- An ANE stack whose only output is a number ANEForge, Orion, or maderix already published. Calling those stacks is in scope. The question is whether an explicit call beats the GPU on this OS, including when several Macs train, and whether the Neural Engine should infer while the GPU trains.
- A new collective. The multi-Mac baseline is MLX's TCP ring. Heterogeneous all-reduce, a within-Mac sum followed by that ring, and a tree are the reductions under test. JACCL is out of reach without Thunderbolt 5.
- Wi-Fi as a contribution by itself. The link is measured when a second Mac exists.
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
- **Not planned:** anything requiring M5 or 128 GB machines. Private ANE APIs are allowed when the work addresses a gap the existing stacks leave open.

## The six directions

| Direction | Status | Reason in terms of verified sources |
|---|---|---|
| 1 Disaggregated prefill/decode, KV transfer versus recompute | **Narrowed into D3** | Done cross-machine by [S30](../sources/S30.md) and [S31](../sources/S31.md) with disagreeing crossovers; we keep the decision rule, not the system, and need a second Mac for the remote arm |
| 2 ANE draft plus GPU verifier | **Gated option inside D1** | ANE prefill/decode handoffs exist ([S23](../sources/S23.md), [S25](../sources/S25.md), [S26](../sources/S26.md), [S27](../sources/S27.md)); [S26](../sources/S26.md) says a separate tiny drafter is needed and did not test it; ANE placement on macOS 26.x is unverified ([S25](../sources/S25.md), [S36](../sources/S36.md)) |
| 3 MapReduce scheduler picking machine and engine | **Adopted, narrowed to inference** | No source read makes per-request engine plus depth plus prefix-source decisions for autoregressive LLM serving; this is a coverage statement, not a novelty claim, and the required search is in [gaps](../research/gaps.md) |
| 4 Continuous inference-to-training loop | **LoRA baseline adopted; the closed loop stays later** | Stock LoRA exists ([S19](../sources/S19.md), [S28](../sources/S28.md)). The training baseline is QLoRA of the pinned Qwen3-0.6B on WikiText-2, scored by perplexity. Wiring that adapter back into serving is still later |
| 5 Cost-aware mixture of specialists across Macs | **Demoted** | Needs several Macs and models we do not have; per-request model routing resembles [S29](../sources/S29.md) |
| 6 Self-learning hardware scheduler as foundation | **Adopted in the narrow form of online cost estimation** | "Self-learning" is limited to measured, updated cost tables; no reinforcement learning claim |

## Alternatives considered

- Continue decision 0001 and make the Wi-Fi scheduler the claim: rejected because [S02](../sources/S02.md), [S28](../sources/S28.md), and [S35](../sources/S35.md) cover connecting Macs for training, and [S04](../sources/S04.md)–[S08](../sources/S08.md) cover the synchronization schedule.
- Build an ANE-first system as the only deliverable: still not the whole project. Private-API training is in scope when it measures a gap those prototypes leave open. The serving scheduler remains.
- Build a general heterogeneous fabric: rejected; too large, and EXO plus MLX distributed already occupy the connective layer.

## Consequences

- [novelty.md](../research/novelty.md) is rewritten around this target; the old candidates A–D are kept as demoted history.
- [state.md](../project/state.md) records that no scheduler code exists yet; the trainer is a measured prototype whose coordinator and chunked-transfer code may be reused for telemetry and KV-file transfer.
- [roadmap.md](../project/roadmap.md) is replaced with a dated plan to 2026-12-04.
- Two unverified items stay open in [gaps](../research/gaps.md): whether any ANE path executes on this M4 under macOS 26.5, and whether a literature search on adaptive speculation depth and KV reuse policies finds the same decision procedure.

## Amendment 2026-10-26

The lines that kept private ANE APIs out of scope are withdrawn. Recreating an ANE training stack is in scope when it addresses a gap ANEForge, Orion, or maderix leave open, rather than restating a result they already report. The serving scheduler is unchanged.

The training baseline is QLoRA, not the 29,888-parameter local-SGD model. The dataset is Salesforce WikiText-2 raw v1 at revision `b08601e04326c79dfdd32d625aee71d232d685c3`. The base model is `mlx-community/Qwen3-0.6B-4bit` at the serving pin. The reported metrics are training loss, validation loss, and test perplexity before and after the adapter.

**Revisit when:** course feedback on 2026-10-28 rejects the framing in [the proposal](../../docs/proposal.md); the ANE gate passes and changes D1's cost ranking; a second Mac becomes available; or the static-sweep pilot shows no regime where static policies differ, which would remove the hypothesis.

## Amendment 2026-10-27

Distributed QLoRA across Macs is in scope for the 10-805 proposal. The lead is Gautam Diwan (gdiwan). The serving scheduler stays a measured study. It is not the dataset, metric, or baseline on that proposal.

The single-Mac baseline is GPU QLoRA. The multi-Mac baseline is MLX ring all-reduce over GPU ranks. A heterogeneous all-reduce puts a GPU rank and an explicit Neural Engine rank in one reduction, on one Mac and again on two Macs. A hierarchical all-reduce sums those ranks inside each Mac, then rings the sums. A tree uses the same four ranks. A flat ring of mixed ranks is rejected by `macreduce train-plan`, because a ring assumes equal participants. BlueConnect, Blink, and BytePS are the sources for that split. They were identified by search on 2026-10-27 and have not been ingested as source cards. A second Mac is required before a multi-machine number is reported. One Mac today is Mac16,12. There is no Thunderbolt 5, so JACCL is not the backend.

Explicit private interfaces are in scope: ANEForge (`import aneforge`), Orion, and maderix/ANE. None is pinned. Orion and maderix have no pinned import, so the probe marks them unpinned rather than missing. Core ML remains an implicit request, and a compute-unit flag is still not residency. `macreduce train-plan` writes the comparison list and the import probe. It does not train, and it does not substitute the GPU when a stack is absent.

The curve is three epochs on the first 4,096 training lines of WikiText-2 raw v1, batch size 8, gradient accumulation 2. Validation and test stay the full published splits. Validation perplexity is recorded at the end of each epoch. Full-split test perplexity is kept for the single-Mac GPU run and for each all-reduce that finishes. The model remains the pinned Qwen3-0.6B-4bit checkpoint.

## Amendment 2026-10-27 (evening)

The course question is heterogeneous federated learning. The name stays MacReduce. This supersedes the all-reduce-only course framing earlier on this date. Serving placement and speculative decoding stay in the repository as future scope. They are not the scored question.

Clients share one WikiText-2 mixture, in disjoint shards. Heterogeneity is the device. The intra-Mac modes are GPU, GPU and Neural Engine as one client, and GPU and Neural Engine as two clients. Every Mac in a run uses the same mode. Two Macs in split mode are therefore two GPUs and two Neural Engines.

The network payload is the LoRA adapter delta. Base weights stay on each client. FedAvg reduces rounds by taking local steps before the average. The per-step all-reduce is the control that moves more bytes. Int8 quantization keeps an error residual. Unequal local step counts use the FedNova normalization already noted on [S06](../sources/S06.md). Star, ring, and tree averages are required to match on uncompressed deltas. `src/macreduce/federated.py` implements those checks. It does not train, and it does not fill a missing Neural Engine stack with the GPU.

FetchSGD and hierarchical quantized averaging were identified by search on 2026-10-27, abstracts only, and are not source cards. A second Mac is still required before a multi-Mac number is reported. The curve has not been run.

## Amendment 2026-10-27 (later)

The round loop is implemented in `src/macreduce/federated.py`. The MLX QLoRA client and the Neural Engine refusal live in `src/macreduce/lora_client.py`. `macreduce federate --execute` is the training entry. It has not been run. Tests cover the CPU reference, the relay, and the refusal. They do not cover a WikiText perplexity.
