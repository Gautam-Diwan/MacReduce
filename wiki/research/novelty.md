---
title: "Research contribution candidates and falsification plans"
type: research
status: proposed
updated: 2026-10-25
sources: ["S02", "S03", "S05", "S06", "S07", "S08", "S09", "S10", "S12", "S23", "S25", "S26", "S27", "S28", "S29", "S30", "S31", "S32", "S33", "S34", "S35", "S36"]
---

# Research contribution candidates and falsification plans

**Recommendation (2026-10-25, [decision 0003](../decisions/0003-build-target.md)):** build a per-request phase-placement scheduler for LLM serving on Apple Silicon and the benchmark that tests it against the strongest static configuration. The earlier recommendation, a Wi-Fi local-SGD scheduling study, is demoted below. No candidate on this page is a verified novelty claim; each lists the sources that block the obvious version of the claim and the search still owed.

## Why the earlier recommendation was demoted

The 2026-10-23 recommendation was "build a credible physical Wi-Fi baseline first, then test objective-preserving scheduling under changing Wi-Fi and thermal conditions." Sources read on 2026-10-25 remove its footing:

- Connecting Macs for training is first-party: `mlx.launch`, ring/MPI/JACCL backends, and `nn.average_gradients` ([S02](../sources/S02.md), [S35](../sources/S35.md)); community guides already run data-parallel LoRA across Macs with stock tooling ([S28](../sources/S28.md)); AirTrain already does local SGD over Wi-Fi ([S03](../sources/S03.md)).
- The synchronization schedule itself is covered by DiLoCo and its variants ([S04](../sources/S04.md), [S05](../sources/S05.md)) and by heterogeneity-aware local SGD ([S06](../sources/S06.md), [S07](../sources/S07.md), [S08](../sources/S08.md)).
- ANE training prototypes exist and report their own limits ([S10](../sources/S10.md), [S12](../sources/S12.md), [S23](../sources/S23.md)).

What remains of that work is a measured prototype: a CPU and MLX-GPU local-SGD trainer with chunked transfer and tests ([state](../project/state.md)). It is infrastructure, not the claim.

## Primary candidate: per-request placement decisions a static runtime gets wrong

**Question:** on a Mac, does choosing drafter engine, speculation depth, and prefix-KV source per request from online measurements beat the best fixed configuration, and in which regimes?

**Hypothesis:** on one M4, the adaptive policy yields lower p50 TTFT and higher decode tok/s on a mixed trace than the hindsight-best static configuration, without changing greedy output. **Falsified if** the static configuration is within the A-B-B-A noise floor on both metrics, or policy overhead exceeds the gain, or greedy outputs differ.

**The three decisions** are specified in [decision 0003](../decisions/0003-build-target.md): D1 drafter engine (none, GPU, CPU stream, gated ANE), D2 speculation depth k per step, D3 prefix source (recompute, local-disk KV, peer KV or remote prefill when a second Mac exists).

**What blocks the naive versions of this claim (source-reported unless marked):**

| Sub-claim | Blocked by | What is left |
|---|---|---|
| Speculative decoding on MLX | stock `mlx-lm` `--draft-model` ([S32](../sources/S32.md)) | choosing engine and depth, and verifying equivalence given [S33](../sources/S33.md) |
| Adaptive draft length from acceptance | ANE Fieldfare's `SpeculationPolicy` ([S27](../sources/S27.md)), inside one Swift runtime on an M5 | a measured comparison against best-fixed-k on MLX; academic prior work on dynamic speculation is **not yet searched** |
| ANE drafter with GPU verifier | ANE prefill/decode handoffs ([S23](../sources/S23.md), [S25](../sources/S25.md), [S26](../sources/S26.md), [S27](../sources/S27.md)); [S26](../sources/S26.md) names the idea | a placement-verified measurement on this M4; may fail the gate, which is a result |
| Concurrent GPU plus ANE routing | laya-apple ([S29](../sources/S29.md)) for a classifier | autoregressive decode, online updating, second machine |
| KV transfer versus recompute | [S30](../sources/S30.md), [S31](../sources/S31.md) on DGX plus Mac Studio | a decision rule parameterized by prefix length, bytes, link rate, local prefill rate; literature on KV-cache reuse policies is **not yet searched** |
| Memory-tier placement | Hypura ([S24](../sources/S24.md)) | nothing; do not rebuild |
| Within-model compute-unit choice | Core ML `compute_units` ([S36](../sources/S36.md)) is Apple's decision, not ours | treat as a resource to verify, not a knob to claim |

**Baselines (strongest static, not strawmen):** no speculation; best fixed k with GPU drafter from an offline sweep per workload; best fixed k with CPU drafter; always-recompute and always-reuse for D3; a hindsight-best static triple for the whole trace. Report the oracle only as a retrospective bound.

**Primary outcomes:** p50 and p95 TTFT; decode tok/s after first token; requests per minute on the trace; greedy exact-match rate against non-speculative generation (must be 100% or the configuration is excluded); bytes moved; probe overhead. Energy only as a labeled `powermetrics` package estimate.

**Ablations:** each decision alone with the other two fixed; short-only, long-only, and code-only traces; cold versus warm model and Core ML compile state; one versus two concurrent requests. If gains appear only on one trace type, say so.

**Resources:** one M4 now (all of D1–D2, local D3, harness). A second Mac on any IP link unlocks the remote arm of D3. No Thunderbolt 5, M5, or private ANE API is required.

**Stages to 2026-12-04** are in the [roadmap](../project/roadmap.md).

## Demoted candidates (kept for history)

### A. Objective-preserving scheduling under changing Wi-Fi and thermal conditions

Demoted 2026-10-25. The data-objective rules it introduced (fixed mixture weights p_i, solver-appropriate normalization, [decision 0002](../decisions/0002-measurements.md)) stay in force for any training work. The Wi-Fi control study is not pursued because connecting Macs is covered ([S02](../sources/S02.md), [S28](../sources/S28.md), [S03](../sources/S03.md)) and the schedule-adaptation literature is dense ([S06](../sources/S06.md), [S07](../sources/S07.md), [S08](../sources/S08.md)).

### B. ANE-resident local training blocks versus synchronization boundaries

Demoted. Three ANE training prototypes exist ([S10](../sources/S10.md), [S12](../sources/S12.md), [S23](../sources/S23.md)); all use private APIs; maderix reports 5–9% utilization and CPU-side weight gradients. The boundary-cost accounting idea is sound but requires a working ANE training backend that this project will not build.

### C. Calibration that does not damage the workload it is measuring

Folded into the primary candidate as the probe-overhead and bounded-probe rules. [FusionML](../sources/S09.md) remains the nearest prior work for calibration effects.

### D. A reproducible Mac-cluster training and profiling benchmark

Replaced by the serving benchmark in the primary candidate. The discipline carries over: per-phase traces, hashes, honest missing counters, profiling-on versus off.

## What not to claim

Do not claim first Mac cluster, first ANE anything, first speculative decoding on MLX, first prefill/decode disaggregation to a Mac, or first GPU-plus-ANE routing. Do not call a scheduling gain novel until the dynamic-speculation and KV-reuse literature searches listed in [gaps](../research/gaps.md) are done and the nearest papers are read in full. A measured negative result with attributed costs is an acceptable outcome of this plan.
