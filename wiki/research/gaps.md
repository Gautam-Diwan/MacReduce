---
title: "Unknowns contradictions and next literature searches"
type: research-ledger
status: active
updated: 2026-10-25
sources: ["S02", "S03", "S06", "S07", "S08", "S09", "S10", "S12", "S25", "S27", "S28", "S29", "S30", "S31", "S33", "S36"]
---

# Unknowns contradictions and next literature searches

This ledger distinguishes missing implementation, missing evidence, and incomplete literature coverage. It is not a backlog of already solved research problems. Entries are reprioritized for [decision 0003](../decisions/0003-build-target.md).

| Priority | Gap or contradiction | Why it matters | Resolution |
|---|---|---|---|
| P0 | **No scheduler or benchmark code exists** | The deliverable is unbuilt | [Roadmap](../project/roadmap.md) items 1–2 by 2026-11-05 |
| P0 | **Does the serving drafter execute on the ANE of this M4?** A convolution plan does: `CPU_AND_NE` assigns conv and ReLU to `MLNeuralEngineComputeDevice`, `CPU_ONLY` assigns them to the CPU ([probe](../../evidence/ane-probe/README.md)). [S25](../sources/S25.md) reports Core ML `ALL` ran on GPU on macOS 26.3; [S36](../sources/S36.md) says the flag is a constraint | D1's ANE option is still unverified for an LLM draft. Wall time did not separate the two constraints, and ANE watts were not read | Convert the draft, compare outputs to MLX, and read a power sample or a draft compute plan. Do not treat the convolution plan as draft residency |
| P0 | **Stock speculative decoding** matches greedy tokens for Qwen2.5 at the depths in the old sweep, and for Qwen3 at depths 1, 2, and 4. Qwen3 depth 8 mismatches the short prompt at token 26 ([Qwen3 check](../../evidence/serve-equivalence-qwen3/README.md)). [S33](../sources/S33.md) lists other divergences | A speed number at a diverging depth is invalid | Keep the token diff on every depth. Do not score Qwen3 depth 8 until the short-prompt mismatch is gone |
| P0 | **Literature on adaptive speculation depth and KV reuse policies not searched** | The primary candidate could duplicate a published rule | Searches listed in [related work](related-work.md); read nearest two papers in full; log results |
| P1 | **Second Mac, RAM, and link are unknown** | D3's remote arm and machine choice cannot be planned | Record exact hardware when known; measure the link; do not assume Thunderbolt 5 RDMA ([S28](../sources/S28.md) lists JACCL provider bugs) |
| P1 | **Disaggregation crossovers disagree:** [S30](../sources/S30.md) favors the split up to 8K ctx; [S31](../sources/S31.md) favors it only from about 32K on code | A single crossover number is workload- and hardware-specific | Treat crossover as the output of the D3 rule, parameterized by prefix length, bytes, link rate, local prefill rate |
| P1 | **Hidden runtime state changes costs:** KV dtype mismatch slow path ([S30](../sources/S30.md)); rotating-cache path divergence ([S33](../sources/S33.md)); Core ML compile cache 0.7–1.4 GB per bucket and minutes of cold compile ([S29](../sources/S29.md)) | A cost model that ignores state will be wrong | Log dtype, cache type, compile state, and warm or cold in every run manifest |
| P1 | **Energy telemetry:** `powermetrics` is package-level and needs privileges; no per-process attribution | Joules can only be a labeled estimate | Report as estimate with method, or omit; never zero |
| P2 | **Trainer gaps retained:** token weighting changes source influence; MLX gradient audit and 100-step check unrun; model is the 29,888-parameter toy | Only matters if training work resumes | [Heterogeneity](../guides/heterogeneity.md), [MLX validation](../guides/mlx-validation.md) stay as guides |
| P2 | AirTrain README bandwidth figures inconsistent in one example ([S03](../sources/S03.md)) | Unchanged | Not on the critical path |
| P2 | ANE/mactop counters unavailable in sandbox | Cannot infer zero use | Normal terminal plus isolated workloads when the gate runs |
| P2 | M5 unavailable; not planned | None for decision 0003 | Drop unless hardware appears |

## Research search coverage: 2026-10-25 local date

Fetched and read in full, as primary README or doc text: maderix/ANE top README, t8/hypura, AtomGradient/hybrid-ane-mlx-bench, eejd/rs-ane-infer, zw5/ane-fieldfare, RavenXAILabs/mlx-distributed-lora, tc3oliver/laya-apple, BTankut P/D DGX Spark plus Mac Studio, Aileuxe/mac-spark-playbook, MLX data- and tensor-parallel example pages, coremltools compute-units section; read in part: `mlx_lm/generate.py` speculative path and four mlx-lm issue bodies with 28 titles. The EXO README fetched on 2026-10-25 was byte-identical to [S18](../sources/S18.md) (SHA-256 match); API metadata shows the repository is not archived and names no successor. GitHub repository search was used only to resolve URLs the user described; search results were not treated as evidence.

Not covered: the AtomGradient paper PDF and its "hybird-batch-prefill-on-ane" successor; ANE-LM; Turbo Fieldfare; `PERFORMANCE.md` and `SpeculationPolicy` source in ANE Fieldfare; laya-apple's support-matrix and benchmark files; the DGX-Mac benchmark reports under `docs/`; any academic literature on adaptive speculation or KV reuse. These are leads, not read sources.

## Retrieval issues and provenance

2026-10-25: the uploaded `distributed-0.html` is a text rendering of the same MLX 0.32.4 "Distributed Communication" page as [S02](../sources/S02.md) (identical headings and sentences, different serialization and byte length); it was not intaken and S02 bytes were untouched. `https://ml-explore.github.io/mlx/build/html/examples/distributed_data_parallel.html` and `.../distributed_tensor_parallel.html` returned 404; the correct paths `examples/data_parallelism.html` and `examples/tensor_parallelism.html` were found from S02's link targets and saved as [S35](../sources/S35.md) and [S34](../sources/S34.md). `mlx_lm/examples/pipeline_generate.py` returned 404 on `main`; pipeline parallelism is therefore cited only from the EXO README. Apple Developer pages for `MLComputeUnits` and `MLComputePlan` returned a JavaScript-required shell with no content and were not saved; the coremltools guide was used instead ([S36](../sources/S36.md)). Earlier issues from 2026-10-22 remain as recorded in the log.

## Claim-resolution rules

When sources disagree, retain both claims with scopes and versions; inspect primary implementation or run an experiment. Never resolve a conflict by silently choosing the more exciting number. Distinguish public supported APIs, private experimental interfaces, inferred hardware presence, and measured execution. Report "not inspected" where appropriate. A green wiki lint result checks structure, not the truth of this ledger.
