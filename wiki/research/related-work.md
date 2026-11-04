---
title: "Related work and limits on novelty"
type: research
status: active
updated: 2026-10-28
sources: ["S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S18", "S19", "S20", "S23", "S24", "S25", "S26", "S27", "S28", "S29", "S30", "S31", "S32", "S33", "S34", "S35", "S36", "S37"]
---

# Related work and limits on novelty

This is a literature and systems map, not an exhaustive review. README files and documentation establish what maintainers report; they do not establish implementation details or a fair baseline. Repository claims below have not been reproduced by MacReduce. The map is organized around [decision 0003](../decisions/0003-build-target.md).

## Connecting Macs (covered; do not build)

| Prior work | What it covers | Consequence | Inspected scope |
|---|---|---|---|
| [MLX distributed](../sources/S02.md) | ring (TCP), MPI, JACCL (RDMA over Thunderbolt, macOS 26.2+, Thunderbolt 5), `mlx.launch`, `mlx.distributed_config` | First-party transport; no collectives to write | Official docs, main sections read |
| [MLX tensor parallelism](../sources/S34.md) | `AllToShardedLinear`, `ShardedToAllLinear`, `shard_linear` | Tensor-parallel inference across Macs exists | Sharded-layer section |
| [MLX data parallelism](../sources/S35.md) | `nn.average_gradients` | Data-parallel training is one line | Complete page |
| [EXO](../sources/S18.md) | Discovery, topology-aware auto-parallel, pipeline or tensor sharding, RDMA over Thunderbolt | The cluster-inference product space is occupied; repository active as of 2026-10-13 | README only, re-read 2026-10-25 |
| [RavenX distributed LoRA](../sources/S28.md) | Stock `mlx_lm lora` data-parallel over Thunderbolt; JACCL provider bugs | Distributed fine-tuning across Macs is done with stock tools | README only |
| [AirTrain](../sources/S03.md) | Mac training over Wi-Fi, DiLoCo, discovery | "Training on Macs over Wi-Fi" is not a claim | README only |

## Synchronization schedules (covered)

| Prior work | Overlap | Consequence | Inspected scope |
|---|---|---|---|
| [FedAvg](../sources/S20.md), [DiLoCo](../sources/S04.md), [Streaming DiLoCo](../sources/S05.md) | Infrequent averaging and overlap | Local SGD schedule is established | Abstracts |
| [FedNova](../sources/S06.md), [AdaComm](../sources/S07.md), [HEW-LocalSGD](../sources/S08.md) | Unequal local work, adaptive K, heterogeneity weights | "Adapt K" or "weight by speed" alone is not novel; objective must be preserved ([decision 0002](../decisions/0002-measurements.md)) | Abstracts and excerpts |

## ANE and engine selection on one Mac (largely covered; one gap)

| Prior work | What it covers | Consequence | Inspected scope |
|---|---|---|---|
| [ANEForge](../sources/S10.md), [maderix training README](../sources/S11.md), [maderix top README](../sources/S23.md), [Orion](../sources/S12.md) | Private-API ANE training; GPU-prefill to ANE-decode over IOSurface (S23) | ANE training is prototyped; 5–9% utilization and CPU dW reported by S23 | Docs and READMEs; Orion abstract |
| [AtomGradient bench](../sources/S25.md) | Five prefill/decode strategies on M2 Ultra; Core ML `ALL` routed to GPU on macOS 26.3; incompatible KV formats | ANE prefill plus GPU decode is benchmarked with negatives; ANE placement must be verified | README only |
| [rs-ane-infer](../sources/S26.md) | ANE prefill, Metal decode, CPU GEMV on M5; "needs separate tiny draft model" | A second handoff implementation; names the ANE-drafter question without testing it | README only |
| [ANE Fieldfare](../sources/S27.md) | ANE expert prefill for an MoE on 16 GB M5; DFlash drafter; per-step draft-length policy from acceptance; fail-closed or failover | Adaptive speculation depth exists inside one runtime; the benchmark method is worth copying | README only |
| [laya-apple](../sources/S29.md) | Concurrent GPU plus ANE serving with per-request routing and a parity gate for a classifier; 4.57x and 79.5 ms P99 project-reported on one M4 Max | Routing a short-request stream to the ANE is done; LLM decode and online updating are not in that source | README only |
| [coremltools compute units](../sources/S36.md) | `compute_units` is a load-time constraint | Within-model placement is Apple's; a scheduler must verify placement | One section |
| [ANE architecture reference](../sources/S37.md) | Direct user-space route, fp16 datapath, training built from forward ops, M1 and M5 measurements | Engine training is a published capability. The read chapters are not a Qwen3 QLoRA result and do not measure this M4 | Abstract, chapter 1, chapter 6 opening, 14.4–14.10, chapter 15 |
| [FusionML](../sources/S09.md) | CPU/GPU co-execution on unified memory, calibration costs | Adding engines can lose; calibration costs count | Abstract |
| [Hypura](../sources/S24.md) | GPU/RAM/NVMe tensor placement with a profile-then-solve step on one Mac | Memory-tier placement is done | README only |

## Prefill/decode disaggregation and KV reuse (covered cross-machine; rule still open)

| Prior work | What it covers | Consequence | Inspected scope |
|---|---|---|---|
| [BTankut P/D](../sources/S30.md) | DGX Spark prefill, M4 Max decode, ZMQ KV over 10GbE, crossover 8K–16K ctx; dtype-mismatch slow path | The system exists; the crossover is hardware- and workload-specific | README only |
| [Mac Spark playbook](../sources/S31.md) | GB10 prefill, M3 Ultra decode over RDMA; split wins on long code prompts, loses on prose TTFT | A second table that disagrees with the first | README only |
| [mlx-lm prompt cache](../sources/S32.md) | `mlx_lm.cache_prompt`, `--prefill-step-size` | Local prefix reuse exists as a file; the reuse-versus-recompute decision is the caller's | README and code |

## Speculative decoding on MLX (covered; policy and verification open)

| Prior work | What it covers | Consequence | Inspected scope |
|---|---|---|---|
| [mlx-lm generate.py](../sources/S32.md) | `--draft-model`, fixed `--num-draft-tokens`, same-device draft, cache rewind by trimming | Speculation exists; engine choice and depth policy do not | Code read, not run |
| [mlx-lm issue search](../sources/S33.md) | Non-trimmable cache failure (#1446), greedy divergence (#1423), skipped tokens (#846), rotating-cache path divergence (#1914), MoE warnings | Baselines must be pinned and equivalence-checked | Four issue bodies, 28 titles |

## Useful negative evidence

FusionML's reported prefill, decode, and training outcomes differ ([S09](../sources/S09.md)). AtomGradient found no crossover for its CoreML hybrid on a 9B model and found the ANE flag did not place work on the ANE ([S25](../sources/S25.md)). BTankut found DGX-only beats the split at 16K+ context ([S30](../sources/S30.md)). AirTrain's README bandwidth example remains internally inconsistent ([S03](../sources/S03.md), [gaps](gaps.md)). These are reasons to measure crossovers rather than assume them.

## Required follow-up search (owed before any novelty wording)

- Dynamic or adaptive speculation depth and draft-model selection: search `adaptive speculative decoding`, `dynamic draft length`, `speculative decoding scheduler`, `heterogeneous draft verify`. Read the nearest two papers in full.
- KV-cache reuse and transfer policies: `KV cache transfer vs recompute`, `prefix caching policy`, `disaggregated prefill decode scheduling`, `KV compression for transfer`.
- Engine placement on NPUs: `NPU GPU co-scheduling LLM`, `heterogeneous SoC inference scheduler`.
- Still owed from 2026-10-23: OpenDiLoCo, Decoupled DiLoCo, Hivemind, FedProx, SCAFFOLD, Oort, federated LoRA aggregation, if training work resumes.

Log each search with query, date, database, candidates, exclusions, and inspected scope. Negative searches are coverage, never proof of absence.
