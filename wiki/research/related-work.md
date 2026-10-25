---
title: "Related work and limits on novelty"
type: research
status: active
updated: 2026-10-23
sources: ["S02", "S03", "S04", "S05", "S06", "S07", "S08", "S09", "S10", "S11", "S12", "S18", "S19", "S20"]
---

# Related work and limits on novelty

This is an initial literature map, not an exhaustive review. Paper abstracts establish relevance; they do not establish implementation details or a fair baseline. Read the full closest papers before writing a novelty claim. Repository claims below have not been reproduced by MacReduce.

| Prior work | Overlap with MacReduce | Consequence for our proposal | Inspected scope |
|---|---|---|---|
| [FedAvg](../sources/S20.md) | Repeated local updates followed by aggregation | Model averaging itself is established | Abstract |
| [DiLoCo](../sources/S04.md) | Infrequent communication for language-model training | Local/outer optimizer semantics must be implemented faithfully before using this name | Abstract; full algorithm still required |
| [Streaming DiLoCo](../sources/S05.md) | Communication overlap, parameter subsets, compressed exchange | These broad directions are existing work | Abstract |
| [FedNova](../sources/S06.md) | Unequal local work and objective inconsistency | A speed policy must preserve an explicit optimization objective | Abstract and plain-SGD normalization equation in paper |
| [AdaComm](../sources/S07.md) | Adaptive synchronization frequency | “Adapt K” alone is not novel | Abstract and paper excerpt |
| [HEW-LocalSGD](../sources/S08.md) | Heterogeneity-aware weights, local horizons, mini-batches | Need a direct comparison of assumptions and policy; convex theory is not evidence for our Transformer | Abstract only |
| [AirTrain](../sources/S03.md) | Mac training over Wi-Fi, DiLoCo, discovery and observability | “Training on Macs over Wi-Fi” is not a defensible novelty claim | README only |
| [FusionML](../sources/S09.md) | CPU/GPU co-execution on Apple unified memory; calibration and lazy execution | Simple CPU/GPU splitting is already explored, including negative results | Abstract only |
| [ANEForge](../sources/S10.md), [maderix/ANE](../sources/S11.md), [Orion](../sources/S12.md) | Private-API ANE training research | ANE training is possible in experimental systems; it is neither a supported public training API nor a new idea by itself | Training docs / README / Orion abstract |
| [MLX distributed](../sources/S02.md) | Existing communication primitives | Reuse or benchmark against supported facilities instead of assuming all transport needs reinvention | Official docs |
| [EXO](../sources/S18.md) | Distributed inference on heterogeneous devices | A large model spread across Macs already has relevant implementations | README only |
| [MLX-LM LoRA](../sources/S19.md) | Practical parameter-efficient training | Start with a validated local baseline before distributed adapter research | Training docs |

## Useful negative evidence

FusionML's reported prefill, decode, and training outcomes differ. This challenges the assumption that adding accelerators always helps; it does not prove that every design loses. The [source note](../sources/S09.md) gives scope and limitations.

AirTrain's README contains throughput assertions that need independent verification. One stated example combines approximately 250 MB transfers, 30–100 Mbps, and 2–8 second synchronization. A single 250 MB payload alone requires approximately 20–67 seconds at those link rates before protocol overhead. The figures may refer to different conditions or quantities; record this ambiguity rather than treating the README as a benchmark or concluding the entire project is incorrect. See [the gaps ledger](gaps.md).

## Required follow-up search

Before selecting a contribution, review OpenDiLoCo, Decoupled DiLoCo, Hivemind, FedProx, SCAFFOLD, Oort, distributed LoRA aggregation, and current MLX/EXO issues and papers. Search combinations of `local SGD`, `heterogeneous`, `wireless`, `thermal`, `adaptive communication`, `Apple silicon`, `Neural Engine`, and `calibration overhead`. Log negative searches as coverage, never as proof of absence.

The [novelty candidates](novelty.md) are research hypotheses conditioned on this unfinished review.
