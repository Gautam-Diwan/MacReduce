---
title: "Unknowns contradictions and next literature searches"
type: research-ledger
status: active
updated: 2026-10-23
sources: ["S03", "S06", "S07", "S08", "S09", "S10", "S12"]
---

# Unknowns contradictions and next literature searches

This ledger distinguishes missing implementation, missing evidence, and incomplete literature coverage. It is not a backlog of already solved research problems.

| Priority | Gap or contradiction | Why it matters | Resolution |
|---|---|---|---|
| P0 | No physical multi-Mac/Wi-Fi result | Loopback cannot predict airtime contention or heterogeneous hardware | Run [Wi-Fi protocol](../guides/wifi.md), preserve raw measurements |
| P0 | MLX GPU path not validated in this sandbox | Implemented code is not numerical or device evidence | Run [MLX gates](../guides/mlx-validation.md) in a GPU-capable terminal |
| P0 | Current token weighting changes source influence with work | A faster run may optimize a different objective | Define p_i and test [objective-preserving aggregation](../guides/heterogeneity.md) |
| P1 | Fixed tiny model and 2 MB response cap | Current transport is unsuitable for materially larger models | Extend model configuration and bounded transfers together |
| P1 | ANE repository documentation covers multiple execution modes | “ANE training” may include host optimizer/state traffic | Pin code; inspect selected mode; validate gradients and full costs |
| P1 | Closest adaptive/heterogeneous methods not fully reviewed | Broad novelty claims would be unsupported | Read full AdaComm, FedNova, HEW-LocalSGD, DiLoCo and Streaming DiLoCo methods |
| P1 | FusionML reports limited training benefit from CPU/GPU co-execution | Adding devices may reduce speed | Reproduce matched workload before a co-execution scheduler |
| P1 | AirTrain README bandwidth/time figures appear inconsistent in one example | Marketing numbers are not a reliable baseline | Verify units, exact revision, actual payload and end-to-end trace; contact maintainer only if the team chooses |
| P2 | M5 unavailable locally | GPU neural-accelerator dispatch and speed cannot be measured here | Capability-aware fallback now; validate on borrowed M5 later |
| P2 | ANE/mactop counters unavailable in sandbox | Cannot infer zero use or reliable attribution | Normal terminal, Instruments plus isolated workloads; document permissions and counter meaning |
| P2 | Large-model inference separate from trainer | Training results do not validate sharding or KV-cache memory | Evaluate EXO separately with [inference guide](../guides/inference.md) |
| P2 | Federated LoRA and model-merging review incomplete | Factor averaging has known issues and prior solutions | Search/read relevant methods before implementing a claimed new reducer |

## Research search coverage: 2026-10-22 local date

The initial review fetched official MLX/Apple documentation, public ANE training project documentation, distributed training papers, a Mac Wi-Fi training repository, and a distributed inference repository. Source snapshots were retrieved late on October 22 Eastern time (October 23 UTC); the manifest stores UTC times. The [source notes](../index.md#sources) distinguish abstracts, README files, and documentation from audited implementations.

Topics examined included Apple Neural Engine training APIs; Mac Wi-Fi distributed training; local-SGD communication reduction; heterogeneous local steps and objective inconsistency; Apple CPU/GPU co-execution; M5 GPU neural accelerators; SME capability and profiling; Core ML/Metal placement; and distributed inference. This record describes coverage, not an exhaustive or reproducible systematic literature review.

**Searches still required:** Hivemind/OpenDiLoCo/Decoupled DiLoCo implementation and scaling; wireless federated scheduling and client-selection methods; FedProx/SCAFFOLD/Oort; online autotuning with probe interference; federated LoRA factor alignment/aggregation; ANE training papers and repository revisions after this snapshot. Record query, date, database, candidates, exclusions, and inspected scope when doing this review.

## Retrieval issues and provenance

The gist's normal page and API retrieval initially failed; its raw `llm-wiki.md` URL succeeded. AirTrain's `main` README URL failed and `master` succeeded. MLX-LM's top-level `LORA.md` URL failed and `mlx_lm/LORA.md` succeeded. These are access/path failures, not evidence that content does not exist. Snapshot checksums identify saved bytes, not commits. Mutable documentation and unversioned arXiv URLs should be refreshed before relying on current behavior.

## Claim-resolution rules

When sources disagree, retain both claims with scopes and versions; inspect primary implementation or run an experiment. Never resolve a conflict by silently choosing the more exciting number. Distinguish public supported APIs, private experimental interfaces, inferred hardware presence, and measured execution. Report “not inspected” where appropriate. A green wiki lint result checks structure, not the truth of this ledger.
