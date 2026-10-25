---
title: "Current implementation and evidence"
type: status
status: measured
updated: 2026-10-23
sources: []
---

# Current implementation and evidence

**Measured locally:** MacReduce is the renamed version of the original MacTrain prototype. The canonical package is `macreduce`, the console command is `macreduce`, and environment settings use `MACREDUCE_`. Historical files keep their original labels. [Migration record](../../docs/archive/migration.json).

| Capability | State | Evidence or next step |
|---|---|---|
| Causal byte Transformer and CPU SGD | Executed | 29,888 parameters; [original reference result](../../evidence/reference-run/result.json) |
| Coordinator and independent workers | Executed on loopback | Two workers, batches 2 and 4, eight rounds, ten steps, loss 6.0587 to 1.7681 |
| Weighted averaging, retries, restart | Tested | [Tests](../../tests/test_cluster.py), [numerical reference](../../tests/test_model.py) |
| MLX CPU/GPU backend | Implemented; hardware validation pending | Metal import was blocked in the original sandbox; [validation guide](../guides/mlx-validation.md) |
| Wi-Fi and mixed physical Macs | Not tested | [Wi-Fi protocol](../guides/wifi.md) |
| CPU operator traces and Accelerate SGEMM | Executed | [Evidence inventory](../../evidence/README.md) |
| ANE training | Not implemented | [ANE integration gates](../guides/ane.md) |
| M5 GPU neural accelerators | Inventory concept; no M5 test | [Hardware guide](../guides/accelerators.md) |
| Automatic batch assignment or adaptive K | Not implemented | [Heterogeneity guide](../guides/heterogeneity.md) |
| Real corpus and larger model | Extension work | [Model scaling](../guides/model-scaling.md) |
| Large model across several Macs | Not implemented | [Separate inference path](../guides/inference.md) |

Every training worker currently holds the complete model. The code does not pool model memory. A toy-model loss decrease establishes learning through the protocol, not useful LLM quality or multi-Mac speedup.

The original tests reported 20 passed and one intentionally skipped Metal test. The renamed project now passes **25 tests, with one opt-in MLX test skipped**. A fresh CPU loopback run completed all eight rounds with loss 6.058714 to 1.768063. [Migration tests](../../evidence/macreduce-migration/tests.txt), [new run](../../evidence/macreduce-migration/reference-run/result.json), and [preservation report](../../evidence/macreduce-migration/preservation.json) are stored separately; all 31 historical evidence files retain their original hashes. Raw event logs are retained unchanged, including old run labels and paths.

The proposal is in [docs/proposal.md](../../docs/proposal.md). The original DOCX and feasibility report are historical inputs in `docs/archive/`. Current implementation details in this wiki take precedence over old plans when they disagree.
