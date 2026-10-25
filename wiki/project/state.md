---
title: "Current implementation and evidence"
type: status
status: measured
updated: 2026-10-24
sources: []
---

# Current implementation and evidence

**Measured locally:** MacReduce is the renamed version of the original MacTrain prototype. The canonical package is `macreduce`, the console command is `macreduce`, and environment settings use `MACREDUCE_`. Historical files keep their original labels. [Migration record](../../docs/archive/migration.json).

| Capability | State | Evidence or next step |
|---|---|---|
| Causal byte Transformer and CPU SGD | Executed | Default remains the 29,888-parameter one-block model; [original reference result](../../evidence/reference-run/result.json) |
| Serializable `ModelConfig` | Implemented and tested | Vocab, width, heads, layers, context, dtype, tokenizer. Default matches the historical parameter bytes. A 2-layer config round-trips and trains on CPU |
| Coordinator and independent workers | Executed on loopback | Two workers, batches 2 and 4, eight rounds, ten steps, loss 6.0587 to 1.7681 |
| Weighted averaging, retries, restart | Tested | [Tests](../../tests/test_cluster.py), [numerical reference](../../tests/test_model.py) |
| Chunked artifact transfer | Implemented and tested | 64 MiB cap, manifest, chunk hashes, temp file, atomic complete. Happy path, corrupt chunk, duplicate retry, oversize, and partial rejection. Compression and the full fault list are not done |
| MLX GPU path | Measured on this M4, partially | Doctor smoke, forward and one SGD step versus CPU for the default and 2-layer configs, and a 3-round loopback demo. No Metal capture. [Record](../../evidence/mlx-validation/README.md) |
| Wi-Fi and mixed physical Macs | Not tested | [Wi-Fi protocol](../guides/wifi.md) |
| CPU operator traces and Accelerate SGEMM | Executed | [Evidence inventory](../../evidence/README.md) |
| ANE training | Not implemented | [ANE integration gates](../guides/ane.md) |
| M5 GPU neural accelerators | Inventory concept; no M5 test | [Hardware guide](../guides/accelerators.md) |
| Automatic batch assignment or adaptive K | Not implemented | [Heterogeneity guide](../guides/heterogeneity.md) |
| Real corpus and 1M-parameter model | Not implemented | Configuration and chunked transfer are in place; [model scaling](../guides/model-scaling.md) |
| Large model across several Macs | Not implemented | [Separate inference path](../guides/inference.md) |

Every training worker currently holds the complete model. The code does not pool model memory. A toy-model loss decrease establishes learning through the protocol, not useful LLM quality or multi-Mac speedup.

The original tests reported 20 passed and one intentionally skipped Metal test. After the rename, the suite passed 25 tests with one opt-in MLX test skipped, and a CPU loopback run completed eight rounds with loss 6.058714 to 1.768063. [Migration tests](../../evidence/macreduce-migration/tests.txt), [new run](../../evidence/macreduce-migration/reference-run/result.json), and [preservation report](../../evidence/macreduce-migration/preservation.json) are stored separately; historical evidence files were not rewritten.

**Measured on 2026-10-24:** the default suite passes **33 tests, with 2 opt-in MLX tests skipped**. With `MACREDUCE_TEST_MLX=1`, those two tests pass (default model and a 2-layer config). `ruff check .` passes. On this M4, `macreduce doctor --smoke` ran a Metal matmul (result 4096.0). A loopback `mlx-gpu` demo of 3 rounds and 4 local steps finished with validation loss 6.058713912963867 to 3.713829755783081. [MLX record](../../evidence/mlx-validation/README.md). `xctrace` was unavailable, so GPU execution was not captured in Xcode. That demo is one machine and is not a Wi-Fi or speedup result.

The proposal is in [docs/proposal.md](../../docs/proposal.md). The original DOCX and feasibility report are historical inputs in `docs/archive/`. Current implementation details in this wiki take precedence over old plans when they disagree.
