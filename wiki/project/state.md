---
title: "Current implementation and evidence"
type: status
status: measured
updated: 2026-10-25
sources: []
---

# Current implementation and evidence

**Scope as of 2026-10-25:** [decision 0003](../decisions/0003-build-target.md) makes the deliverable a per-request phase-placement scheduler for LLM serving plus its benchmark. The scheduler and a one-M4 Qwen2.5 study now exist. The serving pin is Qwen3 1.7B/0.6B 4-bit; that pair has an equivalence check, not a new speed sweep. The local-SGD trainer remains a measured prototype, not the research claim. The joint hypothesis, better time-to-first-token and better decode rate than the hindsight depth, was not met on the Qwen2.5 study.

**Measured locally:** MacReduce is the renamed version of the original MacTrain prototype. The canonical package is `macreduce`, the console command is `macreduce`, and environment settings use `MACREDUCE_`. Historical files keep their original labels. [Migration record](../../docs/archive/migration.json).

## Scheduler and benchmark (decision 0003)

| Capability | State | Evidence or next step |
|---|---|---|
| Placement rule for engine, depth, and prefix | Implemented and unit-tested | `src/macreduce/placement.py`. ANE is ineligible unless `placement_verified`. Missing rates stay unknown |
| Pinned `mlx-lm` 0.32.2 executor | Qwen2.5 sweep measured; Qwen3 pin checked for tokens | The speed study used Qwen2.5 and is unchanged. The current pin is Qwen3 1.7B/0.6B 4-bit. Depths 1, 2, and 4 matched greedy tokens for 32 tokens. Depth 8 mismatched the short prompt at token 26. A local Gemma 4 E4B 4-bit target runs, and its MTP assistant is refused by this mlx-lm. Qwen3.5 2B loads, and speculative decoding refuses its linear-attention cache. The 4B config uses the same cache rule. [Qwen3 check](../../evidence/serve-equivalence-qwen3/README.md), [Gemma check](../../evidence/gemma4-local/README.md), [Qwen3.5 check](../../evidence/qwen35-local/README.md), [Qwen2.5 study](../../evidence/serve-benchmark/README.md) |
| Trace and depth sweep | Measured on this M4 for Qwen2.5 | [Qwen2.5 study](../../evidence/serve-benchmark/README.md). Short prompts prefer no draft. Long prompts decode fastest at depth 4 and reach first token fastest at depth 0. Code decodes fastest at depth 2. Those rates are not Qwen3 rates |
| D2 cost-model depth | Measured; hypothesis not met | The model picked the hindsight decode depth for short and long, and depth 4 instead of 2 for code, which was slower on the evaluation pass. Per-step depth changes were not run |
| D1 CPU draft | Measured once on the short prompt | Tokens matched. Decode was about 88 tok/s against about 97 tok/s with no draft. One stream, not a concurrency claim |
| D1 ANE gate | Convolution plan measured; mactop saw one ANE burst; drafter not converted | `coremltools==9.0`. `CPU_AND_NE` assigns the conv and ReLU to the Neural Engine device. `mactop` v2.1.5 peaked at 0.607 W and 4.81 GB/s on the first `CPU_AND_NE` sample, then returned to the idle floor. `CPU_ONLY` stayed at 0. `ane_active` stayed 0. `placement_verified` stays false. [Probe](../../evidence/ane-probe/README.md) |
| D3 prefix, warm cache | Measured | Reloading the long prefix matched recomputed tokens. The 0.9 ms load is a warm page cache of a file just written, not cold disk and not a network |
| D3 remote arm | Not run | No second Mac |

## Local-SGD trainer prototype (retained)

| Capability | State | Evidence or next step |
|---|---|---|
| Causal byte Transformer and CPU SGD | Executed | Default remains the 29,888-parameter one-block model; [original reference result](../../evidence/reference-run/result.json) |
| Serializable `ModelConfig` | Implemented and tested | Vocab, width, heads, layers, context, dtype, tokenizer. Default matches the historical parameter bytes. A 2-layer config round-trips and trains on CPU |
| Coordinator and independent workers | Executed on loopback | Two workers, batches 2 and 4, eight rounds, ten steps, loss 6.0587 to 1.7681 |
| Weighted averaging, retries, restart | Tested | [Tests](../../tests/test_cluster.py), [numerical reference](../../tests/test_model.py) |
| Chunked artifact transfer | Implemented and tested | 64 MiB cap, manifest, chunk hashes, temp file, atomic complete. Reusable for KV-file movement in D3 |
| MLX GPU path | Measured on this M4, partially | Doctor smoke, forward and one SGD step versus CPU for the default and 2-layer configs, and a 3-round loopback demo. No Metal capture. [Record](../../evidence/mlx-validation/README.md) |
| Wi-Fi and mixed physical Macs | Not tested; no longer a research goal | [Wi-Fi protocol](../guides/wifi.md) kept as an operational guide |
| CPU operator traces and Accelerate SGEMM | Executed | [Evidence inventory](../../evidence/README.md) |
| ANE training | Not implemented; not planned | [ANE integration gates](../guides/ane.md) |
| M5 GPU neural accelerators | Inventory concept; no M5 test; not planned | [Hardware guide](../guides/accelerators.md) |
| Real corpus and 1M-parameter model | Not implemented; not on the critical path | [Model scaling](../guides/model-scaling.md) |
| Large model across several Macs | Not implemented; covered by MLX and EXO, not planned | [Inference guide](../guides/inference.md) |

Every training worker holds the complete model. A toy-model loss decrease establishes learning through the protocol, not useful LLM quality or multi-Mac speedup.

The original tests reported 20 passed and one intentionally skipped Metal test. After the rename, the suite passed 25 tests with one opt-in MLX test skipped, and a CPU loopback run completed eight rounds with loss 6.058714 to 1.768063. [Migration tests](../../evidence/macreduce-migration/tests.txt), [new run](../../evidence/macreduce-migration/reference-run/result.json), and [preservation report](../../evidence/macreduce-migration/preservation.json) are stored separately; historical evidence files were not rewritten.

**Measured on 2026-10-24:** the default suite passes **33 tests, with 2 opt-in MLX tests skipped**. With `MACREDUCE_TEST_MLX=1`, those two tests pass. `ruff check .` passes. On this M4, `macreduce doctor --smoke` ran a Metal matmul (result 4096.0). A loopback `mlx-gpu` demo of 3 rounds and 4 local steps finished with validation loss 6.058713912963867 to 3.713829755783081. [MLX record](../../evidence/mlx-validation/README.md). `xctrace` was unavailable. That demo is one machine and is not a Wi-Fi or speedup result.

**Known hardware:** one M4 (Mac16,12), macOS 26.5, MLX 0.32.4. Other machines, their RAM, and any interconnect are **unknown**. Thunderbolt 5 RDMA is not assumed.

The proposal in [docs/proposal.md](../../docs/proposal.md) predates decision 0003 and must be revised before 2026-11-06. Current implementation details in this wiki take precedence over old plans when they disagree.
