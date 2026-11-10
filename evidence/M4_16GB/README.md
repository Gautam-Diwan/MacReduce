# Qwen3-0.6B QLoRA on an M4 MacBook Pro, 16 GB

These runs answer two questions on a second Mac model:

1. **Across Macs:** how does the GPU plus Neural Engine baseline behave on this Mac compared with the baseline Mac? The baseline is [evidence/qwen3-0.6b-curve/2026-10-29T00-07-18](../qwen3-0.6b-curve/2026-10-29T00-07-18/result.json), run by Gautam Diwan.
2. **On this Mac:** how does GPU plus Neural Engine compare with GPU only, using the same recipe?

Labels: **measured** means read from the files in this folder. **Source-reported** means taken from the project wiki or handoff and not re-checked here. **Inferred** means a reasoned interpretation. **Unknown** means not established.

## Summary

All numbers are measured unless marked otherwise.

- **Same quality on both Macs.** GPU plus Neural Engine on this Mac reproduced the baseline exactly. All 1,229 train-loss reports, every validation point, and the adapted test perplexity (23.2974) are equal, across macOS 15.7.9 and 26.5 and 16 GB and 24 GB. A second run on this Mac was also identical.
- **Faster on this Mac.** In the clean run, a training step takes 210 ms, against 288 to 295 ms in the baseline's steady epochs, a ratio of about 1.4. The cause is **unknown**.
- **Energy across Macs: no comparison yet.** The baseline's power record is incomplete, so these runs cannot compare energy with it.
- **GPU only is faster than GPU plus Neural Engine on this Mac.** The two reach the same quality: test perplexity 23.31 for GPU only and 23.30 for GPU plus Neural Engine. GPU only is 1.41 times faster per step (148.6 against 210.1 ms median). It finishes in 49 instead of 67 minutes and peaks at 2.2 instead of 4.3 GB of MLX memory. It uses about 5% less energy across the CPU, GPU, Neural Engine, and DRAM rails (44.7 against 46.8 kJ).

## Machines

| | This Mac (measured) | Baseline Mac (source-reported) |
|---|---|---|
| Model | MacBook Pro, Mac16,1 | MacBook Air, Mac16,12 ([state](../../wiki/project/state.md), [handoff](../../docs/handoff-2026-10-28-gpu-ane-qlora.md)) |
| Chip | Apple M4, 10-core GPU | Apple M4, GPU core count unknown (the M4 Air ships with 8 or 10) |
| Memory | 16 GB | 24 GB |
| Cooling | Fan | Fanless |
| macOS | 15.7.9 | 26.5 |
| MLX recommended working set | 11,453,251,584 bytes | 19,069,665,280 bytes |

Software on this Mac: Python 3.12.16, mlx 0.32.4, mlx-lm 0.32.2, aneforge 0.4.0, mactop 2.1.6, installed with `uv sync` from `uv.lock` at commit `d67f2e3`. The default test suite passed 93 tests with 5 skipped. With `MACREDUCE_TEST_MLX=1`, all 98 passed, including the one-block Neural Engine parity test.

## Runs

| Run | Config | Conditions |
|---|---|---|
| [smoke_gpu_ane/2026-11-09T00-17-05](smoke_gpu_ane/2026-11-09T00-17-05/result.json) | [smoke_gpu_ane.yaml](../../configs/M4_16GB/smoke_gpu_ane.yaml), first 500 steps | No `caffeinate`. Chrome played a YouTube live stream throughout |
| [gpu_ane/2026-11-09T01-14-02](gpu_ane/2026-11-09T01-14-02/result.json) | [qlora_gpu_ane.yaml](../../configs/M4_16GB/qlora_gpu_ane.yaml) | `caffeinate -i`. YouTube live stream from 9.3 minutes after launch to the end |
| [gpu/2026-11-10T01-09-44](gpu/2026-11-10T01-09-44/result.json) | [qlora_gpu.yaml](../../configs/M4_16GB/qlora_gpu.yaml) | Clean pair, run back to back with a 300 s pause under one `caffeinate -di` |
| [gpu_ane/2026-11-10T02-03-55](gpu_ane/2026-11-10T02-03-55/result.json) | [qlora_gpu_ane.yaml](../../configs/M4_16GB/qlora_gpu_ane.yaml) | Second run of the clean pair |

All runs were plugged in with Low Power Mode off, and `pmset` logged no sleep during any of them. For the clean pair, `pmset` shows `caffeinate` holding its assertions from 01:09:44 to 03:11:16. It shows no browser video wake lock and no audio assertion in that window, and the last keyboard activity timed out at 01:21:27. The CPU rail averaged 0.75 W (GPU only) and 1.30 W (GPU plus Neural Engine), against 2.20 W in the first full run with the stream playing. Which other applications were open during the clean pair was not recorded.

The two full configs are [configs/qlora_m4.yaml](../../configs/qlora_m4.yaml) with `batch_size: 1` and `grad_checkpoint: false`, the values the baseline's batch search chose, plus their own `logging.dir`. GPU only differs from GPU plus Neural Engine only in `placement.mode: gpu`. Both keep batch size 1 so the step count (12,288) and the learning-rate schedule are identical.

```sh
caffeinate -di sh -c 'uv run python main.py --config configs/M4_16GB/qlora_gpu.yaml; sleep 300; uv run python main.py --config configs/M4_16GB/qlora_gpu_ane.yaml'
```

## Data preparation

The first launch failed before training with `HTTP Error 502: Bad Gateway` from the Hugging Face rows API, which `prepare_dataset` calls about 420 times with no retry. That failed run directory was removed. The files in `data/wikitext-2-raw-v1` were then written by the repository's own `prepare_dataset`, called from a one-off script that retried transient HTTP errors. No repository code changed. The record counts match the baseline manifest: train 23,767, validation 2,461, test 2,891. The Hub's current revision of `Salesforce/wikitext` is the pinned `b08601e04326c79dfdd32d625aee71d232d685c3`. A GPU-only launch that was slowed by an open browser was stopped at step 550 and removed; it is not part of this record.

## Smoke run: first 500 steps of the baseline

The smoke config is the baseline config with three changes:

- `iters: 500` stops early.
- `epochs: 1` places validation at steps 1 and 500 and one checkpoint at step 500.
- `lr_schedule.arguments: [0.0001, 11988, 1.0e-6]` pins the cosine to the baseline's resolved schedule, so the learning rate at every step equals the baseline's.

Seed, data, batch size 1, sequence 512, LoRA rank 8 and scale 2, and the full validation and test splits are unchanged.

| Measured | This Mac | Baseline, same steps |
|---|---|---|
| Neural Engine | Blocks 0–26 on the GPU, block 27 on the Neural Engine at mask `0x4`; 5,422 forward and 500 backward calls | Same placement |
| Untrained validation loss (step 1) | 4.1913 | 4.1913 |
| Untrained test perplexity | 64.2367 | 64.2368 |
| Train loss and learning rate at 50 reports, steps 10–500 | Equal to the baseline at every report | — |
| Validation loss at step 500 | 3.1767 (perplexity 23.97) | No validation at step 500 |
| Test perplexity after 500 steps | 23.088 | — |
| Median step time | 234 ms (431 tokens/s) | 541 ms over the same steps |
| Wall time | 945 s | — |

## GPU plus Neural Engine across Macs

### Quality: identical

| Measured | This Mac, both full runs | Baseline |
|---|---|---|
| Steps | 12,288 (3 epochs, batch 1) | 12,288 |
| Neural Engine calls | 22,132 forward, 12,288 backward, mask `0x4` | Same |
| Train loss, 1,229 reports | Equal at every report | — |
| Validation perplexity at steps 0, 4095, 8191, 12287 | 66.111, 21.321, 22.085, 23.723 | Same values |
| Adapted test loss and perplexity | 3.148341417312622, 23.2974 | Same values |
| MLX peak memory | 4,260,406,834 bytes | 4,260,406,834 bytes |

The two Macs produce the same training trajectory and the same adapter quality to the logged precision. The two runs on this Mac are also identical, so this path is deterministic here. Validation perplexity is lowest after epoch 1 and rises after it in every full run.

### Speed

| Measured | This Mac, clean run | This Mac, first run | Baseline |
|---|---|---|---|
| Wall time | 4,041 s (67 min) | 4,411 s (74 min) | 13,752 s (3 h 49 min) |
| Median step time, epochs 1 / 2 / 3 | 209 / 210 / 211 ms | 228 / 227 / 229 ms | 502 / 295 / 288 ms |
| Median tokens/s | 477 | 441 | 319 |
| Reports below 1 step/s | 2 of 1,229 | 12 of 1,229 | 154 of 1,229 |
| Full validation pass, steps 0 / 4095 / 8191 / 12287 | 199 / 202 / 216 / 226 s | 195 / 201 / 215 / 239 s | 368 / 409 / 413 / 434 s |

Background load matters at the 10% level: the clean run is 8% faster per step than the first run on the same Mac (measured), and the first run had the stream and a higher CPU rail (inferred cause). The baseline's first epoch is not a clean measurement either. It has many sub-1 step/s reports, and the handoff records concurrent experiments on that machine the same night (source-reported).

Comparing steady states, the clean run on this Mac is about 1.37 to 1.40 times faster per step than the baseline's epochs 2 and 3, and validation passes are about 1.8 to 2.0 times faster. Candidate causes, none tested: GPU core count (unknown on the baseline Mac), fan versus fanless cooling under sustained load, background load, and macOS version.

In both GPU plus Neural Engine runs on this Mac, validation time rises across the run (199 to 226 s and 195 to 239 s) while training step time stays flat. GPU-only validation stays at 175 to 179 s. The cause is **unknown**.

### Energy

mactop samples are system-wide, not per process.

| Measured | This Mac, clean run | This Mac, first run | Baseline |
|---|---|---|---|
| mactop coverage | 4,024 s of 4,024 s, no gap over 5 s, exit 0 | 4,374 s of 4,375 s, no gap over 5 s, exit 0 | 8,595 s of 13,721 s, a 359 s gap near the start, exit -9 |
| GPU rail, average | 9.00 W (36.2 kJ) | 8.32 W (36.4 kJ) | 1.40 W (12.0 kJ) |
| GPU rail, median sample | 9.51 W | 8.95 W | 0.23 W |
| Neural Engine rail, average | 0.081 W (326 J) | 0.083 W (364 J) | 0.011 W (98 J) |
| CPU rail, average | 1.30 W (5.2 kJ) | 2.20 W (9.6 kJ) | 3.88 W (33.4 kJ) |
| DRAM rail, average | 1.24 W (5.0 kJ) | 1.28 W (5.6 kJ) | 1.26 W (10.8 kJ) |

The baseline's power record cannot support a cross-machine energy comparison. It stops about 85 minutes before training ended. After its first tenth, its GPU median is about 0.2 W while the run kept training at about 290 ms per step, which a GPU-resident training step should not allow (inferred). The cause is **unknown**.

## GPU only against GPU plus Neural Engine on this Mac

This section compares the two runs of the clean pair, which had the same conditions and the same work. Each trained on 1,248,807 tokens and ran four full validation passes and two full test passes.

| Measured | GPU only | GPU plus Neural Engine | Ratio |
|---|---|---|---|
| Placement | All 28 blocks on the GPU | Blocks 0–26 on the GPU, block 27 on the Neural Engine | — |
| Untrained test perplexity | 64.2365 | 64.2368 | — |
| Validation perplexity at steps 0, 4095, 8191, 12287 | 66.106, 21.293, 22.121, 23.762 | 66.111, 21.321, 22.085, 23.723 | — |
| Adapted test perplexity | 23.3089 | 23.2974 | — |
| Mean train loss, epochs 1 / 2 / 3 | 2.8538 / 2.3258 / 2.0471 | 2.8540 / 2.3262 / 2.0490 | — |
| Median step time, epochs 1 / 2 / 3 (whole run) | 152 / 147 / 147 ms (148.6) | 209 / 210 / 211 ms (210.1) | 1.41 |
| Median tokens/s | 674 | 477 | 1.41 |
| Full validation pass | 175 to 179 s | 199 to 226 s | — |
| Wall time | 2,950 s (49 min) | 4,041 s (67 min) | 1.37 |
| MLX peak memory | 2,206,027,662 bytes | 4,260,406,834 bytes | 1.93 |
| GPU rail, average | 13.08 W (38.4 kJ) | 9.00 W (36.2 kJ) | — |
| Neural Engine rail, average | 0.000 W (0 J) | 0.081 W (326 J) | — |
| CPU rail, average | 0.75 W (2.2 kJ) | 1.30 W (5.2 kJ) | — |
| DRAM rail, average | 1.37 W (4.0 kJ) | 1.24 W (5.0 kJ) | — |
| CPU, GPU, Neural Engine, and DRAM rails, total | 44.7 kJ | 46.8 kJ | 1.05 |

**Quality.** The trajectories are not bit-identical: per-report train loss differs by a median of 0.008 and at most 0.117. Block 27 is computed by a different engine in each run. The validation, test, and per-epoch mean losses agree within 0.2%, so quality is the same at this precision.

**Speed and energy.** In this placement the Neural Engine slows training. The GPU draws less average power with the Neural Engine helping, 9.0 against 13.1 W. The run takes 37% longer, so total energy across the four rails is 5% higher.

**Why it is slower (inferred, from the [handoff](../../docs/handoff-2026-10-28-gpu-ane-qlora.md) design notes).**
- The Neural Engine carries 1 of 28 blocks.
- The two devices alternate within a step with no overlap.
- Each row is padded to sequence 512 and run sequentially on the Neural Engine.
- Converting data between the two devices adds work.

The extra 2 GB of MLX memory is consistent with the dequantized fp16 copy of block 27 and its Neural Engine buffers (inferred).

## Files

Each run directory holds:

- `config.yaml`: the resolved config
- `run.log`
- `history.jsonl`: train reports every 10 steps, plus the validation reports
- `result.json`: summary, energy, and latency
- `mactop.raw.json` and `prometheus.jsonl`: raw telemetry
- `figures/`

Adapter and checkpoint weights (`*.safetensors`) are git-ignored, so `checkpoints/` holds only `state.json`. The committed runs cannot be resumed from the repository.

## Open items

- Ask the baseline owner for the GPU core count and for whether other jobs ran during the baseline's first epoch.
- Repeat the baseline's power measurement on its Mac, ideally as a back-to-back GPU only and GPU plus Neural Engine pair like the one here.
- Find out why validation slows over a GPU plus Neural Engine run but not a GPU-only run.
- The Neural Engine helps only if it takes a larger share of the step or overlaps with the GPU. The current placement does neither.
