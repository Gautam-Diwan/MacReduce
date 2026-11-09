# GPU plus Neural Engine QLoRA on an M4 MacBook Pro, 16 GB

These runs repeat the GPU plus Neural Engine baseline on a second Mac model. The goal is to compare Mac models under the same GPU plus Neural Engine setup. The baseline is [evidence/qwen3-0.6b-curve/2026-10-29T00-07-18](../qwen3-0.6b-curve/2026-10-29T00-07-18/result.json), run by Gautam Diwan.

Labels: **measured** means read from the files in this folder. **Source-reported** means taken from the project wiki or handoff and not re-checked here. **Inferred** means a reasoned interpretation. **Unknown** means not established.

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

## Data preparation

The first launch failed before training with `HTTP Error 502: Bad Gateway` from the Hugging Face rows API, which `prepare_dataset` calls about 420 times with no retry. That failed run directory was removed. The files in `data/wikitext-2-raw-v1` were then written by the repository's own `prepare_dataset`, called from a one-off script that retried transient HTTP errors. No repository code changed. The record counts match the baseline manifest: train 23,767, validation 2,461, test 2,891. The Hub's current revision of `Salesforce/wikitext` is the pinned `b08601e04326c79dfdd32d625aee71d232d685c3`.

## Smoke run: first 500 steps of the baseline

Config: [configs/M4_16GB/smoke_gpu_ane.yaml](../../configs/M4_16GB/smoke_gpu_ane.yaml). Run: [smoke_gpu_ane/2026-11-09T00-17-05](smoke_gpu_ane/2026-11-09T00-17-05/result.json).

```sh
uv run python main.py --config configs/M4_16GB/smoke_gpu_ane.yaml
```

The config is the baseline config with three changes. `iters: 500` stops early. `epochs: 1` places validation at steps 1 and 500 and one checkpoint at step 500. `lr_schedule.arguments: [0.0001, 11988, 1.0e-6]` pins the cosine to the baseline's resolved schedule, so the learning rate at every step equals the baseline's. Seed, data, batch size 1, sequence 512, LoRA rank 8 and scale 2, and full validation and test splits are unchanged.

Conditions: plugged in, Low Power Mode off, no `caffeinate`. Google Chrome played a YouTube live stream for the whole run (`pmset` shows its video wake lock from 00:18:14 to 01:13:32).

| Measured | This Mac | Baseline, same steps |
|---|---|---|
| Status | ok | ok |
| Neural Engine | Blocks 0–26 on the GPU, block 27 on the Neural Engine at mask `0x4`; 5,422 forward and 500 backward calls | Same placement |
| Untrained validation loss (step 1) | 4.1913 | 4.1913 |
| Untrained test perplexity | 64.2367 | 64.2368 |
| Train loss at 50 reports, steps 10–500 | Equal to the baseline at every report | — |
| Learning rate at 50 reports | Equal to the baseline at every report | — |
| Validation loss at step 500 | 3.1767 (perplexity 23.97) | No validation at step 500 |
| Test perplexity after 500 steps | 23.088 | — |
| MLX peak memory | 4,260,406,834 bytes | 4,260,406,834 bytes (full run) |
| Median step time | 234 ms (431 tokens/s) | 541 ms over the same steps |
| Full validation pass | 199 s and 204 s | 368 s at step 1 |
| Wall time | 945 s | — |

The smoke run reproduced the baseline's training trajectory on this Mac. The speed comparison against the baseline's first 500 steps is not a clean machine comparison; see the full run below.

## Full run: three epochs

Config: [configs/M4_16GB/qlora_gpu_ane.yaml](../../configs/M4_16GB/qlora_gpu_ane.yaml). Run: [gpu_ane/2026-11-09T01-14-02](gpu_ane/2026-11-09T01-14-02/result.json).

```sh
caffeinate -i uv run python main.py --config configs/M4_16GB/qlora_gpu_ane.yaml
```

The config is [configs/qlora_m4.yaml](../../configs/qlora_m4.yaml) with `batch_size: 1` and `grad_checkpoint: false`, the values the baseline's batch search chose, and its own `logging.dir`. Every other field equals the baseline's saved [config.yaml](../qwen3-0.6b-curve/2026-10-29T00-07-18/config.yaml) apart from the resolved memory limit.

Conditions: plugged in, Low Power Mode off. `caffeinate` held its idle-sleep assertion from 01:14:02 to 02:27:40, and `pmset` logged no sleep. Chrome played a YouTube live stream from 01:23:25, 9.3 minutes after launch, to the end. During the evaluation-only phase at the start, the GPU rail median was 10.3 W before the stream and 9.6 W over the equal-length evaluation phase at the end with it playing, so the stream's draw is not separable here (measured; interpretation inferred).

### Training result: identical to the baseline

| Measured | This Mac | Baseline |
|---|---|---|
| Status | ok | ok |
| Steps | 12,288 (3 epochs, batch 1) | 12,288 |
| Neural Engine calls | 22,132 forward, 12,288 backward, mask `0x4` | Same |
| Train loss, 1,229 reports | Equal at every report | — |
| Validation perplexity at steps 0, 4095, 8191, 12287 | 66.111, 21.321, 22.085, 23.723 | Same values |
| Test perplexity, untrained then adapted | 64.236 then 23.2974 | 64.237 then 23.2974 |
| Adapted test loss | 3.148341417312622 | 3.148341417312622 |
| MLX peak memory | 4,260,406,834 bytes | 4,260,406,834 bytes |

The two Macs produced the same training trajectory and the same final adapter quality to the logged precision, across different macOS versions and memory sizes. Validation perplexity is lowest after epoch 1 and rises after it in both runs.

### Speed

| Measured | This Mac | Baseline |
|---|---|---|
| Wall time | 4,411 s (73.5 min) | 13,752 s (3 h 49 min) |
| Median step time, epochs 1 / 2 / 3 | 228 / 227 / 229 ms | 502 / 295 / 288 ms |
| Median tokens/s | 441 | 319 |
| Reports below 1 step/s | 12 of 1,229 | 154 of 1,229 |
| Full validation pass, steps 0 / 4095 / 8191 / 12287 | 195 / 201 / 215 / 239 s | 368 / 409 / 413 / 434 s |

The 3.1x wall-time gap mixes machine speed with the baseline's slow first epoch, which has many sub-1 step/s reports. The baseline handoff records concurrent experiments on that machine the same night (source-reported), so its first epoch is not a clean measurement. The steady epochs 2 and 3 give a ratio of about 1.26 to 1.30 per training step. Validation passes are about 1.8 to 2.0 times faster on this Mac. Why validation gains more than training is **unknown**. On this Mac, validation time rose from 195 s to 239 s across the run while training step time stayed flat; the cause is **unknown**.

Candidate causes of the per-step gap, none tested: GPU core count (unknown on the baseline Mac), fan versus fanless cooling under sustained load, background load, and macOS version.

### Power and energy

mactop samples are system-wide, not per process.

| Measured | This Mac | Baseline |
|---|---|---|
| mactop coverage | 4,374 s of the 4,375 s run, no gap over 5 s, exit code 0 | 8,595 s of the 13,721 s run, a 359 s gap near the start, exit code -9 |
| GPU rail, average | 8.32 W (36.4 kJ) | 1.40 W (12.0 kJ) |
| GPU rail, median sample | 8.95 W | 0.23 W |
| Neural Engine rail, average | 0.083 W (364 J) | 0.011 W (98 J) |
| CPU rail, average | 2.20 W (9.6 kJ) | 3.88 W (33.4 kJ) |
| DRAM rail, average | 1.28 W (5.6 kJ) | 1.26 W (10.8 kJ) |

The baseline's power record cannot support a cross-machine energy comparison. It stops about 85 minutes before training ended. After its first tenth, its GPU median is about 0.2 W while the run kept training at about 290 ms per step, which a GPU-resident training step should not allow (inferred). The cause is **unknown**. A cross-machine energy comparison needs a new baseline measurement.

## Files

Each run directory holds `config.yaml` (the resolved config), `run.log`, `history.jsonl` (train reports every 10 steps and validation reports), `result.json` (summary, energy, latency), `mactop.raw.json` and `prometheus.jsonl` (raw telemetry), and `figures/`. Adapter and checkpoint weights (`*.safetensors`) are git-ignored, so `checkpoints/` holds only `state.json` and the committed runs cannot be resumed from the repository.

## Open items

- Ask the baseline owner for the GPU core count and for whether other jobs ran during the baseline's first epoch.
- Repeat the baseline's power measurement on its Mac.
- Run the GPU-only control on this Mac with [configs/M4_16GB/qlora_gpu.yaml](../../configs/M4_16GB/qlora_gpu.yaml).
