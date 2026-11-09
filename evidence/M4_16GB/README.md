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
