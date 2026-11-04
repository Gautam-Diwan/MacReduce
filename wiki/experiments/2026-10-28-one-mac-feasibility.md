---
title: "One-Mac feasibility probe"
type: experiment
status: measured
updated: 2026-10-28
sources: ["S37"]
---

# One-Mac feasibility probe

## Question

Can this M4 MacBook Air run the three one-Mac client modes inside an 8 GB MLX memory budget, and can mactop separate GPU energy from Neural Engine energy while it does?

The modes are one GPU client, one combined GPU and Neural Engine client, and those two devices as separate clients. The proposal metric is validation perplexity and adapter bytes. This run is a short probe, not the three-epoch curve.

## Reproduction

Command, from the project root, after the seeded rerun:

`macreduce feasibility --output evidence/feasibility --data data/wikitext-2-raw-v1 --iters 4 --batch-size 2 --seq 256 --train-examples 64 --eval-batches 2 --prometheus-port 9096`

GPU client: QLoRA rank 8, AdamW, learning rate 1e-4, Qwen3-0.6B-4bit, first 64 WikiText-2 training lines, sequence 256, 4 optimizer steps, 2 validation batches and 2 test batches. MLX memory limit 8,589,934,592 bytes. Seed 0 was set on the MLX RNG before each GPU load. The two GPU loads still logged different longest training sentences (260 and 274), so their perplexities are not a paired comparison.

Neural Engine client: `import aneforge as af`, four Adam steps of a 16-by-1 linear map, then a 256-wide matmul on device mask `0x4`. That graph is not Qwen3. The solo engine window was 8 seconds. The combined arm kept the matmul running until the GPU arm returned.

mactop  headless JSON and Prometheus on port 9096, interval 500 ms. Energy is a trapezoid of system `soc_metrics` power. A second trapezoid uses `mactop_power_watts`. Neither is per process. `ane_active` was 0 in every sample.

## Raw artifacts

[evidence/feasibility](../../evidence/feasibility/README.md). `result.json` holds the summary and the power series. Each arm directory holds `mactop.raw.json` and `prometheus.jsonl`. Plots: `power-gpu.svg`, `power-ane.svg`, `power-gpu_ane.svg`, `energy.svg`, `perplexity.svg`.

The headless series is short: 5, 10, and 3 samples. Prometheus scraped 9, 14, and 5 times. The two integrals disagree by roughly 6 J on the GPU-only rail (41 J headless, 35 J Prometheus). Treat the joules as a trace total, not a millijoule measurement. Average watts in `result.json` divide by wall-clock time, which includes time outside the sample span, so the sample readings are the better power figure.

## Results

Measured on this machine, 2026-10-28.

| Case | Wall time | GPU energy, headless | ANE energy, headless | Peak GPU | Peak ANE | Peak ANE bandwidth |
|---|---:|---:|---:|---:|---:|---:|
| 1 GPU | 6.02 s | 41.1 J | 0 J | 15.9 W | 0 W | 0 |
| 1 GPU + ANE, concurrent | 4.72 s | 20.1 J | 0.96 J | 15.9 W | 0.73 W | 1.99 GB/s |
| ANE client alone | 9.50 s | 0.72 J | 4.23 J | 0.24 W | 0.76 W | 3.28 GB/s |
| 1 GPU and 1 ANE, summed, not overlapped | 15.5 s | 41.9 J | 4.23 J | the two solo peaks | the two solo peaks | the solo peak |

One combined sample shows both rails at once: GPU 15.9 W and ANE 0.73 W, bandwidth 1.99 GB/s. MLX peak memory was 1.22 GB on the GPU arm and 1.27 GB on the combined arm.

The GPU-only probe, 2 batches, is not a stable quality number. Test perplexity moved from 78.6 to 48.1. Validation perplexity stayed near 49.1 to 49.8. The adapter delta is 20,185,088 bytes in fp32 and 5,046,280 bytes in int8 with one scale. One Mac sends 0 wire bytes.

The engine linear losses were 0.308, 0.0483, 0.0567, 0.361, and the weight moved by 0.184. The solo matmul ran 122,833 calls. The combined matmul ran 33,081 calls.

## Interpretation

The GPU QLoRA client fits in the 8 GB budget on this Mac. The Neural Engine can be busy at the same time, and mactop's power and bandwidth rails separate it from the GPU. `ane_active` does not.

The combined and split cases do not yet share a Qwen3 adapter. Their Neural Engine work is a different model, so there is no joint perplexity and no averaged delta. The 4-step probe is also too short to be the proposal baseline. The three-epoch curve on 4,096 lines is still unrun. A second Mac was not involved.

S37 says engine training is a capability and that its own silicon numbers are M1 and M5. This probe does not confirm those roofs. It confirms a small ANEForge graph and a short MLX QLoRA step on this M4.
