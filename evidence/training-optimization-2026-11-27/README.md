# Training optimization measurements, 2026-11-27

Two offline measurements behind the [training optimization strategy](../../wiki/research/training-optimization-strategy.md). Neither is a convergence run. Nothing here measures perplexity, transport, the Pro, the M1, or ANE placement.

## Host and scope

M4 Air (Apple M4, 24 GB, macOS 26.5), MLX 0.32.4, mlx-lm 0.32.2, `mlx-community/Qwen3-0.6B-4bit` at `73e3e38d…`, rank-8 LoRA on all linear layers, scale 2, AdamW lr 1e-4, clip 1.0, GPU only. Training text is the first 4,096 WikiText-2 training lines.

## Step cost (`steps.json`)

`uv run python tools/bench_local_step.py --steps 40 --warmup 5 --out evidence/training-optimization-2026-11-27/steps.json`

Each configuration ran in a fresh process with 5 warmup and 40 timed steps. Throughput counts unpadded target tokens. "Line" is one WikiText line per step, as the federation backend does today. "Pack512" concatenates lines with EOS separators into 512-token rows. The compiled line run includes recompiles for new padded widths inside the timed region.

| Configuration | Target tok/s | Median step | Tokens/step | Peak MLX memory |
|---|---|---|---|---|
| line, eager (current backend) | 432 | 0.259 s | 136 | 1.83 GB |
| line, compiled | 487 | 0.223 s | 136 | 1.66 GB |
| pack512, eager | 505 | 1.021 s | 512 | 2.46 GB |
| pack512, compiled | 558 | 0.916 s | 512 | 2.26 GB |
| pack512 x4, compiled | 509 | 4.140 s | 2,048 | 6.91 GB |
| line, eager, grad checkpoint | 314 | 0.355 s | 136 | 1.00 GB |
| pack512, eager, grad checkpoint | 356 | 1.430 s | 512 | 1.28 GB |

Measured: packing plus `mx.compile` raised throughput 29% over the current step, and batch 4 did not raise it further on this GPU. Gradient checkpointing cost 27–29% throughput. Peak MLX memory was at most 2.46 GB without it at batch 1.

Measured token statistics for the 4,096 lines (Qwen3 tokenizer, EOS appended): 412,181 target tokens per epoch; per-line median 76, mean 100.6, 10th percentile 6, 90th percentile 241; 37.9% of lines have fewer than 32 target tokens; 0.2% are truncated at 512. Per client over three epochs with three IID shards, that is 4,096 line steps or about 805 packed steps.

## Delta codecs (`codecs.json`)

`uv run python tools/delta_codecs.py runs/federation-gpu --out evidence/training-optimization-2026-11-27/codecs.json`

Inputs: the M4 Air client's three `delta.npz` files and the coordinator's four adapters from the [2026-11-24 run](../tailscale-federation-2026-11-24/README.md). Those are two-step deltas from early training; later, longer-interval deltas may differ. Affine codecs use group 64 along the long axis, matching `mx.quantize`. Wire size includes one scale and one bias per group.

| Codec | Wire per client per direction | Air delta relative L2 error (rounds 0, 1, 2) |
|---|---|---|
| fp32 (current) | 19.25 MiB | 0 |
| fp16 | 9.63 MiB | 0.0002 each |
| bf16 | 9.63 MiB | 0.0016–0.0017 |
| int8 g64 | 5.11 MiB | 0.0029, 0.0036, 0.0041 |
| int4 g64 | 2.71 MiB | 0.047, 0.062, 0.070 |
| int2 g64 | 1.50 MiB | 0.21, 0.31, 0.36 |

Measured layout: the top 1% of entries hold 2.8–4.1% of client delta energy, and the top 10% hold 25–31%. That is dense, so top-k sparsification would discard much of the update. Delta energy split A/B was 19/81 in round 0 and about 47/53 and 49/51 afterwards. Each of the 28 layers carries between 2.2% and 5.3% of merged delta energy. MLP projections (gate, up, down) carry about 50–56%.

Simulated error feedback over the three Air rounds lowered cumulative relative error from 0.0025 to 0.0017 (int8), 0.042 to 0.028 (int4), and 0.21 to 0.15 (int2).

An ad hoc check, not saved as JSON, ran `mx.quantize` on round 1 on the GPU. It gave relative error 0.0037 (int8), 0.064 (int4), and 0.103 (MX-FP4). Encoding all 392 tensors took 4–6 ms.

Code inspection of the installed MLX 0.32.4: `optim.AdamW` defaults to `bias_correction=False`. The first two uncorrected updates are about 3.16 and 4.25 times lr for a steady gradient. That predicts the measured round-0 maximum of 7.42e-4 at lr 1e-4.
