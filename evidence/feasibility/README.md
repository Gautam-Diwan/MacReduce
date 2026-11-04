# One-Mac feasibility probe

Measured 2026-10-28 on this M4 MacBook Air. Command:

`macreduce feasibility --output evidence/feasibility --data data/wikitext-2-raw-v1 --iters 4 --batch-size 2 --seq 256 --train-examples 64 --eval-batches 2`

`result.json` is the summary. `gpu/`, `ane/`, and `gpu-ane/` hold mactop's headless JSON and the Prometheus scrapes from port 9096. The SVG files are the power traces, the energy bars, and the GPU probe perplexity.

The GPU client is 4 steps of Qwen3-0.6B QLoRA. The Neural Engine client is an ANEForge linear model plus a 256-wide matmul. They are not the same adapter. MLX was limited to 8 GB and peaked near 1.3 GB. The report is [the experiment page](../../wiki/experiments/2026-10-28-one-mac-feasibility.md).
