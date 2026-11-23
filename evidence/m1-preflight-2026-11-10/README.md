# M1 admission reports, 2026-11-10

Scope: read-only environment checks, **no training or accelerator execution**.
The commands and exit codes are included in [gpu.json](gpu.json) and
[gpu_ane.json](gpu_ane.json). Reports intentionally retain only OS, architecture,
Python, RAM, disk space, package versions, and telemetry availability.

Both arms returned exit 1 (`blocked`). Free space is a point-in-time observation,
not a storage benchmark. The 15 GiB disk reserve and 4 GiB physical-memory
reserve are proposed policies. The 2 GiB MLX budget does not bound ANE compiler
or total process memory. No throughput, power, or perplexity was measured.

Reproduce from the root:

```sh
PYTHONPATH=src .venv/bin/python main.py --config configs/M1_8GB/gpu_smoke.yaml --preflight
PYTHONPATH=src .venv/bin/python main.py --config configs/M1_8GB/gpu_ane_smoke.yaml --preflight
```

The local validation environment uses Python 3.12.0 in a project `.venv` with
system-site packages visible; pytest, Ruff, psutil and seaborn were added only
inside that environment. It is not the pinned accelerator environment from
`uv.lock` and cannot establish hardware compatibility. No MLX packages or model
weights were installed. See the [experiment report](../../wiki/experiments/2026-11-10-m1-preflight.md).

[Validation results](validation.txt): 109 tests passed, 5 hardware tests skipped.
Code/test Ruff checks passed; full-tree Ruff reported 13 existing issues in
immutable source snapshots. Hardware skips are not hardware passes.
