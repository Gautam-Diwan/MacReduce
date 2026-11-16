# M1 MLX GPU verification after macOS upgrade

Measured 2026-11-10 on the M1 MacBook Air, 8 GiB RAM, macOS 27.2.1,
arm64 Python 3.12.0. Installed `mlx==0.32.4` and its `mlx-metal==0.32.4`
dependency only in the existing project `.venv`. No model/dataset downloads,
system settings changes, or QLoRA runs occurred.

[gpu-smoke.json](gpu-smoke.json) records an explicitly selected GPU float32
16×16 all-ones matrix product and sum: expected and observed 4096.0. The probe
sets a 256 MiB MLX allocation limit and 32 MiB cache limit. It does not measure
throughput, power, ANE execution, or full-model memory requirements.

```sh
.venv/bin/python evidence/m1-mlx-2026-11-10/probe.py
PYTHONPATH=src MACREDUCE_TEST_MLX=1 .venv/bin/python -m pytest -q tests/test_mlx_optional.py
```

Both commands ran with Metal access outside the restricted sandbox.
[parity-tests.txt](parity-tests.txt) records **2 passed in 3.95s**, with no skips.
The existing tests compare a forward loss and one SGD update with the CPU
reference for default and two-layer small models. Float32, seeds/shapes and
tolerances are defined in [the test file](../../tests/test_mlx_optional.py).
This is not a multi-step stability audit or Qwen3 correctness result.

The [GPU preflight](gpu-preflight.json) and
[hybrid preflight](gpu_ane-preflight.json) now clear the OS and disk checks;
free space was about 28 GiB. GPU QLoRA remains blocked by missing mlx-lm and
mactop. Hybrid QLoRA additionally needs ANEForge. ANE-only Qwen3 is still
unimplemented. These reports supersede the earlier environment observations,
not their historical records.

The [experiment report](../../wiki/experiments/2026-11-10-m1-mlx.md) states the
accepted scope. The probe passes Ruff; wiki lint passes. No production code or
test tolerances changed during this verification.
