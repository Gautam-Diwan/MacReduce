---
title: "Validate the MLX path on the M4"
type: implementation
status: active
updated: 2026-10-24
sources: ["S02", "S15"]
---

# Validate the MLX path on the M4

**Measured on 2026-10-24** from this project on the M4: `doctor --smoke` initialized Metal and a 16×16 matmul returned 4096.0. The opt-in test passed for the default model and for a 2-layer config (forward loss and one SGD step against CPU, existing tolerances). A loopback `mlx-gpu` demo of 3 rounds and 4 steps reduced validation loss from 6.058713912963867 to 3.713829755783081. Details and versions are in [the evidence record](../../evidence/mlx-validation/README.md). `xctrace` was not available, so there is no saved GPU trace. The gradient audit and 100-step check below are still open. An unavailable device remains a failure, not a pass.

These commands exercise the current trainer:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[cpu,mlx,dev]'
macreduce doctor --smoke --output runs/m4-hardware.json
MACREDUCE_TEST_MLX=1 pytest -q tests/test_mlx_optional.py
macreduce demo --backend mlx-gpu --output runs/m4-mlx
```

The opt-in test compares forward loss and one SGD update against CPU using the same arrays. A skipped test is not a pass. Record OS version, chip, Python and MLX versions, dtype, seed, and test tolerances. The existing test uses small absolute and relative tolerances; if it fails, inspect the first mismatching tensor and operator rather than merely increasing tolerances.

## Extend the numerical gate

This section is not done. Add explicit gradient access to both backend interfaces. Compare norm, max absolute error, and relative error for each parameter on several seeds and sequence lengths. Check causal masking with a changed future token, embedding gradients for repeated token IDs, and all LayerNorm gradients. Confirm finite losses and weights over at least 100 steps. Compare same-backend runs before comparing mixed backends. The 2026-10-24 check covers one forward and one update on two shapes, not this audit.

Start with float32. Introduce fp16 or mixed precision only as a separate experiment with a matched reference and documented loss scaling. Different arithmetic can change trajectories; bitwise equality is generally not the acceptance criterion across runtimes.

## Verify GPU execution and timing

Use [MLX capture](../sources/S15.md), `MTL_CAPTURE_ENABLED=1`, and the existing `--capture` path on a worker. Inspect the saved trace in Xcode. On 2026-10-24, `xcrun` could not find `xctrace`, so this step is blocked on this machine until the developer tools are present. The existing step implementation evaluates loss and updated parameters before returning; retain this synchronization when optimizing so timings include executed work. Capture cold compilation separately from warmed steps.

## Acceptance and fallback

Accept the backend after parity, multi-step stability, and execution attribution pass. If Metal remains inaccessible, keep CPU tests as plumbing evidence and mark GPU results blocked. Do not equate `doctor`'s chip name with verified compute access. Next: [physical Wi-Fi tests](wifi.md) and [profiling](profiling.md).
