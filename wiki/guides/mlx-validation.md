---
title: "Validate the MLX path on the M4"
type: implementation
status: proposed
updated: 2026-10-23
sources: ["S02", "S15"]
---

# Validate the MLX path on the M4

**Current state:** code exists, but the original sandbox could not initialize Metal. Run this from a normal Mac terminal in the project root. These are commands for existing functionality, not future APIs.

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

Add explicit gradient access to both backend interfaces. Compare norm, max absolute error, and relative error for each parameter on several seeds and sequence lengths. Check causal masking with a changed future token, embedding gradients for repeated token IDs, and all LayerNorm gradients. Confirm finite losses and weights over at least 100 steps. Compare same-backend runs before comparing mixed backends.

Start with float32. Introduce fp16 or mixed precision only as a separate experiment with a matched reference and documented loss scaling. Different arithmetic can change trajectories; bitwise equality is generally not the acceptance criterion across runtimes.

## Verify GPU execution and timing

Use [MLX capture](../sources/S15.md), `MTL_CAPTURE_ENABLED=1`, and a new `--capture` path on a worker. Inspect the saved trace in Xcode. The existing step implementation evaluates loss and updated parameters before returning; retain this synchronization when optimizing so timings include executed work. Capture cold compilation separately from warmed steps.

## Acceptance and fallback

Accept the backend after parity, multi-step stability, and execution attribution pass. If Metal remains inaccessible, keep CPU tests as plumbing evidence and mark GPU results blocked. Do not equate `doctor`'s chip name with verified compute access. Next: [physical Wi-Fi tests](wifi.md) and [profiling](profiling.md).
