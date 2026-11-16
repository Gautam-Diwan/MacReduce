---
title: "Stage single-machine experiments on memory-constrained Macs"
type: implementation
status: proposed
updated: 2026-11-10
sources: []
---

# Stage single-machine experiments on memory-constrained Macs

For sustained user-run baselines, use the [standalone suite runner](baseline-suite.md).
The separate `baseline` safety profile permits full epochs and explicit
checkpointing true/false. Smoke-only caps are not a general hardware limit.
[The full-data GPU baseline](../experiments/2026-11-10-full-gpu-baseline.md) was
launched after fit checks; it has not completed.

**Initial staging scope:** `main.py`, the memory/curve/QLoRA runners, the Qwen3 ANE
suffix, the federated ANE rejection, and a read-only host preflight. No model
weights, accelerator runtime, ANE graph, or training workload ran on this M1.
See the [M1 staging report](../experiments/2026-11-10-m1-preflight.md).

**Measured follow-up:** after the macOS upgrade, MLX GPU execution and both
existing small-model CPU parity tests passed on this M1. QLoRA remains blocked
by missing dependencies; ANE is unverified. [Follow-up report](../experiments/2026-11-10-m1-mlx.md).

**Later measured follow-up:** [training pilots](../experiments/2026-11-10-m1-training.md)
now exist. GPU training completed; hybrid execution completed but failed the
real-Qwen3 gradient gate. A matching report is required by the hybrid smoke
profile before further launches. The user selected a 10 GiB disk reserve for
these smoke runs. The supervisor also stops for critical memory pressure,
<256 MiB available memory, >512 MiB swap growth, or 900 seconds. It never stops
unrelated processes. These checks are best-effort, not device isolation.

The dataset loader now requests and verifies the immutable parquet revision;
sampling RNG is isolated from engine setup and evaluation. Tiny ANE gates do
not replace testing real model activations. Missing or invalid power channels
remain unknown for comparison even when a sampler returns numeric zeros.

**Proposed:** use [machine-specific configs](../../configs/M1_8GB/README.md)
with shared code, fixed microbatch/checkpointing, and bounded step counts before
attempting a full curve. The two smoke profiles differ only in placement and
output directory. Their sampled evaluation is not a full benchmark.

**Implemented:** `main.py --preflight` checks host metadata, pinned package
versions without importing their runtimes, telemetry availability, disk space,
and explicit smoke budgets. Profiles containing `safety` run that gate during
normal launches too, before directory creation or batch selection. Unknown
physical memory blocks admission. The disk/memory reserves are design choices,
not measured fit requirements. Passing admission does not verify execution.

An explicit MLX budget also bounds the per-process wired allowance and cache;
it does not bound ANE compiler allocations or total process memory. Do not
change system memory limits to make a run pass. Save personal work and back it
up before experimental hardware execution.

**Blocked:** the runner has GPU and hybrid `gpu_ane` paths, but no ANE-only
Qwen3 executor. The federated split ANE client still raises. A linear ANE demo
is not a comparable Qwen3 baseline. Preserve these distinctions in reports.
The [ANE correctness and placement gates](ane.md) apply separately on each Mac.

The [M1 README](../../configs/M1_8GB/README.md) has commands and the proposed
transition to a matched full recipe. [Current state](../project/state.md)
distinguishes prior M4 measurements from this staged M1 work.
