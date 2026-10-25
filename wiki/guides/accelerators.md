---
title: "CPU matrix units M5 GPU paths and other resources"
type: implementation
status: proposed
updated: 2026-10-23
sources: ["S13", "S14", "S21", "S22"]
---

# CPU matrix units M5 GPU paths and other resources

Treat each resource as a separate hypothesis about useful work. [M1](../sources/S21.md) already included ANE. [M5 GPU neural accelerators](../sources/S13.md) sit in GPU cores and are not the same unit. [SME research](../sources/S22.md) and [Apple's OS documentation](../sources/S14.md) motivate CPU matrix experiments on M4. A hardware label does not prove an operator used that unit.

| Resource | Implementation plan | Verification |
|---|---|---|
| CPU SIMD/Accelerate/SME | Extend the existing SGEMM probe to training-shaped matrices, transposes, batch dimensions, and several precisions | Numerical reference, warmed and sustained timing, CPU profiling; actual internal dispatch may remain unknown |
| GPU | Establish MLX as the baseline, then inspect expensive kernels before writing custom Metal | Metal trace plus quality and completed-step timing |
| ANE | Follow the staged [backend guide](ane.md) | Gradient checks, activity, complete-step cost including CPU work |
| M5 tensor path | Add a feature-query and benchmark adapter on a borrowed M5; retain an untested state on M4 | API availability, supported shader/tensor configuration, hardware trace and matched fallback |
| Video/media engines | Use VideoToolbox for decoding only if a future dataset has a measured decode bottleneck | End-to-end input rate and CPU savings; not a replacement for Transformer math |
| Unified memory | Measure pressure, swap, bandwidth proxies, and cache effects | Sustained workload results under controlled background load |
| SSD and network | Stream data/checkpoints with bounded buffers | Actual payload rate and stalls, not storage/Wi-Fi marketing numbers |

Do not probe undocumented instructions by executing arbitrary opcodes just because a chip name matches. Use supported feature queries, preserve permission failures, and prefer public Accelerate interfaces for CPU numerical work. The existing matrix microbenchmark only proves a public BLAS call completed correctly.

## M5 work that can be prepared on M4

Define the capability schema, unavailable-path tests, shape/precision benchmark matrix, and fallback-selection contract. Keep backend selection explicit. On M5, first reproduce a vendor example at a pinned OS/SDK/runtime, then compare workload-shaped kernels and inspect dispatch. Only after that should the scheduler assign work based on measured benefit. M4 results cannot validate M5 speed or kernel placement.

There is no assumption that GPU, CPU, and ANE throughput is additive. Shared DRAM bandwidth, power, cache residency, and host dispatch can dominate. For concurrency experiments read [the prior-work map](../research/related-work.md) and use [the experimental controls](../research/experiments.md).
