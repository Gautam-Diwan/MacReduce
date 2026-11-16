---
title: "M1 8GB read-only experiment preflight"
type: experiment
status: measured
updated: 2026-11-10
sources: []
---

# M1 8GB read-only experiment preflight

**Measured scope:** host and package metadata only. Machine: Apple M1,
MacBookAir10,1, 8 GiB physical memory, arm64, macOS 13.4.1, Python 3.12.0.
Free data-volume space was about 9.8 GiB at inspection. No model/data downloads,
GPU or ANE execution, training, or network transport were measured.

The [raw reports and commands](../../evidence/m1-preflight-2026-11-10/README.md)
record the GPU and GPU+ANE admission checks. Both returned `blocked`: OS below
the preliminary gate, missing pinned packages, missing mactop, and disk below
the proposed 15 GiB reserve. Memory detection recovered through selected
`system_profiler` fields after the sandbox denied `sysctl hw.memsize`; no machine
serial or account identifier was retained.

**Proposed:** the [M1 profiles](../../configs/M1_8GB/README.md) bound smoke runs
to 10 steps, batch 1, sequence 128, checkpointing, and 2 GiB of MLX allocations.
They have not established fit, numerical correctness, ANE residency, throughput,
power, full-split perplexity, or a completed course result. ANE-only Qwen3
training remains unimplemented. These are unavailable results, not zero values.

The work branch was created from local `main` at `d67f2e3`; remote freshness was
not checked. Existing source snapshots and historical evidence were untouched.
**Validation:** [recorded checks](../../evidence/m1-preflight-2026-11-10/validation.txt)
show 109 tests passed and 5 hardware tests skipped in a CPU-capable environment;
this is not accelerator validation. Code/test Ruff passed; whole-tree Ruff
reported 13 pre-existing immutable-snapshot violations. The inherited SciPy
and NumPy versions emitted one compatibility warning. These checks do not
establish performance or dataset quality.

See the [single-machine guide](../guides/single-machine.md) for admission scope
and the [project state](../project/state.md) for prior measurements.
