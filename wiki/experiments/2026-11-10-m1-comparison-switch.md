---
title: "Switch to the provisional 4,096-record M4 comparison"
type: experiment
status: active
updated: 2026-11-10
sources: []
---

# Switch to the provisional 4,096-record M4 comparison

**Later observation:** the job stopped at update 3,131 of 12,288 after 3,015.81
seconds because total system swap increased beyond the 512 MiB supervisor
policy. Memory pressure was Warning, not Critical; no out-of-memory exception
was observed. There was no saved checkpoint before the first epoch boundary,
so a fresh restart is required. [Raw stop record and restart instructions](../../evidence/m1-comparison-stop-2026-11-10/README.md).
The guard is now configurable, retaining its 512 MiB default; the comparison
config saves state every 500 updates for future attempts. No restart has been
launched and no threshold increase was silently applied. The launch account
below is historical progress, not a completed training result.

The user explicitly selected switching away from the full-data job after
reporting that the M4 runs likely use about 4K records, batch 1, and gradient
checkpointing off. The actual M4 saved configuration has not been inspected.
This is a provisional comparison, not a confirmed hardware-only match.

**Measured control action:** verified and signaled only the full-data
supervisor. It terminated its owned trainer and recorded `interrupted` after
1,340.24 seconds. Last periodic training report: update 800; no checkpoint had
yet been written. Existing logs/artifacts remain unchanged. No unsaved adapter
was resumed. The [raw stop/launch records](../../evidence/m1-comparison-switch-2026-11-10/README.md)
state inspected scope and commands. The [earlier full-run report](2026-11-10-full-gpu-baseline.md)
is now historical, not a completed result.

**Replacement launched, not completed:** started 05:08:41 America/New_York,
first 4,096 nonempty WikiText records, three epochs, batch 1, accumulation 1,
checkpointing false, sequence cap 512, pinned Qwen3-0.6B 4-bit, LoRA rank 8 and
scale 2, seed 0. Planned updates: 12,288. AdamW/schedule, clipping, and weight
decay follow the shared recipe, with schedule length derived from this run.
Full validation/test splits remain selected. Checkpoints save every epoch.
Initial evaluation was progressing with normal pressure and AC power at
inspection. Existing resource guards remain; there is no final score or timing.

The [standalone suite](../guides/baseline-suite.md) now defaults to `comparison`.
`full` preserves the earlier 23,767-record adaptation; `proposal` preserves the
written batch-8/accumulation-2 recipe and explicit admission/coverage gaps.
No ANE/federated case is relabeled as measured. For a valid M4 comparison,
verify data fingerprints, accumulation, optimizer, schedule, seed, precision,
software versions, placement, and evaluation details against saved M4 configs.

Software validation: 121 passed, 5 skipped; project-code Ruff, wiki lint and
whitespace checks passed. These are software checks, not experiment completion.
[State](../project/state.md), [machine configs](../../configs/M1_8GB/README.md).
