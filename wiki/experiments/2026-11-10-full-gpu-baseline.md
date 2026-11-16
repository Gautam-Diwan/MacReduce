---
title: "Full-data GPU baseline with gradient checkpointing disabled"
type: experiment
status: historical
updated: 2026-11-10
sources: []
---

# Full-data GPU baseline with gradient checkpointing disabled

**Later update:** the user selected a 4,096-record comparison instead. The
full-data job was stopped cleanly after its last update-800 report and before
its first checkpoint. It never completed. Logs were retained; the new run
starts from the base model. [Switch report](2026-11-10-m1-comparison-switch.md).

**User-selected scope:** three full epochs of all 23,767 nonempty training
records, batch 1, gradient checkpointing false. The project model remains
pinned Qwen3-0.6B 4-bit with rank-8 LoRA/scale 2. The 512-token record cap
truncates longer records. This differs from the proposal's 4,096-record,
batch-8, accumulation-2 recipe; it is a local GPU baseline, not a FedAvg run.

**Measured fit:** a 10-step probe used the full split, sampled evaluation,
and checkpointing off, with peak MLX allocation 1,194,827,554 bytes. A separate
maximum-length test completed three updates with peak 2,185,500,166 bytes
(2.035 GiB), slightly above the 2 GiB configured allocator budget. That budget
is not a strict total-process/OS bound. A first maximum-length diagnostic hit
a dataset-wrapper interface error before training; its corrected repeat passed.
The short sampled timings imply roughly 7.38 hours of training for 71,301
steps, excluding full evaluation, checkpoints, thermal effects and runtime
variation. It is an estimate, not a measured completion time.

**Launched, not completed:** the supervised GPU job started at
2026-11-10 04:44:55 America/New_York. Its [configuration, raw fit measurements,
launch record and limitations](../../evidence/m1-full-baseline-2026-11-10/README.md)
are retained separately. Full initial test evaluation completed and initial
validation was in progress at inspection. Memory pressure was normal, system
swap had not grown past the guard, and the laptop was on AC power. No full-run
training quality, final time, or final acceptance result exists yet.

The process runs independently of Codex. A supervisor checks critical memory
pressure, available memory, swap growth, free disk (10 GiB), low unplugged
battery, and a 48-hour deadline. Checkpoints save every 5,000 steps. Telemetry
is sampled every 30 seconds to bound long-run log growth. No system security,
memory or fan settings changed; idle sleep is prevented only during the job.
The old 20-step smoke cap and mandatory checkpointing were implementation
policies, not intrinsic training limits. A separate baseline profile now
admits full epochs and either explicit checkpointing setting.

The hybrid remains blocked by the [real-model correctness failure](2026-11-10-m1-training.md).
A longer run cannot establish useful training for a block that returns its
input and zero adapter gradients. Separate ANE clients are not on this run's
scope. The [standalone runner](../guides/baseline-suite.md) supports later user-run
experiments and compact review without a live assistant. State is updated
only with observed progress; launching is not completion.
