---
title: "Evaluation plan and acceptance gates"
type: experiment-plan
status: proposed
updated: 2026-10-25
sources: ["S06", "S07", "S09"]
---

# Evaluation plan and acceptance gates

**Scope note 2026-10-25:** this page describes the training evaluation plan from decisions 0001 and 0002. The serving benchmark for [decision 0003](../decisions/0003-build-target.md) (A-B-B-A runs, noise floor, greedy equivalence, TTFT and decode tok/s) is specified in the decision and the [roadmap](../project/roadmap.md); its measurement and failure-reporting rules below still apply.

The experimental unit is a complete training run, with a fixed seed, data split, model, algorithm, and hardware/network configuration. Per-step measurements from one run are not independent training replicates. Existing loopback results are a correctness smoke test, not a scaling result.

## Minimum staged matrix

| Stage | Comparison | Required result before moving on |
|---|---|---|
| 0: numerical | CPU reference versus MLX, same weights/data | Forward, gradients, update, and short training parity within declared tolerances |
| 1: local reference | One worker; two loopback workers | Loss decreases; K=1 pooled-update oracle holds; retry/resume semantics remain correct |
| 2: physical Wi-Fi | Fastest single Mac; two distinct Macs | Complete run with identified Wi-Fi link, bytes and phase timings; both workers contribute |
| 3: heterogeneity | Equal work; static calibrated work | Fixed data objective and effective batch; memory admission validated |
| 4: policy | Static calibration; candidate A; ablations | Time-to-quality comparison with uncertainty and overhead included |
| 5: accelerator stretch | GPU; validated ANE path | Matched workload/precision and complete boundary cost accounting |

For the initial small study use at least three seeds per algorithm/configuration, alternate run order where practical, and show all outcomes. Choose the quality target on a pilot configuration before comparison trials; retain failures as censored/not-reached outcomes rather than dropping them. Report both time-to-target and loss at fixed tokens. If a target is unattainable for one method, report that and the completed budget; do not invent a finite time-to-target.

## Record before running

Use [the template](../templates/experiment.md). Record commit or source-tree hash; dependency versions; hardware and OS; power mode and charging; thermal state if available; model/config/tokenizer/data hashes; precision; optimizer and schedule; p_i, K, effective batch and micro-batch; transport and compression; network topology; payload sizes; instrument settings; seed and target. Retain a distinction between configured settings and actual measurements.

## Primary plots and tables

- Held-out loss versus elapsed wall time, and separately versus processed tokens.
- Stacked round time: compute, wait, serialization, transfer, reduction, and load, avoiding overlap double-counting.
- Payload bytes versus complete transfer latency; p50 and p95 with sample count.
- Per-worker tokens, data weight, utilization proxy, peak memory, and thermal observations.
- Policy overhead and time spent calibrating, including failed probes.
- Profiling-on versus profiling-off overhead on the same workload.

For energy, specify source and uncertainty. A whole-machine meter measures the whole machine. Package or ANE-related counters are not automatically per-process attribution. Subtracting idle power has uncertainty and needs a matched idle measurement. Missing counters stay null/unknown.

## Failure experiments

Inject an interrupted request, a lost acknowledgement, worker exit, and coordinator restart. Verify update identity, checkpoint integrity, absence of double application, and an explicit run failure or documented recovery. Simulated network faults are useful protocol tests; they do not validate performance on Wi-Fi. Power-loss durability needs a stronger test than killing a process.

## Release gate

Keep raw results and a generated summary together. Have a second person reproduce at least one run if available. Update [project state](../project/state.md) only for the exact capability tested. Publish negative results and limitations alongside improvements. A successful M4 run is not M5 validation, and a small-model result does not establish 70B inference or training feasibility.
