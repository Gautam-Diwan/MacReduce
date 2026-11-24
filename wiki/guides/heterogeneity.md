---
title: "Schedule unequal Macs without hiding objective changes"
type: implementation
status: proposed
updated: 2026-11-12
sources: ["S06", "S07", "S08"]
---

# Schedule unequal Macs without hiding objective changes

**Planning update 2026-11-12:** the [three-Mac study](../research/three-mac-wireless-federation.md) applies this objective/throughput distinction to an M4 Air 24 GB, M4 Pro 16 GB, and M1 Air 8 GB. Its revised main arm calibrates different microbatch/accumulation settings and proportional random shards to balance round duration, with common K, a fixed total data objective, matched controls and separate two-M4 versus three-Mac measurements. Uniform batch-1 work is retained as a control only. M1 fit/speed is unverified.

The current system permits different worker batch sizes but fixes K and learning rate for the run. It weights parameters by processed tokens. Increasing a worker's contribution can also increase the influence of its local domain. Decide the desired data distribution before optimizing scheduling.

## Stage one preserves the simplest semantics

Define per-worker dataset weight p_i explicitly, with weights summing to one. Keep a common K initially, but calibrate microbatch sizes and accumulation counts for each device. Equal effective batches are a useful control, not a requirement for the main throughput arm. For the shared-public-corpus study, larger effective batches can be paired with proportionally larger random shards, preserving the full data union while balancing update counts per epoch. Accumulate/reduce mean gradients using actual supervised-token counts and document objective weights independently of hardware throughput. Different effective batches change optimization noise, so include matched centralized and uniform controls.

Implement a `WorkerProfile` with measured warmed step-time quantiles, available memory, backend, dtype, and calibration timestamp. Add a `RoundPlan` containing run ID, version, expected workers, effective batch, microbatch/accumulation settings, K, optimizer configuration, dataset weights, and a plan hash. Workers echo the plan hash; the coordinator rejects mismatched plans. Persist plans with checkpoints.

Calibrate several safe batch sizes before training, with repeated measurements. Use p95 completion time when predicting a barrier. Reject configurations causing swap or persistent memory pressure. Add hysteresis and a cooldown so calibration itself does not dominate training.

## Stage two allows different local horizons

Unequal K can bias naive aggregation. [FedNova](../sources/S06.md) addresses objective inconsistency; [HEW-Local SGD](../sources/S08.md) is further relevant theory. Read full papers and their assumptions before claiming a correction for Transformer training.

For an initial plain-SGD experiment with common step size, represent a worker delta as its final weights minus the common starting weights. A FedNova-style special case rescales each delta by its local step count, combines using the intended dataset weights, and applies an explicitly chosen effective horizon. This is not a generic rule for Adam, momentum, arbitrary participation, or biased minibatch sampling. Implement the published solver-specific form, not an improvised token-count normalization.

## Tests and baselines

- With K=1 and equal objective weights proportional to batch tokens, retain pooled-SGD equivalence.
- Use a two-client quadratic with distinct optima and known weighted optimum to expose objective drift when K differs.
- Test microbatch accumulation against the same effective full batch.
- On a real corpus, compare equal settings, static calibrated batches, fixed-objective accumulation, and any adaptive policy. Report domain-level losses as well as aggregate loss.
- Add an artificially slowed worker and a thermally warmed machine as separately labeled experiments. Do not label injected delay as physical hardware evidence.

[Adaptive communication already has prior art](../sources/S07.md). The research question is whether a defined policy handles measured Wi-Fi and thermal variation better under a fixed quality objective. See [candidate A](../research/novelty.md) and [experiment design](../research/experiments.md).
