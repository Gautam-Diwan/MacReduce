---
title: "Research contribution candidates and falsification plans"
type: research
status: proposed
updated: 2026-10-23
sources: ["S03", "S05", "S06", "S07", "S08", "S09", "S10", "S12"]
---

# Research contribution candidates and falsification plans

**Recommendation:** build a credible physical Wi-Fi baseline first, then test candidate A. It fits the existing trainer and can produce an informative result without depending on private ANE APIs. Candidate B is more hardware-specific and higher risk. None is a verified novelty claim.

## A. Objective-preserving scheduling under changing Wi-Fi and thermal conditions

**Question:** can a small Mac cluster reach a fixed validation target sooner by adapting work to measured compute and communication conditions, while retaining fixed data-source weights?

**Hypothesis:** a policy that accounts for uncertainty, calibration cost, and slow changes in compute/network conditions improves time-to-quality over a static calibrated policy. The useful result could also be that policy overhead or optimization drift eliminates the gain.

**Potential distinction to investigate:** not generic adaptive synchronization ([AdaComm](../sources/S07.md)), heterogeneous local work ([FedNova](../sources/S06.md), [HEW-LocalSGD](../sources/S08.md)), or Wi-Fi training ([AirTrain](../sources/S03.md)). The narrower candidate is measured control on small, shared-memory Macs where thermal drift, shared wireless airtime, and intrusive probes interact, with an explicitly preserved data objective. Full-paper comparison may show that this is an application or evaluation contribution rather than a new algorithm.

**Build in stages:**

1. Fix dataset mixture weights p_i independently of device speed. Start with equal K and equal effective batch, allowing device-specific micro-batches and accumulation. Record p_i in the run manifest.
2. Estimate worker step time, peak memory, transfer tails, and calibration overhead. Keep confidence intervals and sample age. Only measure safe candidate configurations; avoid probes near memory exhaustion.
3. Compare static calibrated allocation with a simple bounded feedback rule. Change only micro-batch/accumulation first. Introduce K adaptation later as a separate algorithm experiment. Specify update frequency, hysteresis, bounds, and fallback before trials.
4. For unequal local steps, implement and verify a solver-appropriate normalization baseline. Do not assume token weighting preserves p_i. Compare on a quadratic oracle before a language model. See [heterogeneity](../guides/heterogeneity.md).
5. Evaluate benign physical Wi-Fi changes and thermal drift; separately use repeatable injected delay/bandwidth scenarios. Do not combine those into a claim about measured radio behavior.

**Baselines:** single fastest Mac; equal work; static calibration; the same policy without uncertainty or recalibration cost; fixed K sweep; an appropriate adaptive-communication baseline after full-paper review. Use an offline oracle only as an explicitly retrospective bound, not a deployable baseline.

**Primary outcome:** wall time to a predeclared held-out loss target, reporting failures to reach it. Secondary outcomes: validation loss at fixed tokens, per-domain loss, total bytes, tail round time, idle time, energy when measured credibly, and probe overhead. At least three seeds for the small study, with variation shown; this is not a universal statistical power guarantee.

**Ablations:** stationary versus changing conditions; one versus multiple concurrent transfers; uniform versus skewed data; without uncertainty; without calibration cost; fixed versus adapted K; equal versus unequal hardware. If the gain disappears against static calibration or comes from changing the data mixture, the central hypothesis fails.

**Resources:** two or preferably three physical Macs; one can be the available M4. An M5 is not required. First validate a configurable 1–10M parameter model; scale only after the transfer protocol and memory accounting support it. Publish scripts, configuration, and anonymized measurements.

## B. ANE-resident local training blocks versus synchronization boundaries

**Question:** at what model size, local-step count, and synchronization frequency does keeping computation/state on the ANE pay off after host materialization, exchange, and recompilation costs?

**Potential distinction:** distributed synchronization-boundary cost accounting for an ANE training backend, not inventing ANE training. [ANEForge](../sources/S10.md), [maderix/ANE](../sources/S11.md), and [Orion](../sources/S12.md) already occupy the latter space. The available APIs are private and their compatibility needs independent assessment.

First pass [ANE correctness gates](../guides/ane.md). Hold architecture, precision, effective batch, and algorithm fixed across GPU and ANE. Compare local-only, loopback synchronization, and physical Wi-Fi synchronization. Separate compile, warmup, compute, host fallback, materialization, serialize, transfer, reduce, and reload costs. Sweep K and payload size; report quality at fixed tokens as K changes. Do not call a host-optimizer configuration “all ANE.”

**Falsification:** numerical error, insufficient operator coverage, or materialization overhead consumes the local gain at every feasible setting. That is still a useful measured result if the implementation is validated and bottlenecks are attributed. This is a stretch scope after the baseline, not a dependency of the main project.

## C. Calibration that does not damage the workload it is measuring

**Question:** can a lightweight admission/calibration method choose configurations more reliably than aggressive probing under shared-memory pressure?

[FusionML](../sources/S09.md) already discusses calibration effects and CPU/GPU execution, so the broad idea is not new. Investigate a cluster policy with bounded probe memory/time, confidence-based remeasurement, and thermal-aware sample expiry. Compare prediction error, time-to-quality, memory pressure, failed admissions, and total probe cost with static and exhaustive calibration. Never cause system-wide memory exhaustion merely to collect a benchmark. Novelty requires a closer search of online autotuning and systems calibration literature.

## D. A reproducible Mac-cluster training and profiling benchmark

A carefully documented benchmark can be a strong course contribution even without a novel optimizer. Combine per-phase traces, model/dataset/config hashes, honest missing counters, physical Wi-Fi trials, recovery scenarios, and paired profiling-on/off measurements. Focus on questions existing papers do not measure consistently: communication fraction, probe perturbation, energy attribution uncertainty, and cross-machine heterogeneity.

This is an evaluation/tooling contribution. It becomes research evidence only with real multi-Mac measurements, public reproduction instructions, and explicit limitations; the current loopback demo is insufficient.

## What not to claim

Do not claim first Mac cluster, first Wi-Fi local SGD, first ANE training, first adaptive synchronization, or automatic utilization of every accelerator. Do not call a scheduling gain novel until the closest papers are read in full and the claimed distinction survives comparison. Record the final claim and evidence in a [decision](../templates/decision.md) after [experiments](experiments.md).
