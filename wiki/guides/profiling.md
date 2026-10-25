---
title: "Make profiling useful for scientific claims"
type: implementation
status: proposed
updated: 2026-10-23
sources: ["S09", "S15", "S16", "S17"]
---

# Make profiling useful for scientific claims

The existing [profiling guide](../../docs/profiling.md) contains tool commands and the distinction between application events and system telemetry. This page specifies the remaining measurement work.

## Add sustained-window measurements

Extend `observe.py` with explicit warmup markers, a configurable sample interval, thermal pressure and swap/available-memory samples where permitted, and interval start/end timestamps. Record source, units, scope, availability, and collector version per channel. Keep a missing counter null with a reason. Summarize cold start, warmup, and steady state separately.

Add a run manifest containing code revision/hash, model/tokenizer/data hashes, full configuration, participant identities, OS/runtime versions, and network topology. The current event logs identify runs but do not provide a complete multi-machine reproducibility manifest. Preserve both wall time for approximate correlation and monotonic time for local duration. Cross-Mac causal matching should use run/version/request IDs, with clock uncertainty reported if you estimate offsets.

## Confirm placement rather than infer it

Use [Metal capture](../sources/S15.md) and [Core ML/Neural Engine instrumentation](../sources/S16.md) for detailed attribution. Operator traces identify work, while [mactop](../sources/S17.md) provides complementary system observations. No single power gauge establishes per-process ANE execution. If private-API events are absent, correlate explicit operation intervals and inspect activity while controlling unrelated workloads; state residual uncertainty.

[Co-execution research](../sources/S09.md) warns that lazy evaluation and materialization boundaries can change actual overlap. Two submitted tasks are not necessarily concurrent on hardware. Preserve explicit completed-work synchronization in performance tests and inspect the resulting timeline.

## Measure instrumentation cost

Use identical seeds and work in three modes: minimal measurements, normal events plus periodic samples, and a short detailed operator trace. Randomize trial order, repeat, and report the difference. Do not compare an instrumented ANE run against an uninstrumented GPU run. Trace a representative short interval, then use low-overhead measurements for longer experiments.

## Energy evidence

Estimated SoC subsystem watts are useful within a controlled device experiment, but are not automatically comparable across generations or collectors. Keep idle baseline, battery/AC state, and background work. For cross-device energy claims use an external meter or explain the uncertainty and coverage of an alternative. Avoid adding overlapping system and subsystem readings. Report joules per fixed workload or time-to-quality, not just low instantaneous watts.

**Acceptance:** one reproducible report explains a bottleneck using both phase timings and appropriate execution evidence; missing sensors and profiler perturbation remain visible. See [measurement decision](../decisions/0002-measurements.md).
