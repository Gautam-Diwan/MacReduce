---
title: "Decision 0002 Preserve measurement and objective semantics"
type: decision
status: active
updated: 2026-10-23
sources: ["S06", "S09"]
---

# Decision 0002 Preserve measurement and objective semantics

**Decision:** preserve raw evidence, label unsupported capabilities unknown, and separate observed activity from attributed execution. A requested backend must fail clearly when unavailable; it must not silently fall back and retain the original label.

**Decision:** new automatic work allocation must explicitly define dataset mixture weights independently of device throughput. The existing prototype's token-weighted reducer remains a documented baseline until a new algorithm and protocol version are validated.

**Reason:** heterogeneous local work can change the optimized objective ([FedNova](../sources/S06.md)); adding execution resources can introduce synchronization and calibration costs ([FusionML](../sources/S09.md)). Fast-looking counters or a falling training loss alone are insufficient.

**Consequences:** use the [profiling guide](../guides/profiling.md), [heterogeneity guide](../guides/heterogeneity.md), and [experiment gates](../research/experiments.md). Store precision, optimizer semantics, source weights, and instrumentation in run manifests. Report measured, reported, inferred, and untested claims separately.

**Revisit when:** a validated new counter or attribution method is available, or a different optimization objective is deliberately selected and documented before comparison.
