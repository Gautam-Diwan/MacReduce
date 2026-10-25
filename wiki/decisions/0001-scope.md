---
title: "Decision 0001 Trainer first over Wi-Fi"
type: decision
status: active
updated: 2026-10-23
sources: []
---

# Decision 0001 Trainer first over Wi-Fi

**Decision:** use MacReduce as the project and package name. Build an observable local-SGD trainer over ordinary IP with heterogeneous Macs as the first complete vertical slice. The available M4 is the initial reference. Physical Wi-Fi is the target network; loopback is only a development test.

**Reason:** coarse local training amortizes communication and gives a feasible course-scale baseline. A large sharded inference engine, an ANE training backend, a new transport, and a new optimizer all at once would make failures difficult to interpret.

**Consequences:** the [roadmap](../project/roadmap.md) gates model scaling, hardware support, and policy experiments on correctness. [Sharded inference](../guides/inference.md) remains a separate evaluated integration. This is a scope decision, not a claim that training is faster on multiple Macs or that ANE will improve it.

**Revisit when:** physical results show the workload is communication-dominated at every useful local-step setting, or the team chooses a different primary research question with a measured baseline.
