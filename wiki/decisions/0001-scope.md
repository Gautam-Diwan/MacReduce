---
title: "Decision 0001 Trainer first over Wi-Fi"
type: decision
status: historical
updated: 2026-10-25
sources: []
---

# Decision 0001 Trainer first over Wi-Fi

**Superseded 2026-10-25 by [decision 0003](0003-build-target.md).** The trainer built under this decision is retained as a measured prototype; the Wi-Fi scheduling study is no longer the research claim because Apple's MLX distributed stack and community systems already cover connecting Macs, and ANE training prototypes already exist. The text below is unchanged history.

**Decision:** use MacReduce as the project and package name. Build an observable local-SGD trainer over ordinary IP with heterogeneous Macs as the first complete vertical slice. The available M4 is the initial reference. Physical Wi-Fi is the target network; loopback is only a development test.

**Reason:** coarse local training amortizes communication and gives a feasible course-scale baseline. A large sharded inference engine, an ANE training backend, a new transport, and a new optimizer all at once would make failures difficult to interpret.

**Consequences:** the [roadmap](../project/roadmap.md) gates model scaling, hardware support, and policy experiments on correctness. [Sharded inference](../guides/inference.md) remains a separate evaluated integration. This is a scope decision, not a claim that training is faster on multiple Macs or that ANE will improve it.

**Revisit when:** physical results show the workload is communication-dominated at every useful local-step setting, or the team chooses a different primary research question with a measured baseline.
