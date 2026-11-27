---
title: "Run and characterize a physical WiFi cluster"
type: implementation
status: proposed
updated: 2026-11-25
sources: ["S03", "S04", "S05"]
---

# Run and characterize a physical WiFi cluster

**Current executable path:** for QLoRA over different networks, use [Tailscale + Flower](tailscale-federation.md). Its default example is explicitly a connectivity pilot; the measurements below still need to be performed on the actual three Macs.

**Planning update 2026-11-12:** the active proposal again includes physical federated training. The team now reports two M4s and an 8 GB M1. See the [three-Mac wireless study](../research/three-mac-wireless-federation.md) for LAN/SSH, optional Tailscale, the M4 Air 24 GB as coordinator and training client, adapter-sized transfer calculations, and fixed-work experiments. Older scope notes below describe their dates, not a new exclusion of wireless federation.

**Measured 2026-11-24:** a GPU Flower/Tailscale pilot ran on those three Macs. This Wi-Fi characterization protocol was not used; Tailscale path (direct vs relay) was not recorded. [Report](../experiments/2026-11-24-three-mac-gpu-federation.md).

**Scope note 2026-10-25:** under [decision 0003](../decisions/0003-build-target.md) Wi-Fi is no longer the research question. Keep this page as the operational procedure for any multi-Mac measurement (for example the remote arm of the prefix-source decision); the link rate is a measured parameter, not a claim.

Use [the README commands](../../README.md) with unique worker IDs, the same source revision, a shared secret, and explicit IP addresses. Choose a trusted private network or TLS/SSH transport. University client isolation may prevent peer connections; distinguish this from a software failure. Keep a coordinator worker optional and record whether it shares compute with reduction and validation.

## Required measurements

1. Save each machine's `doctor` output, active Wi-Fi interface, OS and runtime versions, power mode, RAM, and selected backend. Do not guess that `en0` is Wi-Fi. Record access-point model and band when known; SSID or physical location can be omitted from shared results.
2. Run the existing tiny model to verify transport. Then measure application payload transfers at representative sizes in both directions and under concurrent workers. An extension should provide a bounded probe endpoint or a separate transfer benchmark; the current 123 KB model cannot establish large-transfer bandwidth.
3. Establish one-Mac and two-Mac baselines at matched token budgets and fixed validation data. Repeat at least three times. Report median and spread, not just the best run.
4. Sweep common K over 1, 10, and 50; keep optimizer, effective batch policy, dataset mixture, and seed schedule documented. Compare wall-clock time to a predefined validation threshold, plus loss versus processed tokens.
5. Interrupt one connection after an upload is accepted but before its acknowledgement. Verify a byte-identical retry does not double-count. Stop a worker before upload and confirm the deadline fails the fixed cohort. Restart from the last committed version.

## Interpret the network cost

For a full snapshot of S bytes and effective per-worker transfer rate B bytes/s, download plus upload costs roughly 2S/B before contention. Example planning values: 30M fp32 parameters imply about 120 MB, so 50 MB/s implies at least 4.8 seconds per worker per round. These are decimal units and assumptions, not Wi-Fi measurements.

The access point may relay client-to-client traffic on the same radio. N concurrent workers do not each receive the isolated rate B. Record aggregate bytes, queueing, transfer p95, and barrier wait. A 10-second local training block can still lose to a slow upload tail. Check units carefully: Mbps is megabits/s; MB/s is megabytes/s.

[A competing Mac trainer](../sources/S03.md) already advertises Wi-Fi operation. Reproduce its relevant baseline if practical; do not adopt README speedup statements as measured facts. [Streaming DiLoCo](../sources/S05.md) is relevant to reducing synchronization bursts, but overlapping updates introduces algorithm and versioning changes. See [transport work](transport.md) before implementing it.

**Exit artifacts:** commands/configuration, node inventory, topology notes, raw events, baseline and sweep tables, interruption result, and explicit limitations. File the report using [the template](../templates/experiment.md).
