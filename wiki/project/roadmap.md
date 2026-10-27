---
title: "Implementation order and completion gates"
type: plan
status: active
updated: 2026-10-27
sources: ["S27", "S32", "S33"]
---

# Implementation order and completion gates

This plan follows [decision 0003](../decisions/0003-build-target.md). The 10-805 calendar supplied on 2026-10-27 is feedback 2026-10-28, teams 2026-11-02, check-in 2026-11-18, report 2026-12-01. Effort figures are estimates, not measurements.

The course project is heterogeneous federated learning on WikiText-2, still named MacReduce. The curve is three epochs of the first 4,096 training lines. `macreduce federate` lists GPU, combined GPU and Neural Engine, and split clients, repeated on every Mac. FedAvg is the communication reduction and a per-step all-reduce is the control. The payload is the adapter delta. Multi-Mac cells wait for a second machine. The serving table below is future scope, not the submitted dataset, metric, and baseline. As of the 2026-10-25 study, items 1, 2, 4, and the local half of item 6 have measured results on one M4 for the Qwen2.5 pair ([evidence](../../evidence/serve-benchmark/README.md)). The serving pin is now Qwen3, with a token check and no new speed sweep ([Qwen3 check](../../evidence/serve-equivalence-qwen3/README.md)). The 2026-10-26 amendment allows a private-API ANE stack when it addresses an open gap, and adds a QLoRA baseline on WikiText-2. Item 5's CPU draft ran once. Its ANE gate times a decoder block and a 28-layer Qwen3-0.6B-shaped forward ([decoder](../../evidence/ane-decoder/README.md), [infer](../../evidence/infer-qwen3/README.md)). The saved decoder ranking omitted the Neural Engine because only two bandwidth samples cleared 1 GB/s. The full-depth package did not fit on the remaining disk. The pinned Qwen3 weights are still not converted. The earlier convolution record remains ([usage](../../evidence/ane-usage/README.md)). Item 7 was not run. The joint time-to-first-token and decode hypothesis was not met. An early end-of-sequence stop was not observed: all 45 Qwen2.5 sweep trials hit the 32-token cap.

| Order | Work package | Window | Exit evidence |
|---|---|---|---|
| 1 | **Pin and verify the engine.** Pin `mlx`, `mlx-lm`, one target model and one or two draft models; verify greedy equivalence of speculative versus plain decoding on the M4 before any timing ([S33](../sources/S33.md) failures) | by 2026-10-31 | Pinned versions in a manifest; equivalence report with token-level diff; any failing model pair excluded with the reason |
| 2 | **Harness and trace.** Request trace with short chat, long-context, and code segments; A-B-B-A runner in fresh processes with a noise floor ([S27](../sources/S27.md) method); TTFT, decode tok/s, requests per minute, bytes, exact-match | by 2026-11-05 | One static sweep on the M4: k in {0,1,2,4,8} with the GPU drafter; raw JSON plus a summary; noise floor stated |
| 3 | **Proposal** | 2026-11-06 | Proposal cites the sweep; states hypothesis, baselines, falsification, and what needs a second Mac |
| 4 | **D2 adaptive depth.** Per-step k from running acceptance and measured draft and verify costs; compare with best fixed k | by 2026-11-13 | A-B-B-A result per trace segment; ablation with D1 and D3 fixed |
| 5 | **D1 drafter engine.** CPU-stream drafter concurrent with GPU verify; ANE gate: Core ML conversion of the draft model, placement check by power or compute plan, parity gate; include the ANE engine only if the gate passes | by 2026-11-20 | Engine cost table; gate outcome recorded either way; if the gate fails, that is the finding for direction 2 |
| 6 | **D3 prefix source, local.** `mlx_lm.cache_prompt` files on disk; decide reuse versus recompute from prefix length, bytes, and measured disk rate | by 2026-11-25 | Crossover curve on the M4 with both static arms shown |
| 7 | **D3 remote arm (only with a second Mac).** Pull KV from peer versus send prompt for remote prefill versus local; reuse the existing chunked artifact transfer; measure the link rather than assuming it | 2026-11-20 to 2026-11-30 | Same crossover with link rate as a parameter; if no second Mac, report the arm as not run |
| 8 | **Combined policy and report.** All three decisions against the hindsight-best static triple; negative results kept | by 2026-12-03 | Final tables, limitations, and an updated [state](state.md) and [log](../log.md) |

## What is reused from the existing code

The coordinator and worker processes, chunked artifact transfer with hashes, and the run-manifest conventions in `macreduce` are reusable for telemetry collection and KV-file movement. The trainer itself is not on the critical path and stays as the measured prototype described in [state](state.md).

## Definition of finished

Pinned versions; greedy equivalence verified; A-B-B-A results with a noise floor for each decision alone and combined; the hindsight-best static baseline reported honestly; the ANE gate outcome recorded; the second-Mac arm either measured or marked not run; limitations and the unverified items from [gaps](../research/gaps.md) in the report. Do not describe this as satisfying a general-purpose MapReduce assignment; map and reduce describe the scheduler's structure only.
