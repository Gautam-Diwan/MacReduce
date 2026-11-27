---
title: "Training optimization strategy for federated QLoRA"
type: research
status: proposed
updated: 2026-11-27
sources: ["S04", "S05", "S06", "S07", "S08", "S163", "S164", "S165", "S184", "S185", "S186"]
---

# Training optimization strategy for federated QLoRA

**Status:** proposed. The bottleneck numbers come from the [2026-11-24 three-Mac run](../experiments/2026-11-24-three-mac-gpu-federation.md) and the [2026-11-27 offline measurements](../../evidence/training-optimization-2026-11-27/README.md). No optimization below has been implemented or validated end to end.

## Bottom line

**Inferred:** the federated recipe is slow because it synchronizes far too often, not because of fp32 or the GPU. At two local steps per round, the three-epoch recipe needs 2,048 rounds. With the measured 55–87 s per round, that is about 31–50 hours. The training itself is small: one client's third of three epochs is 412,181 target tokens. That is about 12–16 minutes of M4 Air GPU time at the measured 432–558 tokens per second.

Packing lines into 512-token rows and syncing every 64 local steps would cut the run to about 13 rounds. Wire compression then decides how many extra rounds the same time budget buys.

The user's intuition holds in that form: a smaller wire format is worth it when it buys more synchronization rounds than fp32 for the same wall clock. It is a second-order lever after the sync interval, not a substitute for it.

## Time model

One run takes R × (slowest client's K-step compute + transfer + fixed round cost).

- R is the number of rounds: the per-client steps divided by K, the local steps per round.
- Per-client steps are 4,096 line steps today, or about 805 packed 512-token steps (measured token count).
- The 2026-11-24 round had about 1–4 s of Air compute inside a 55–87 s barrier (measured).
- The rest is Pro and M1 compute, three fp32 uploads and three fp32 downloads through the Air, the Flower hello exchange, and checkpoint writes. Their split is **unknown** because no per-client timings were logged.

| Lever | Effect on the model | Size of effect |
|---|---|---|
| Raise K | divides R | 2 → 64 is 32 times fewer rounds (arithmetic) |
| Pack lines into 512-token rows | divides steps per client | 4,096 → 805, 5.1 times (measured token count) |
| Compile the step, drop checkpointing | shortens compute per step | +13% to +29% tok/s; checkpointing cost 27–29% (measured, Air) |
| Wire codec | shortens transfer per round | 3.8 times (int8) to 7.1 times (int4) smaller than fp32 (arithmetic) |
| Time-budgeted heterogeneous rounds | removes straggler wait | **unknown** until Pro and M1 step rates are measured |

## Bottlenecks, ranked

1. **Round count (dominant; inferred from measured inputs).** The pilot ran K = 2 at one WikiText line per step. That was a connectivity test setting, not a recipe. In the project's terms, the "gradient accumulation" the user wants lowered is this sync frequency. Accumulation itself is already 1.

   [S184](../sources/S184.md) reports sync intervals of hundreds of inner steps with mild loss. DiLoCo's H = 1000 cost 2.9% perplexity against H = 50 in pretraining. Short WikiText lines make it worse: 37.9% of the 4,096 lines carry fewer than 32 target tokens, and the median is 76 (measured). So a step does little work.

2. **Per-round fixed cost (unknown split).** Each round moves 19.25 MiB up and 19.25 MiB down per client in fp32, all through the Air in a star. The coordinator broadcasts the full adapter, not a delta (code inspection of `state.py`).

   Every round also does a Flower `get_properties` exchange and writes about 80 MB of checkpoints per worker. The Tailscale path, direct or DERP relay, was not recorded.

3. **Straggler.** Uniform work per round makes the slowest Mac set the pace. The M1 also ran with gradient checkpointing. On the Air, checkpointing cost 27–29% throughput (measured), and the uncheckpointed step peaked at 1.83 GB (line) and 2.46 GB (pack512) of MLX memory (measured). That suggests the 8 GB M1 does not need checkpointing (inferred; M1 memory is unmeasured).

4. **Step efficiency (measured on the Air).** The backend retokenizes each line on every step, syncs the GPU three times per step, and does not compile. Packing plus `mx.compile` gave 558 against 432 tok/s. Batch 4 at 512 tokens gave no further gain (509 tok/s) and used 6.9 GB, so batch size is not the lever on this GPU.

5. **Wire precision (measured on real deltas).** On the Air's deltas, relative L2 error was:

   | Codec | Relative error | Cosine |
   |---|---|---|
   | fp16 | 0.0002 | 1.0 |
   | bf16 | 0.0017 | 1.0 |
   | int8 g64 | 0.003–0.004 | 0.99999 |
   | int4 g64 | 0.05–0.07 | 0.998 |
   | int2 | 0.21–0.36 | — |

   These are early two-step deltas, so the result may change at larger K (unknown). The deltas are dense: the top 10% of entries hold 25–31% of the energy. That argues against top-k sparsification, matching the [S185](../sources/S185.md) report that dropping values hurt while FP4 did not.

## Strategies by viewpoint

### Distributed optimization

- **Sync by tokens, not lines (proposed).** Choose K so each client sends after a fixed token count, for example 32k tokens, which is 64 packed steps. Sweep K in {16, 64, 128}. [S07](../sources/S07.md) AdaComm and [S08](../sources/S08.md) HEW are prior art for adapting K. A fixed sweep comes first.
- **Outer Nesterov momentum (proposed arm).** [S184](../sources/S184.md) reports outer SGD with Nesterov momentum (lr 0.7, momentum 0.9) beat plain FedAvg in pretraining. In MacReduce it is a server-side change: keep a momentum buffer over the merged delta. The baseline is FedAvg, which is outer lr 1 without momentum. The outer hyperparameters must be re-tuned for LoRA.
- **Freeze A (FFA-LoRA, proposed arm).** [S186](../sources/S186.md) makes factor aggregation exact and halves the payload. On our deltas, though, A carried 47–49% of update energy after round 0 (measured). Freezing it removes real movement, so it needs its own single-node FFA control. It is the "freeze one factor" baseline already named in [adapter aggregation](../guides/optimizers-adapters.md), not a contribution.

### Wire format and "quantization-aware" training

- **Separate master precision from wire precision (proposed).** Keep fp32 LoRA weights and fp32 AdamW on each client. Quantize only the outgoing delta, as [S185](../sources/S185.md) does with FP32 accumulation and FP4 transport.
- **Codec ladder:** bf16 as the safe default (half the bytes, 0.17% error); int8 with error feedback; int4 with error feedback.
- **Error feedback makes training quantization-aware.** Each client keeps its residual from earlier rounds and adds it to the next delta, so compression error is carried forward rather than dropped. In simulation over three Air rounds, the cumulative error fell by about one third at every bit width (measured).
- **Persist the residual.** It goes in worker checkpoints next to the AdamW moments.
- **Quantize the downlink too (proposed).** Broadcast the quantized merged delta rather than the full fp32 adapter. The server must apply the same dequantized values it sends, so every replica stays bit-identical. That way the full bidirectional saving is realized.
- **Use the MLX quantizer on the GPU.** `mx.quantize` (affine, group 64, along the long axis of each factor) matched the NumPy reference: 0.0037 for int8 and 0.064 for int4. It encoded all 392 tensors in 4–6 ms (measured). MX-FP4 was smaller but had 0.103 error (measured).
- **The trade, as the user framed it:** at equal bytes, 13 fp32 rounds cost the same as about 49 int8 rounds or 92 int4 rounds (arithmetic). Whether that wins depends on the fixed per-round latency, which is the reason to instrument first.

### Heterogeneity

- **Time-budgeted rounds (proposed).** Each client runs as many packed steps as fit in a fixed wall-clock budget. The faster Pro then does more work instead of idling. Aggregation weights would follow the tokens each client actually processed.
- **Normalize unequal step counts (proposed arm).** [S06](../sources/S06.md) and [S163](../sources/S163.md) FedNova normalization is the arm for unequal local steps. Its scalar form is derived for SGD, so treat the AdamW variant as unvalidated.
- **Measure each Mac first.** The per-device microbatch and step budget should come from running `tools/bench_local_step.py` on each Mac. The objective (data shares, token weights) stays separate from the throughput calibration, per [heterogeneity](../guides/heterogeneity.md).
- **Treat the ANE as its own client.** An older single-Mac GPU+ANE curve averaged about 276 tok/s with evaluation interleaved. The GPU-only step here ran 432 tok/s (measured, different conditions). So the Neural Engine may add more as a separate client in the split cohort than as a suffix inside a GPU client (inferred; needs a matched measurement).

### Mac architecture

- **Pack and compile on every client (proposed).** Retokenize once at setup, not per step.
- **Drop M1 checkpointing (proposed).** Remove it unless an M1 benchmark shows memory pressure.
- **Expect the GPU, not memory, to cap throughput.** The Air's GPU saturated at one 512-token row: batch 4 gave no gain (measured). The Pro may differ (unknown).
- **Watch the fanless Airs.** Log tokens per second per round to catch thermal throttling. A 40-step benchmark cannot see sustained thermals (unknown).
- **AdamW bias correction.** MLX AdamW leaves bias correction off by default. Early updates are about 3–4 times the learning rate, which the round-0 delta maximum confirms (code plus measurement). Restore the single-node warmup in the federated plan, expressed in tokens.

### Systems

- **Instrument first (proposed, prerequisite).** Log per client per round:
  - setup, compute, snapshot, and encode times;
  - upload send time and coordinator receive time;
  - bytes and checkpoint time;
  - the Tailscale path (`tailscale ping` / `netcheck` once per run).

  Without this, the barrier cannot be attributed. Missing values stay missing, not zero.
- **Prune old checkpoints (proposed).** Keep the last N worker generations and coordinator adapters. At K = 2 the run would have written about 160 GB per worker (arithmetic).
- **Overlap transfer with compute (deferred).** Streaming DiLoCo-style overlap and fragmenting (tau, alpha) in [S185](../sources/S185.md) are only worth it if transfer stays a large share of the round after the changes above. They also need a transport beyond Flower's synchronous fit round.

## Ordered plan

| Phase | Change | Gate before the next phase |
|---|---|---|
| 0 | Per-client timing and bytes in receipts; Tailscale path; benchmark on Pro and M1; checkpoint pruning | Round barrier decomposed into compute, transfer, and fixed cost on all three Macs |
| 1 | Packed 512-token rows, compiled step, no M1 checkpointing, K by tokens. Rerun the single-node packed control on one Mac, about 40 minutes at the measured Air rate | The packed single-node validation perplexity is the new reference. Packing changes the sample definition, so the 23.30 test perplexity is not directly comparable |
| 2 | Codec layer: bf16, int8 with error feedback, int4 with error feedback, on uplink and downlink | Validation perplexity per arm against fp32 at the same K, plus bytes and wall clock |
| 3 | Outer Nesterov versus FedAvg; FFA-LoRA versus full LoRA, each with its own controls | Same metrics; one change at a time |
| 4 | Time-budgeted rounds with token weights; FedNova arm; split GPU and ANE clients | Straggler idle time and perplexity against uniform rounds |
| 5 | Overlap or fragment sync, only if transfer is still above roughly 20% of the round | Measured compute utilization |

**Course experiment budget (inferred):** a packed K = 64 run is about 13 rounds. If the M1 runs about 2.5 times slower than the Air, which is unknown, a round is about 2–3 minutes. So one federated arm takes under an hour, and the phase 2 and 3 matrices fit in a few evenings.

## What this does not establish

No convergence result exists for any arm. Packing, larger K, compression, outer momentum, and FFA-LoRA can each change final perplexity, and the size of that change is unknown until the matched runs exist. The codec errors come from three early two-step deltas from one client. Step rates are from one M4 Air over 40 steps. None of these methods is new. The [three-Mac study](three-mac-wireless-federation.md) and [gaps ledger](gaps.md) track the remaining unknowns.
