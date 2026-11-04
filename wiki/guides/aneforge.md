---
title: "Write an ANEForge program"
type: guide
status: active
updated: 2026-10-29
sources: ["S39", "S45", "S49", "S52", "S55", "S57", "S58", "S62", "S65", "S67", "S70", "S76", "S102", "S158", "S159", "S160", "S161"]
---

# Write an ANEForge program

Look here before writing a new `aneforge` call. The short table is [ANEForge lookup](../queries/aneforge.md). The URL inventory is [the crawl index](../research/aneforge-url-index.md). The installed package in this environment is `aneforge` 0.4.0. The docs below are the Read the Docs pages retrieved on 2026-10-29. Where a name was also checked in the installed 0.4.0 package, the sentence says so.

## The shape of a call

ANEForge is a graph, then one compile, then many runs ([S65](../sources/S65.md)). Build tensors, call `af.compile(out)` once, and call the result. A native program bakes constant weights into the blob. A weight that must change every step is `af.parameter(array)`, not a constant, or the next update forces a recompile ([S76](../sources/S76.md)).

`af.compile` lowers one output into one fused program ([S45](../sources/S45.md)). A graph whose output is itself an input is wrapped in a multiply by 1, because an empty program crashes the compiler. Arrays go in as fp16 and come back as fp32 ([S65](../sources/S65.md)). The installed 0.4.0 package exports `compile`, `parameter`, `backward`, `backward_from`, `Trainer`, `load_llm`, `rope`, and `rope_tables`. It does not export `compile_multi`. That function lives on `aneforge._compile` in 0.4.0, and the compile-pipeline page describes it as the multi-output compile ([S55](../sources/S55.md)).

`compile_multi` builds one program with several named outputs. The resident trainer uses it so an updated weight, moment, and second moment can be aliased back onto their inputs with `share_buffer` ([S55](../sources/S55.md), [S76](../sources/S76.md)). Port names are snapshotted at compile time. A later compile that shares those tensors reassigns `t._name`, so do not compile a second graph on the same tensors and then feed the first program by the old names ([S55](../sources/S55.md)).

Bridge ops such as `af.sdpa`, `argmax`, and `topk` cannot stay inside that one fused program. They cut the graph into a `SegmentedModel` ([S65](../sources/S65.md), [S55](../sources/S55.md)).

## What to compile, and what not to

The dispatch is single-in-flight. Threads do not hide it. `compile` warns with `DispatchFloorWarning` when the predicted time is the fixed round trip, about 0.2 ms on M1-class parts, rather than the math ([S55](../sources/S55.md)). The paper's abstract puts a small fused call near 90 µs against a 70 µs floor ([S39](../sources/S39.md)); that abstract was read, the PDF body was not. Amortize by putting more work in one program or by raising the batch. A loop that compiles or dispatches one tiny matmul per sample pays the floor every time.

The e5rt route is the unentitled path the docs recommend. Path B (`_ANEChainingRequest`) needs `com.apple.aned.private.allow`, which third-party processes do not have ([S62](../sources/S62.md)). Do not plan on Core ML placement either: the training page says the same e5rt path is used, with no Core ML ([S76](../sources/S76.md)).

The compute dtype is fp16 only. fp32, int32, and bf16 graphs are rejected at construction ([S58](../sources/S58.md), [S57](../sources/S57.md)). Rank is at most 5, and one axis is at most 65536 ([S58](../sources/S58.md)). A conv kernel is limited to width 15; height is not under that limit. `dynamic_conv` also requires batch 1.

## Numerics that change the graph

Matmul accumulates in a wide accumulator. `reduce_sum` accumulates in fp16. They are not interchangeable. A careful reduction is written as a matmul against ones, and the optimizer rewrites a single-axis `reduce_sum` that way ([S57](../sources/S57.md)).

Three correctness cliffs are measured on M2 Pro and M5, and the page says other families still need their own sweep ([S70](../sources/S70.md)):

- A matmul output becomes infinite near 32752, below the fp16 maximum of 65504. The bound is the output magnitude, not the contraction length.
- On pre-A16 chips, a slice with a nonzero last-axis offset saturates at 4094. M5 does not. A graph that must run on both stays inside the older limit.
- Integer totals above 2048 are not exact in fp16. A sum of 2049 can come back as 2050.

`sdpa` stays on the native layer only while the sequence is at most 2048 and `min(query, key)` is below 512. Outside that, the docs say native output is garbage and the op decomposes, except causal attention outside the native window is refused because the decomposition has no host-constant mask add ([S58](../sources/S58.md)). An attention mask must be `[1, 1, Sq, Skv]`. A per-head mask is rejected. Causal `sdpa` was checked on M1 at cosine 1.0 against the explicit softmax form.

A last-axis gather lowers to a width-offset slice and returns the wrong elements on A13 and A14. Transpose the gathered axis off the last position, gather, and transpose back ([S58](../sources/S58.md)).

## Training

There is no backward instruction. A gradient is a second graph of ordinary forward ops ([S102](../sources/S102.md), [S76](../sources/S76.md)). `af.backward(loss, params, loss_scale=1024.0)` builds that graph. `af.backward_from(grad_root, root, params)` starts from an explicit cotangent instead of a scalar loss ([S76](../sources/S76.md)). The seed scale is folded into an added constant so the compiler does not see a multiply of a reduction by zero.

The documented loss scale is 1024. It lifts small fp16 gradients, and the host divides it back out before SGD. On-engine Adam must not divide the learning rate by that scale: the moments absorb it and the ratio is invariant, so a second division collapses the step ([S76](../sources/S76.md)). The `Trainer` default in the API page is `loss_scale=1.0`; the worked examples pass 1024 ([S49](../sources/S49.md), [S76](../sources/S76.md)).

`Trainer` compiles one forward and one backward program per parameter. `device_optimizer=True` puts the update on the engine. `resident_state=True` fuses the step and keeps the optimizer state on the device; the host then feeds the batch and the scalar learning rate ([S76](../sources/S76.md)). `UnrolledTrainer` runs K of those steps in one dispatch. The page says that collapse is performance-neutral on the resident path, because the per-step dispatch was not the bottleneck.

Classification uses `af.softmax_cross_entropy`. The backward seed is `(softmax(logits) - target) * (loss_scale / N)`, with no logarithm, because a log overflows fp16 ([S76](../sources/S76.md), [S52](../sources/S52.md)).

`rms_norm`, `layer_norm`, and `group_norm` accept a baked array or a parameter tensor. The VJP's gamma is not a graph constant. It is re-injected as a fed value-input, and the trainer feeds every input that carries `attrs["value"]` ([S52](../sources/S52.md)). A caller that compiles the backward graph by hand has to feed those extra inputs. A baked matmul stores `W.T` and differentiates only the activation. `bmm` differentiates both sides ([S52](../sources/S52.md)).

The conv weight-gradient has no native op. On M1 and M2 its width-offset path saturates above 4094, so a huge loss scale can corrupt that gradient. The same CNN reaches 0.9080 on M1 and 0.9070 on M5 at loss scale 1024, one test sample in a thousand ([S102](../sources/S102.md), [S76](../sources/S76.md)). Those accuracies are the docs' numbers, not a MacReduce measurement.

## Decoder models

`af.load_llm("Qwen/Qwen3-0.6B")` returns a prefill plus KV-cache decode model. `prefill` returns next-token logits. `generate` prefills, then decodes one resident cache step at a time ([S67](../sources/S67.md)). The page reports Qwen3-0.6B prefill near 8,600 prompt tokens per second and decode near 75 tokens per second, and says int8 is byte-identical to fp16 on that small model but collapses a Qwen3-8B into repetition. Those rates were not remeasured here. The building blocks named on the page are `af.rope`, `af.rope_tables`, and `af.prefill_block`.

The same page says a long prefill should use the decomposed `softmax(Q @ K.T * scale + mask) @ V` inside one fused program. Native `af.sdpa` cuts the graph once per layer, and the page calls that path about 170× slower for prefill. Qwen3 specifics it says the loader already handles: a `head_dim` that is not `dim / n_heads`, QK-norm before RoPE, and a vocab large enough that the `lm_head` is tiled with `compile_multi` when `ane_lm_head` is on. For Qwen3.5 it scales the residual by a constant so deep fp16 activations do not overflow, and it says that scale is exact because every read goes through a scale-invariant RMSNorm. That residual scale is source-reported. It was not applied or measured in MacReduce.

Getting started names `target="h16s"` as the compile gate for an M4 ([S65](../sources/S65.md)). The installed 0.4.0 `af.compile` accepts `target`. This Mac was not recompiled under that flag for this page.

## Examples that show the intended call

The examples README at commit `896c4943c5ebf811c0e815f8d1381fd1e157fa7d` is [S158](../sources/S158.md). Three programs from that tree were read:

- [S159](../sources/S159.md) trains a linear map with `af.parameter`, `af.mse`, and `af.Trainer(obj, [w], lr=0.05, optimizer="adam", device_optimizer=True)`. The host replaces `tr.data` each step and calls `tr.step()`.
- [S160](../sources/S160.md) aliases a program's output buffer onto its input with `share_buffer`, then calls `execute` with no host array. The README says the same primitive keeps a KV cache and resident optimizer state.
- [S161](../sources/S161.md) builds one pre-norm LLaMA block: `rms_norm`, a reshape into `[1, heads, seq, head_dim]`, `af.sdpa(..., is_causal=True)`, and SwiGLU. Its own check uses sequence 16 and width 64, inside the native `sdpa` window in [S58](../sources/S58.md).

The README also names `UnrolledTrainer` for the MNIST and transformer trainers, and `aneforge.streaming.CheckpointedStack` for the neural cellular automaton and the 16-layer character model. Those files were not snapshotted.

## What this page does not cover

The paper PDF ([S38](../sources/S38.md)) was saved and not text-extracted. The abstract is [S39](../sources/S39.md). The architecture-guide chapters [S77](../sources/S77.md) through [S125](../sources/S125.md) were text-extracted into their cards so a search can find a term; they were not turned into call rules except the training chapter [S102](../sources/S102.md) and the sentences already cited above. The GitHub directory HTML for fonts, badges, and folder chrome was removed. The file tree at the saved sha remains [S41](../sources/S41.md). The installed 0.4.0 package was checked for `compile`'s signature, including `target`, and for `compile_multi` living on `_compile`. It was not diffed against that GitHub tree.
