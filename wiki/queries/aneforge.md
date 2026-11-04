---
title: "ANEForge lookup"
type: guide
status: active
updated: 2026-10-29
sources: ["S45", "S52", "S55", "S58", "S65", "S67", "S70", "S76", "S102", "S158", "S159", "S160", "S161"]
---

# ANEForge lookup

Search this page before writing an `aneforge` call. The procedures and the reasons are in [Write an ANEForge program](../guides/aneforge.md). The saved URLs are in [the URL index](../research/aneforge-url-index.md). Names below were checked against the docs retrieved on 2026-10-29. Where a name was also checked in the installed 0.4.0 package, the guide says so.

## Which call

| You are about to | Use | Do not |
|---|---|---|
| Run a fused forward | `af.compile(out)` once, then call the model ([S65](../sources/S65.md)) | Compile inside the step loop |
| Change a weight every step | `af.parameter(array)` ([S76](../sources/S76.md)) | Bake the weight as a constant; the next update recompiles |
| Take a gradient of a scalar loss | `af.backward(loss, params, loss_scale=1024.0)` ([S76](../sources/S76.md)) | Divide the on-engine Adam learning rate by that scale again |
| Start from a host cotangent | `af.backward_from(grad_root, root, params)` ([S76](../sources/S76.md)) | Feed a raw fp16 cotangent whose peak is far below 1 without a scale |
| Train a small model on the engine | `af.Trainer(..., device_optimizer=True)` as in [S159](../sources/S159.md) | A host loop of one tiny matmul per linear |
| Keep state across steps | `Program.share_buffer` as in [S160](../sources/S160.md) | Re-upload the KV cache or the Adam moments every step |
| Several outputs, one program | `aneforge._compile.compile_multi` in 0.4.0 ([S55](../sources/S55.md)) | `af.compile_multi`; 0.4.0 does not export that name |
| Prefill and decode a Qwen | `af.load_llm("Qwen/Qwen3-0.6B")` ([S67](../sources/S67.md)) | Treat that loader as a training step |
| Causal attention inside the native window | `af.sdpa(..., is_causal=True)` as in [S161](../sources/S161.md) | Native `sdpa` for a long prefill; the LLM page says the graph cut is about 170× slower than the decomposed matmul ([S67](../sources/S67.md)) |
| Gate a compile for this M4 | `af.compile(y, target="h16s")` ([S65](../sources/S65.md)) | Assume an M5 measurement is this machine |
| Classification loss | `af.softmax_cross_entropy` ([S76](../sources/S76.md)) | `log` of a probability; it overflows fp16 |

## When a number looks wrong

| Symptom | Check |
|---|---|
| Matmul output is inf while the math is below 65504 | Output magnitude near 32752. Scale before the matmul and undo it after ([S70](../sources/S70.md)). |
| A slice or gather is inf or the wrong elements | Nonzero last-axis begin. Pre-A16 saturates above 4094. Transpose the axis off the end, gather, transpose back ([S58](../sources/S58.md), [S70](../sources/S70.md)). |
| An integer sum is off by one | Totals above 2048 are not exact in fp16 ([S70](../sources/S70.md)). |
| Backward expects more inputs than the forward | `rms_norm` / `layer_norm` / `group_norm` re-inject gamma as `attrs["value"]`. Feed every such input ([S52](../sources/S52.md)). |
| On-engine Adam step is ~1024× too small | The moments already absorbed `loss_scale`. Do not also divide the learning rate ([S76](../sources/S76.md)). |
| `DispatchFloorWarning`, or a rank-8 matmul takes ~0.2 ms | The program is smaller than the round trip. Fuse more work into one compile ([S55](../sources/S55.md)). |
| Causal `sdpa` is refused or garbage | Native is reliable only for sequence ≤ 2048 and `min(query, key) < 512`. Causal outside that window is refused ([S58](../sources/S58.md)). |
| Deep activations overflow fp16, and the block starts with RMSNorm | The Qwen3.5 loader scales the residual and relies on RMSNorm being scale-invariant ([S67](../sources/S67.md)). That is a source-reported pattern, not a measurement on this Mac. |
| A second compile, then the first program gets the wrong ports | `compile_multi` reassigns `t._name`. Snapshot port names at the first compile ([S55](../sources/S55.md)). |

## Which example to copy

The examples README at the tree `896c4943c5ebf811c0e815f8d1381fd1e157fa7d` is [S158](../sources/S158.md). The file list and sizes for the rest of that tree are [S41](../sources/S41.md). Only the four files below were read.

| Pattern | File |
|---|---|
| End-to-end on-engine Adam, host feeds the batch | `examples/demos/training_on_ane.py` ([S159](../sources/S159.md)): `af.parameter`, `af.mse`, `Trainer(..., device_optimizer=True)` |
| State that stays on the device | `examples/demos/resident_state.py` ([S160](../sources/S160.md)): `share_buffer` from the output onto the input, then `execute` with no host feed |
| One causal LLaMA block | `examples/llama_block_causal.py` ([S161](../sources/S161.md)): RMSNorm, `af.sdpa(is_causal=True)`, SwiGLU. The script's own shape is S=16, D=64 |
| K steps in one dispatch, resident optimizer | `UnrolledTrainer`, named in the README for `train_mnist_mlp.py`, `train_transformer.py`, `train_llama_block.py` ([S158](../sources/S158.md)). Those programs were not snapshotted |
| Gradient checkpointing so depth does not bound the compile | `aneforge.streaming.CheckpointedStack`, named for `train_neural_ca.py` and `train_charlm_deep.py` ([S158](../sources/S158.md)). Those programs were not snapshotted |
| Prefill plus resident KV decode | `af.load_llm` ([S67](../sources/S67.md)). The README points at `llm_chat.py` and `llm_prefill.py`; those files were not snapshotted |

## What was removed

GitHub directory HTML for fonts, badges, workflows, and folder listings was not kept. Those pages were the file tree rendered as a website. The tree itself is [S41](../sources/S41.md). The published docs are the Read the Docs snapshots [S42](../sources/S42.md) through [S76](../sources/S76.md). The architecture guide is [S77](../sources/S77.md) through [S125](../sources/S125.md).
