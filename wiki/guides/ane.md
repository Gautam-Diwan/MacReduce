---
title: "Integrate ANE training through falsifiable gates"
type: implementation
status: proposed
updated: 2026-10-29
sources: ["S10", "S11", "S12", "S16", "S37"]
---

# Integrate ANE training through falsifiable gates

**Scope note 2026-10-28:** [S37](../sources/S37.md) is Bryngelson's ANE reference, read in the scope on that card. It says the engine has no native backward op and that a host-built backward graph still trains, including transformers whose ops are in the registered set. The silicon numbers in the chapters that were read are M1 and M5. They are not a measurement of Qwen3 QLoRA on this M4.

**Scope note 2026-10-27:** The 10-805 project treats the Neural Engine as a federated client, either combined with the GPU or separate from it. ANEForge 0.4.0 is installed. Orion and maderix are not pinned. `macreduce federate` records which import. An import does not by itself show residency. The gates below still apply before a number is kept.

**Scope note 2026-10-26:** [Decision 0003](../decisions/0003-build-target.md) allows a private-API ANE training stack when it addresses a gap ANEForge, Orion, or maderix leave open. The inference option remains: a Core ML draft model whose placement is verified by power or compute plan and whose outputs pass a parity gate ([S25](../sources/S25.md), [S29](../sources/S29.md), [S36](../sources/S36.md)). A later probe converted a small convolution, not that draft, and read a compute plan ([probe](../../evidence/ane-probe/README.md)). A decoder block and a 28-layer Qwen3-0.6B-shaped forward were then timed ([decoder](../../evidence/ane-decoder/README.md), [infer](../../evidence/infer-qwen3/README.md)). The saved decoder ranking omitted the Neural Engine on a two-sample bandwidth trace, and the full package did not fit on disk. The gates below remain the right shape for the Qwen draft check.

Before a new `aneforge` call, search [ANEForge lookup](../queries/aneforge.md). The call rules are in [Write an ANEForge program](aneforge.md).

One Qwen3 decoder block, frozen base plus LoRA, compiles for the Neural Engine and is the GPU+ANE training suffix. The other blocks stay 4-bit QLoRA on the GPU. The fp16 backward scales the cotangent to peak 1 and backs off if that overflows. On this M4 that backward matched float32 at peak 1, and an isolated rms_norm backward did not overflow. The three-epoch curve has not finished. ANEForge 0.4.0 has also run a linear Adam step and a matmul on device mask 0x4, recorded in [evidence/ane-forge](../../evidence/ane-forge/README.md). Public inference support is not a public general training API. Orion and maderix still have to be pinned and validated on this macOS build.

## Candidate selection

[ANEForge](../sources/S10.md) offers a Python-oriented training interface with different optimizer/residency modes. [maderix ANE](../sources/S11.md) supplies native training examples including a dynamic-weight path. [Orion](../sources/S12.md) is another reported training system. Their published results are external claims. MacReduce has executed the installed ANEForge linear step, not the upstream training examples and not Orion or maderix. The model, precision, and API still have to fit the Qwen3 adapter before an engine client can share a delta with the GPU.

## Gate A availability and simple execution

Pin a repository commit and dependency versions in a separate environment. Follow that revision's build instructions; do not copy commands from an old summary. Run an upstream tiny example, retain full logs, and capture actual Neural Engine activity through Instruments where accessible. Record unsupported ops, compilation failures, and runtime fallbacks. Do not declare success from an ANE power change alone.

## Gate B forward and backward correctness

Start with matmul and a two-layer MLP. Compare forward outputs, parameter gradients, and one SGD step against the CPU reference with identical shapes and numerical precision where possible. Add a finite-difference check on a small float32 reference problem and distinguish finite-difference error from backend rounding. Report relative and absolute errors; cover zero/large/small inputs and shape boundaries. Require at least 100 stable updates before adding attention.

Then validate normalization, activations, embeddings, residuals, causal masking, and attention backward independently. An unsupported backward op must be explicit in the placement map. Make CPU fallbacks visible in timings and correctness tests.

## Gate C fit the backend interface

Add a backend implementing load, evaluate, step, export, and memory reporting with the existing shared parameter layout. If the selected runtime requires a different model layout, build a tested converter and identity check rather than changing checkpoint meaning silently. Include optimizer reset/resume semantics and buffers needed by resident execution. A load after averaging must actually update the values consumed by the next compiled step.

Log compilation count/time, parameter upload, forward/backward compute, host optimizer work, downloads, and complete step time. Run cache-hit and cache-miss cases. Account for compilation after shape changes and distinguish weight-as-input programs from weight-baked programs. Run one backend locally before introducing it as a cluster worker.

## Gate D benefit and stop decision

Compare matched GPU-only and ANE paths for time to a fixed validation loss. Include initialization amortized over an explicitly stated run length. Test resident local blocks between synchronization events, because distributed averaging may force an expensive state materialization. Measure concurrent GPU/ANE execution only after isolated baselines; shared memory and thermal limits may erase the gain.

Stop integration if gradients are unexplained, required ops are missing, the selected OS cannot run it, or complete-step time fails the chosen benefit criterion. A documented negative result still answers the feasibility question. Do not call a mostly CPU optimizer path “ANE-only training.” Related: [profiling](profiling.md), [hardware resources](accelerators.md), and [candidate B](../research/novelty.md).
