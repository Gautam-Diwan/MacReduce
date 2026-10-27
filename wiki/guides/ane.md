---
title: "Integrate ANE training through falsifiable gates"
type: implementation
status: proposed
updated: 2026-10-27
sources: ["S10", "S11", "S12", "S16"]
---

# Integrate ANE training through falsifiable gates

**Scope note 2026-10-27:** The 10-805 project treats the Neural Engine as a federated client, either combined with the GPU or separate from it. ANEForge, Orion, and maderix are the private stacks. `macreduce federate` records which import. None of the three stacks is pinned, and an import would not by itself show residency. The gates below still apply before a number is kept.

**Scope note 2026-10-26:** [Decision 0003](../decisions/0003-build-target.md) allows a private-API ANE training stack when it addresses a gap ANEForge, Orion, or maderix leave open. The inference option remains: a Core ML draft model whose placement is verified by power or compute plan and whose outputs pass a parity gate ([S25](../sources/S25.md), [S29](../sources/S29.md), [S36](../sources/S36.md)). A later probe converted a small convolution, not that draft, and read a compute plan ([probe](../../evidence/ane-probe/README.md)). A decoder block and a 28-layer Qwen3-0.6B-shaped forward were then timed ([decoder](../../evidence/ane-decoder/README.md), [infer](../../evidence/infer-qwen3/README.md)). The saved decoder ranking omitted the Neural Engine on a two-sample bandwidth trace, and the full package did not fit on disk. The gates below remain the right shape for the Qwen draft check.

ANE training is not implemented in MacReduce. Investigate it as a bounded backend project after the GPU and physical network baselines work. Public inference support is not a public general training API. The candidate runtimes below use experimental paths and must be pinned and validated on the actual macOS build.

## Candidate selection

[ANEForge](../sources/S10.md) offers a Python-oriented training interface with different optimizer/residency modes. [maderix ANE](../sources/S11.md) supplies native training examples including a dynamic-weight path. [Orion](../sources/S12.md) is another reported training system. Their results are external claims; none has been executed by MacReduce. Choose one integration first and document why its model, precision, and API fit the reference workload.

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
