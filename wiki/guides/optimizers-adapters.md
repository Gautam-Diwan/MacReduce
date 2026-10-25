---
title: "Optimizer state DiLoCo and adapter aggregation"
type: implementation
status: proposed
updated: 2026-10-23
sources: ["S04", "S19", "S20"]
---

# Optimizer state DiLoCo and adapter aggregation

MacReduce currently uses local SGD without momentum and averages full parameters. There is no optimizer-state merge. Choose one extension at a time; naming it correctly makes comparisons interpretable.

## DiLoCo path

[DiLoCo](../sources/S04.md) uses a local optimizer and an outer optimizer with different roles. Read the full algorithm and checkpoint requirements before implementation. Introduce an optimizer interface and persist local and outer state with an explicit ownership policy. Test outer-update sign, learning-rate schedule, effective participation weights, and resume equivalence on a small deterministic problem. A successful local-SGD run does not validate this algorithm.

## LoRA and QLoRA path

Use [MLX-LM's supported fine-tuning path](../sources/S19.md) as an integration reference. Identify the exact trainable parameters and require identical base-model/tokenizer hashes on all workers. Record quantization and compute precision separately. Exporting only adapters reduces exchanged state, but each worker still needs the base model and training working set.

For a represented update DeltaW = B A, average(B) average(A) generally does not equal average(B A). This is an algebraic issue, not a claim that all factor averaging is invalid. A simple controlled baseline can freeze one common factor and train the other so aggregation is linear in the trainable factor. That restricts expressiveness and must be stated. Another baseline forms full deltas before averaging and optionally re-factorizes, but increases memory/communication and introduces approximation if rank is truncated.

Implement a tiny matrix counterexample test, then compare adapter outputs before and after aggregation. Keep rank, scaling, target layers, initialization, and optimizer states fixed. Search current federated-LoRA literature before claiming any reducer is new; the current wiki has not completed that review.

## Model averaging versus continued optimization

One-shot branch/train/merge and iterative local SGD answer different questions. For a course model-averaging comparison, use a common starting model and evaluate merged models on both shared and specialized held-out data. Averaging independently initialized models is not a valid substitute for that common-origin experiment. See [research gaps](../research/gaps.md) for work not yet reviewed deeply.
