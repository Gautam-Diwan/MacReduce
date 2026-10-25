---
title: "Move beyond the synthetic Transformer"
type: implementation
status: proposed
updated: 2026-10-23
sources: ["S19"]
---

# Move beyond the synthetic Transformer

The initial 29,888-parameter model is a learning and protocol smoke test. Its corpus is generated and combinations can overlap between training and validation. Do not call its loss decrease a generalization benchmark.

## Implement in this order

1. Introduce an immutable `ModelConfig` carrying vocabulary, width, heads, layers, context length, dtype, and tokenizer identity. Validate divisibility of width by heads and serialize it in every run and checkpoint. Remove the global-MODEL assumptions in coordinator initialization, worker batching, token-count checks, and checkpoint decoding together.
2. Generalize repeated Transformer blocks with deterministic parameter names shared across backends. Keep a small configuration in tests. Add save/load round-trip and CPU/MLX parity for two or more layers.
3. Prepare a licensed corpus with document-level train/validation/test separation, deduplication rules, exact tokenizer revision, and hashes of splits. Record bytes and tokens; token counts across tokenizers are not directly comparable. Freeze validation before hyperparameter selection.
4. Start with a 1M–10M model and scale toward 30M only after memory and transport are measured. Add chunked or streamed model transfer before exceeding the current 2 MB response limit; see [transport](transport.md).
5. Introduce an optimizer abstraction. Momentum/Adam state needs explicit resume and averaging semantics. Matching model weights alone does not restore the same optimizer trajectory. Keep the current SGD path as the numerical oracle.
6. Add accumulation and activation checkpointing with equivalence tests. Set a safe working-set budget using observed memory pressure, not total RAM minus a guessed constant. Calibrate outside production training first.

## Memory accounting

Raw fp32 weights require 4P bytes for P parameters. Full training also needs gradients, activations, optimizer state, serialization buffers, and framework caches. A common mixed-precision Adam setup can require roughly 16 bytes per parameter before activations, depending on which master copies and dtypes are retained; calculate the actual implementation rather than treating that estimate as universal. The coordinator currently retains complete updates from all workers and float64 reduction temporaries, so it can hit memory limits first.

## Adapter path

MLX-LM already documents [LoRA and QLoRA](../sources/S19.md). A practical extension can wrap its supported model/training APIs instead of recreating every Qwen architecture. Pin model weights, tokenizer, runtime, and code revision; check licensing and actual peak training memory. Quantized inference fit does not establish training fit.

For a first distributed adapter experiment, use identical base-model hashes and a clearly defined aggregation operator. Independently averaging LoRA A and B factors generally differs from averaging the represented B×A updates. See [optimizer and adapter semantics](optimizers-adapters.md).

**Acceptance:** disjoint data, stable training, checkpoint/optimizer resume parity, bounded memory, and quality curves at matched tokens. Only then use this workload to judge scheduling or ANE speed.
