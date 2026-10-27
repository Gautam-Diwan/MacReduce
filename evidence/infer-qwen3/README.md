# Full inference, 2026-10-25

One M4 (Mac16,12), 24 GiB, macOS 26.5. Command:

```sh
.venv/bin/python -m macreduce infer --output evidence/infer-qwen3/result.json
```

**Pinned Qwen3 generation.** Target `mlx-community/Qwen3-1.7B-4bit` at `3b1b1768f8f8`, draft `mlx-community/Qwen3-0.6B-4bit` at `73e3e38d9813`. Prompt `Reply with one word: ready`, 32 tokens. Plain decode was 83.0 tok/s, time to first token 0.086 s. Speculative depth 2 was 92.7 tok/s, time to first token 0.058 s, 17 tokens from the draft, and the tokens matched the plain run.

**28-layer Core ML forward.** The network has the Qwen3-0.6B widths: 28 layers, hidden 1024, 16 query heads, 8 key/value heads, head dim 128, MLP 3072, vocabulary 151936, 596,042,752 parameters. Weights are random, scaled by 0.02. Input is 64 hidden states. Output is the next-token logits. A run that still had room to compile placed 1,213 ops on `MLNeuralEngineComputeDevice` and 1 op on `MLCPUComputeDevice`, and three CPU+Neural Engine predictions took 0.0207 s each. That package was not retained.

`macreduce infer` then failed while writing the compiled weights: `[MIL FileWriter]: Unknown error occurred while writing data to the file.` A later compile reported that `weight.bin` could not be copied because the volume was out of space. The data volume had under 1 GB free. The float16 weights are about 1.1 GB and the compiler writes a second copy, so the command now refuses to start this conversion below 3 GB free. The Qwen3 checkpoint itself was not converted.
