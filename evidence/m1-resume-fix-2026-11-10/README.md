# Real Qwen checkpoint resume fix, 2026-11-10

Measured on the user's M1 8GB, macOS 27.2.1, MLX 0.32.4 / MLX-LM 0.32.2.
The manual `comparison-resume-1` run failed at its first resumed update with
`KeyError: 'embed_tokens'`, after successfully loading checkpoint 2,000.
Its original log is retained under
`runs/baseline-suites/comparison-resume-1/local_gpu/job/stdout.log`.

Inspected scope: that traceback, `load_checkpoint`, MLX Optimizer initialization,
and the checkpoint regression test. Flattening optimizer tensors discards empty
containers for frozen modules. Restore assigned `_initialized = True`, preventing
MLX from reconstructing these containers before mapping gradients. Restore now
calls the public `optimizer.init(model.trainable_parameters())`, preserving loaded
moments and the step while rebuilding the missing structure.

The previous [inspection report](../m1-checkpoint-recovery-2026-11-10/README.md)
tested a fully trainable linear model and inspected checkpoint files; it did not
validate real Qwen recovery. This report extends that validation and records the
failure its test missed. Historical evidence remains unchanged.

Measured recovery: [probe.py](probe.py) uses cached pinned Qwen weights, the real
update-2,000 checkpoint, and the normal compiled trainer. It reconstructs batch
order at seed 0 on the first 4,096 records and completes only update 2,001.
Recipe: batch 1, accumulation 1, sequence cap 512, no gradient checkpointing,
three-epoch/12,288-update schedule, rank 8 scale 2 LoRA and original AdamW settings.
Validation is omitted for this bounded probe, and the report callback stops it
before any new checkpoint save. No long job was launched.

[Raw result](result.json): loss 2.6632692814, learning rate 0.0000951741,
56 tokens in this update, peak MLX allocation 679,633,674 bytes.
These are one-update diagnostics, not final perplexity or throughput results.
The source checkpoint's three hashes matched before and after. [Raw log](probe.log).

Reproduce from the project root (writes probe artifacts only):

```sh
HF_HOME="$PWD/runs/M1_8GB/huggingface" HF_HUB_OFFLINE=1 PYTHONPATH=src \
  .venv/bin/python evidence/m1-resume-fix-2026-11-10/probe.py
```

Regression tests now cover nested frozen embeddings and a compiled next update
matching uninterrupted AdamW parameters, moments and step. Metal-enabled checkpoint
tests: [8 passed](hardware-tests.txt). Default suite: [133 passed, 6 skipped](tests.txt).
The first sandboxed attempt could not bind loopback sockets; the successful full
suite ran with that access. Sandbox Metal initialization was also unavailable;
hardware results above used device access. Existing SciPy/NumPy warning remains.
Project code and probe Ruff pass; whole-tree Ruff retains 13 preexisting errors
in immutable raw source snapshots. Structural wiki lint and whitespace checks pass.

Retry instructions and limitations: [standalone guide](../../wiki/guides/baseline-suite.md).
Full training, final evaluation, M4 recipe equivalence and ANE correctness remain
unvalidated by this recovery probe.
