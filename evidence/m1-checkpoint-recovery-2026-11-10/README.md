# Output collision and resume preparation

The requested output `comparison-no-swap-1` already contained an interrupted
run; a repeated invocation failed at directory creation before starting any
training. The old directory and all files were preserved.

Inspected a complete checkpoint at update 2,000. The source exit says
`interrupted`, and its last periodic training report was at update 2,100.
[inspection.json](inspection.json) records checkpoint-file hashes and the scope
of the checks. Saved data/model/training config matches the requested
comparison. The real 0.6B checkpoint was not restored on hardware during this
inspection; successful restoration is still checked by the trainer at resume.

The standalone runner now chooses `-2`, `-3`, etc. when an output name exists,
and records the actual output directory. `--resume-gpu` checks a complete,
matching checkpoint and sends its path to the existing trainer. Existing
results/checkpoints remain intact; no large training or resume was launched.

Default suite validation: [133 passed, 5 skipped](tests.txt). Separately enabled
hardware command:

```sh
PYTHONPATH=src MACREDUCE_TEST_MLX=1 .venv/bin/python -m pytest -q tests/test_checkpoint.py::test_checkpoint_restores_adapter_and_adamw
```

Observed output: `1 passed in 1.11s`. This tests adapter and AdamW restoration
on a tiny 4×3 model; it is not full-Qwen3 resume validation. Project-code Ruff,
wiki lint, and whitespace checks passed. Existing immutable-source Ruff
violations and the inherited SciPy/NumPy warning were not changed.

Use [the standalone guide](../../wiki/guides/baseline-suite.md) for the actual
resume command. Resumed segment time excludes time spent in earlier segments.
