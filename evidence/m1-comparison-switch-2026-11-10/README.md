# Switch from the full-data run to a 4,096-record comparison

The user explicitly requested this switch to approach the reported M4 recipe.
The M4 saved config has not been inspected, so the match is **provisional**.
The M4 report supplied batch 1, about 4K records, checkpointing off; accumulation
1 and the shared remaining recipe are assumptions pending that config.

[switch.json](switch.json) records the old run's last training report and new
scope. The full-data supervisor exited cleanly after SIGTERM with
`stop_reason: interrupted`; [old-exit.json](old-exit.json) records this.
Its last periodic report was update 800. It had not reached its first
checkpoint at update 5,000. Logs remain in the original ignored run directory;
unsaved model/optimizer state was not carried into the replacement. No files
were deleted or overwritten.

[new-launch.json](new-launch.json) records a new supervised GPU job started at
2026-11-10 05:08:41 America/New_York. [comparison-config.yaml](comparison-config.yaml)
uses the first 4,096 nonempty training records, three epochs (12,288 updates),
batch 1, accumulation 1, checkpointing off, and a 512-token record cap. The
model is the pinned Qwen3-0.6B 4-bit base with rank-8 LoRA/scale 2. Resume is
null. Full validation/test splits remain selected, and checkpoints save each
epoch. The AdamW schedule is recomputed for the new total step count.

The new run is `runs/M1_8GB/gpu-m4-comparison/2026-11-10T05-08-42/`.
It is launched, not completed; no training-quality or final-time claim is made.
At inspection, full initial test evaluation was progressing with normal
memory pressure, no swap growth from the new baseline, and AC power.
The supervisor retains its memory, swap, 10 GiB disk, battery, and time guards.
No ANE training or unrelated process was started/stopped.

```sh
tail -f runs/M1_8GB/comparison-supervisor-1/stdout.log
```

After completion, summarize the finished run without repeating training:

```sh
.venv/bin/python tools/run_baselines.py --run \
  --existing-gpu runs/M1_8GB/gpu-m4-comparison/2026-11-10T05-08-42 \
  --output runs/baseline-suites/m1-comparison-review
```

The runner defaults to `--scope comparison`. `--scope full` retains the earlier
full-split recipe; `--scope proposal` retains batch 8/accumulation 2 and its
explicit M1 admission blockers. This local comparison is not a FedAvg run or
proof that all M4 settings match. Share the saved M4 config, software versions,
and data fingerprints before attributing differences solely to hardware.

Validation: [121 passed, 5 skipped](tests.txt); project-code Ruff and structural
wiki lint passed. Hardware skips do not validate accelerator execution.
[Maintained switch report](../../wiki/experiments/2026-11-10-m1-comparison-switch.md).
