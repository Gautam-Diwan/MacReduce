---
title: "Run and review baselines without a live assistant"
type: implementation
status: active
updated: 2026-11-16
sources: []
---

# Run and review baselines without a live assistant

**Completion update, 2026-11-16:** both local M1 arms finished 12,288 updates and
full evaluation. See the [completed results](../../evidence/M1_8GB/README.md).
Launch/stop examples below retain their historical context. No restart is needed.
The M4 PR configs confirm the core comparison recipe, but code and runtime
differences prevent a hardware-only comparison. Proposal-wide coverage is incomplete.

Use [tools/run_baselines.py](../../tools/run_baselines.py) from a normal terminal
with the project environment. Codex is not required while it runs. It plans
without executing by default, runs available local arms sequentially with
`--run`, prints progress every 20 seconds, and saves `summary.json`,
`summary.csv`, per-arm configs, preflight reports, logs and checkpoint paths.
It refuses to overlap another active supervised run. Keep the terminal open;
Codex may be closed. On macOS the wrapper prevents idle system sleep while
running; it does not override lid-close sleep or change system settings.

```sh
.venv/bin/python tools/run_baselines.py
```

The default `comparison` scope uses the user's provisional M4-matching recipe:
first 4,096 nonempty WikiText records, three epochs, batch 1, accumulation 1,
checkpointing off, 512-token record cap, saved checkpoints every 500 updates. Verify the
actual M4 saved config before treating this as a hardware-only comparison.
Longer records are truncated by the existing loader. `--scope full` retains
the earlier 23,767-record recipe; neither adaptation is the exact proposal.

The previous comparison stopped before its first epoch-end checkpoint and
cannot resume. Its [logs and stop reason](../../evidence/m1-comparison-stop-2026-11-10/README.md)
are retained. For a fresh restart, choose a new output directory:

```sh
.venv/bin/python tools/run_baselines.py --scope comparison --run \
  --no-swap-growth-guard \
  --output runs/baseline-suites/comparison-no-swap-1
```

Once a run completes, `--existing-gpu PATH_TO_COMPLETED_RUN` can reuse its
results without training it again. The imported run must have a successful result and matching data/model/training
config. In-progress, failed, or mismatched runs are blocked, not substituted.
If the requested output directory exists, the runner preserves it and selects
a new sibling with `-2`, `-3`, etc., printing the actual path. Repeating a run
without a resume option starts fresh; it does not append to the old experiment.
To deliberately run a fresh
suite, omit `--existing-gpu` and use a new output path. The supervisor retains
the configured disk/memory checks and a default 48-hour per-arm ceiling.
Ctrl-C asks the supervisor to terminate only its own training processes.

The first long hybrid run stopped near update 607 when free disk fell below
10 GiB. At the user's request, `comparison-hybrid-resume-1` was launched with
a 5 GiB reserve and checkpoint 500. The saved policy confirms 5368709120 bytes;
other memory checks remain. Use `--min-free-disk-gib 5` for this override and
`--resume-gpu-ane` to resume the hybrid while importing the completed GPU arm:

```sh
.venv/bin/python tools/run_baselines.py --scope comparison --run \
  --no-swap-growth-guard --min-free-disk-gib 5 \
  --existing-gpu runs/baseline-suites/comparison-resume-2/local_gpu/logs/2026-11-10T15-15-52 \
  --resume-gpu-ane runs/baseline-suites/comparison-hybrid-1/local_gpu_ane/logs/2026-11-10T17-33-30/checkpoints/0000500 \
  --output runs/baseline-suites/comparison-hybrid-resume-1
```

That command has already been launched; inspect its logs before starting another
copy. Resume verifies the placement as well as the data/model/training recipe.
Initial evaluation is repeated; updates since checkpoint 500 are repeated.

To resume a matching incomplete run, use `--resume-gpu CHECKPOINT_DIRECTORY`
instead of `--existing-gpu` (which only imports completed results). The runner
requires the adapter, optimizer, and state files and matching data/model/training
settings. The trainer also validates its checkpoint identity when restoring.
Resumed timing describes only the new segment; retain prior segment logs for
end-to-end time. The currently inspected `comparison-no-swap-1` run was marked
interrupted and has a complete update-2,000 checkpoint:

```sh
.venv/bin/python tools/run_baselines.py --scope comparison --run \
  --no-swap-growth-guard \
  --resume-gpu runs/baseline-suites/comparison-no-swap-1/local_gpu/logs/2026-11-10T12-18-28/checkpoints/0002000 \
  --output runs/baseline-suites/comparison-resume-2
```

This restores progress through update 2,000; later unsaved updates are repeated.
The original output and checkpoint remain intact. Resume repeats the initial
full test evaluation before continuing training. The first manual resume failed
with `KeyError: 'embed_tokens'`; optimizer restoration now reconstructs empty
frozen-layer containers. A bounded real-checkpoint probe completed update 2,001
with finite loss, without changing the source checkpoint or starting a long run.
[Real recovery validation](../../evidence/m1-resume-fix-2026-11-10/README.md).
[Checkpoint inspection and validation scope](../../evidence/m1-checkpoint-recovery-2026-11-10/README.md).

The swap-growth stop is configurable through `safety.max_swap_growth_bytes` in
the machine config. If omitted, the supervisor defaults to 536870912 (512 MiB).
At the user's request, the comparison profile now explicitly sets it to `null`,
disabling only this stop. For example, 1073741824 instead allows
a 1 GiB increase in total system swap; it does not cap absolute swap or attribute
growth to the trainer. Increasing it does not reduce memory usage. Other guards
remain. Gradient checkpointing is separate from periodic model/optimizer saves.

The command above explicitly disables the swap-growth stop. Use
`--max-swap-growth-mib 1024` to re-enable it with a 1 GiB allowance. These flags
are mutually exclusive. Omitting both uses the machine config. The runner
prints the resolved policy before launch and saves it in each arm's config
and supervisor policy. Use a new output directory each time: the user's
`comparison-retry-1` already stopped at update 232 under the old 512 MiB value.
[Retry diagnosis](../../evidence/m1-comparison-retry-stop-2026-11-10/README.md).
Stopped-run summaries now retain the last reported training step, loss, tokens
and peak allocation, while final perplexity remains unavailable.

For the proposal's data/batch/accumulation settings, inspect:

```sh
.venv/bin/python tools/run_baselines.py --scope proposal
```

That scope uses 4,096 records, batch 8, accumulation 2, three epochs, full
validation/test splits, and the pinned Qwen3 model/data. The current M1 admission
policy requires batch 1, so the exact proposal recipe is blocked rather than
silently changed. The [proposal](../../docs/proposal.md) also requires FedAvg,
per-step all-reduce, combined/split clients, topology and payload variants,
and physical multi-Mac runs. A local continuous-training curve does not test
those algorithms. The coverage ledger lists those requirements and their
implementation/hardware blockers; it does not mark them measured.

The corrected hybrid arm now passes the M1 real-model normalization/gradient
gate at sequence cap 512 and a 20-update pilot. The runner uses the
[new validation report](../../evidence/m1-hybrid-validation-2026-11-10/README.md);
preflight requires its sequence cap to match the config. The three-epoch
comparison was launched under `runs/baseline-suites/comparison-hybrid-1`, reusing
the completed GPU result via `--existing-gpu` and starting hybrid training fresh
from the base model. Check its summary/logs before launching another copy.
The independent full-Qwen3 ANE client is missing.
No GPU fallback or unrelated tiny linear workload is substituted. Exit code 2
means the suite has incomplete/blocked coverage; inspect each arm's status to
distinguish a completed GPU run from a failed or blocked arm.

Reprint saved results without training:

```sh
.venv/bin/python tools/run_baselines.py --report runs/baseline-suites/comparison-no-swap-1
```

For later evaluation, share `summary.json` and `summary.csv`. A failure may
also need its `preflight.json`, `job/exit.json`, or the end of `job/stdout.log`.
Model weights, dataset content, and full telemetry files are unnecessary.
Energy is left unknown until power channel validity is reviewed. Timing from
one machine/run is not automatically a speedup or a complete proposal result.

See [the current comparison record](../experiments/2026-11-10-m1-comparison-switch.md),
[the stopped full-baseline record](../experiments/2026-11-10-full-gpu-baseline.md),
[the prior numerical failure](../experiments/2026-11-10-m1-training.md), and
[the single-machine workflow](single-machine.md).
