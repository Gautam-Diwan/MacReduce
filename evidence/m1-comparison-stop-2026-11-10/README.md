# Incomplete comparison stopped by swap-growth policy

The comparison stopped at progress-bar update 3,131 of 12,288 after 3,015.81
seconds. [exit.json](exit.json), [history.jsonl](history.jsonl), and
[resources.jsonl](resources.jsonl) retain the raw scope. The last periodic
training report is update 3,100. There is no final score or completed epoch.

[stop-summary.json](stop-summary.json) distinguishes this from an observed
out-of-memory exception: the last memory-pressure value was Warning (2), with
991,772,672 bytes available. System swap grew from 1,407,975,424 to
2,044,329,984 bytes, exceeding the then-hardcoded 512 MiB increase threshold.
This is shared system telemetry, so the growth is not all attributed to the
training process. The stop is a policy outcome, not proof of hardware capacity.

The saved config selected epoch-end checkpoints at update 4,096. No checkpoint
exists, so this run cannot resume at update 3,131; it must restart from the
pinned base model. Logs/configs are preserved and are not overwritten.

Following the user's question, the supervisor now reads
`safety.max_swap_growth_bytes`, with the existing 536,870,912-byte default.
The default has not been raised. To choose a 1 GiB increase allowance, edit
the existing safety block in `configs/M1_8GB/gpu_m4_comparison.yaml`:

```yaml
  max_swap_growth_bytes: 1073741824
```

Critical-pressure, low-available-memory, disk, battery and time stops remain.
The threshold controls an increase in total system swap, not the 512-token
sequence cap or a memory allocation limit. Raising it permits more paging
and possible slowdown; it does not reduce the workload's memory requirements.

Restart from a normal project terminal using a fresh suite directory:

```sh
.venv/bin/python tools/run_baselines.py --scope comparison --run \
  --output runs/baseline-suites/comparison-retry-1
```

No restart was launched during this inspection. The comparison config now
saves model/optimizer state every 500 updates, independently of gradient
checkpointing (which remains false). This changes checkpoint I/O cadence,
not optimizer updates; record that cadence in timing comparisons. All stored
checkpoints stay within this experiment's run directory; none were deleted.

Batch size is already 1. Increasing it usually increases activation memory;
accumulation does not reduce a batch-1 peak. Gradient checkpointing true can
reduce activation memory at a compute cost, but would make this run differ
from the reported M4 settings. The real M4 saved config is still uninspected.
Validation: [128 tests passed, 5 skipped](tests.txt); project-code Ruff and wiki
structural checks passed. The inherited SciPy/NumPy warning remains. No hardware
run was used to validate a larger swap allowance, and no allowance increase was applied.
[Workflow](../../wiki/guides/baseline-suite.md),
[comparison report](../../wiki/experiments/2026-11-10-m1-comparison-switch.md).
