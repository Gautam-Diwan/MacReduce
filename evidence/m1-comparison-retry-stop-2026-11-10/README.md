# Manual retry stopped under the unchanged 512 MiB swap policy

The user's `comparison-retry-1` suite reached progress-bar update 232 of 12,288
and stopped after 963.06 seconds. The [copied run config](run-config.yaml) and
[resolved policy](guard-policy.json) both still select 536,870,912 bytes, so
the earlier documented 1 GiB example had not been applied. This is not a new
model/backend error or an observed OOM exception.

[Raw resources](resources.jsonl) show system swap growing from 1,766,588,416 to
2,477,064,192 bytes. Pressure was Warning (2); available memory at stop was
1,388,331,008 bytes. These system-wide changes are not all attributed to the
trainer. [Exit](exit.json) confirms the swap-policy stop. There is no checkpoint
before update 500, so this attempt cannot resume. Final test perplexity is
unavailable. All source-run files remain unchanged.

The [retrospective partial summary](retrospective-partial-summary.json) preserves
the last periodic report: update 200, mean training loss 3.4528, 21,329 tokens,
peak MLX allocation 2,185,501,596 bytes. It does not claim exactly 200 executed
steps or a completed evaluation; the progress bar separately reached 232.
The original suite summary is retained separately.

The runner now accepts an explicit command-line override and prints the chosen
allowance at launch. A [read-only plan test](tested-plan-override.json) confirms
that 1024 MiB reaches both arm configs without changing the training recipe,
the 10 GiB disk floor, or other guards. No larger-allowance training test or
automatic restart was performed. The config's default remains 512 MiB.

For a user-selected 1 GiB increase allowance and a fresh run directory:

```sh
.venv/bin/python tools/run_baselines.py --scope comparison --run \
  --max-swap-growth-mib 1024 \
  --output runs/baseline-suites/comparison-retry-2
```

This allows additional system paging; it is not a reduction in memory demand
or a guarantee that another resource guard will not stop the run. Critical
pressure, low available memory, disk, battery and time guards remain. Blocked
proposal coverage is independent of this GPU stop and is not the stop cause.
Validation: 130 tests passed, 5 skipped; project-code Ruff and wiki lint passed.
[Workflow](../../wiki/guides/baseline-suite.md).
