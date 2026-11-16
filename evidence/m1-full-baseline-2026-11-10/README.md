# Full-data GPU baseline launch, not completion

The user selected three full epochs, batch 1, checkpointing off. The planned
local GPU run has 71,301 updates over 23,767 nonempty training records, a
512-token cap with truncation, full validation/test evaluation, and accumulation
1. This differs from the exact proposal recipe and does not execute FedAvg.

[launch-summary.json](launch-summary.json) records the planned scope and the
preliminary runtime estimate. [fit-result.json](fit-result.json) records ten
updates with checkpointing off over the full split (sampled evaluation only).
[max-length-result.json](max-length-result.json) records three full-length
updates: peak MLX 2.035 GiB, close to and slightly above the configured 2 GiB
allocator budget. The budget is not a strict process memory bound.

[baseline-config.yaml](baseline-config.yaml) is the full-run recipe;
[launch.json](launch.json) records the background process launch at
2026-11-10 04:44:55 America/New_York. The ongoing run is under
`runs/M1_8GB/gpu-full-baseline/2026-11-10T04-44-55/`. No final result is archived
here. Do not mistake the short fit report for full-dataset completion.

Live status:

```sh
tail -f runs/M1_8GB/full-baseline-supervisor-1/stdout.log
```

The supervisor has a 48-hour deadline, 10 GiB disk floor, critical-pressure and
swap-growth checks, and a low-battery stop. It saves resource samples and
`exit.json` on completion/stop. On macOS `caffeinate -i` prevents idle system
sleep for the job; keeping the laptop on AC with the lid open is still needed.
The job does not require Codex to stay open. To stop only this supervised job:

```sh
kill -TERM "$(cat runs/M1_8GB/full-baseline-supervisor-1/supervisor.pid)"
```

The stop handler terminates the owned training process group and records the
stop. It does not stop other apps. Checkpoints save every 5,000 updates; the
latest complete checkpoint is the recovery point after an interruption.

The [standalone runner guide](../../wiki/guides/baseline-suite.md) explains how
to reuse the finished GPU run, run available later baselines sequentially, and
send only JSON/CSV summaries for review. The plans here show [full M1 scope](suite-full-plan.json)
and [proposal scope](suite-proposal-plan.json). Neither plan is a measurement.
ANE correctness and federated coverage remain explicitly blocked.

[Maintained report](../../wiki/experiments/2026-11-10-full-gpu-baseline.md).
