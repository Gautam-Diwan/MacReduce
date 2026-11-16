# Resumed GPU baseline launch, 2026-11-10

At the user's request, started the standalone comparison suite from the real
update-2,000 checkpoint after the [recovery fix](../m1-resume-fix-2026-11-10/README.md).
No other supervised trainer was active before launch; free disk was 15.72 GiB.

Command from the canonical project root:

```sh
.venv/bin/python tools/run_baselines.py --scope comparison --run \
  --no-swap-growth-guard \
  --resume-gpu runs/baseline-suites/comparison-no-swap-1/local_gpu/logs/2026-11-10T12-18-28/checkpoints/0002000 \
  --output runs/baseline-suites/comparison-resume-2
```

Measured scope: preflight passed, full initial evaluation completed, the trainer
restored step 2,000 and advanced beyond step 2,054 with no exception or resource
stop at observation. [Raw observation](observation.json). This is a launch and
recovery confirmation, not a completed three-epoch result or final perplexity.

The job was left running in an independent session. Launcher PID 43583, launcher
log `runs/baseline-suites/launch-resume-20261110T151552/stdout.log`. Suite outputs
are under `runs/baseline-suites/comparison-resume-2`; training log is
`local_gpu/job/stdout.log`, and the actual run is
`local_gpu/logs/2026-11-10T15-15-52`. Status may change after this snapshot.
The earlier source checkpoint and failed-run artifacts are preserved.

Recipe remains first 4,096 records, three epochs, batch/accumulation 1,
gradient checkpointing off, sequence cap 512, same pinned GPU QLoRA model and
data. Swap-growth stop is disabled by user choice; other configured checks
remain. No training code changed in this launch. The user's newly reported
error was not provided and was not reproduced by this command. ANE and proposal
coverage blockers remain as documented in the [standalone guide](../../wiki/guides/baseline-suite.md).
