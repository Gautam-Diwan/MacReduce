# Hybrid retry after closing other apps, 2026-11-10

At the user's request, restarted the hybrid from checkpoint 500 after they
reported closing memory-heavy apps. No supervised trainer was active before
launch; free disk was 13.05 GiB and all three checkpoint files were present.
Retained the comparison recipe, 5 GiB disk reserve, disabled swap-growth stop,
and other memory checks. Reused the completed GPU result.

Launch: 22:28:06 America/New_York, PID 71325. Suite:
`runs/baseline-suites/comparison-hybrid-resume-2`. Launcher log:
`runs/baseline-suites/launch-hybrid-retry-20261110T222806/stdout.log`.

Measured: initial evaluation finished, checkpoint 500 was restored, and hybrid
training advanced beyond update 578. The previous retry stopped at update 501
under critical memory pressure. The new observation records normal pressure
(level 1) and approximately 1.8 GiB available system memory. [Raw observation](observation.json).
This is a progress snapshot; the process was left running and completion is
not established. No training code or resource policy changed for this retry.

See the [resume command](../../wiki/guides/baseline-suite.md) (change output to
`comparison-hybrid-resume-2`) and [hybrid validation](../m1-hybrid-validation-2026-11-10/README.md).
