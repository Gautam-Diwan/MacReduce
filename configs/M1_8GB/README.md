# M1 MacBook Air, 8GB

Both local comparison arms completed 12,288 updates and full validation/test.
See the [measured results and limits](../../evidence/M1_8GB/README.md), audited
2026-11-16. Earlier interrupted runs and failed pilots remain historical evidence.

| Profile | Purpose and measured status |
| --- | --- |
| `gpu_m4_comparison.yaml` | Completed GPU curve: first 4,096 records, three epochs |
| `gpu_ane_m4_comparison.yaml` | Completed hybrid curve with corrected ANE numerics |
| `gpu_smoke.yaml`, `gpu_ane_smoke.yaml` | Historical 10-update profiles; old hybrid validation report intentionally fails |
| `gpu_full_baseline.yaml` | Earlier full-dataset request; interrupted and superseded by the 4,096-record comparison |

The comparison uses batch 1, accumulation 1, gradient checkpointing off, sequence
cap 512, pinned model/data, seed 0, rank-8 LoRA, AdamW and a cosine schedule.
Full evaluation uses 2,461 validation and 2,891 test records. Checkpoints save
every 500 updates; checkpoint saving is distinct from gradient checkpointing.
The M4 PR uses the same core recipe, but implementation/resource differences
mean this is not yet a controlled hardware-only comparison.

Shared code lives in `src/macreduce`; this directory holds machine policies.
The hybrid runs blocks 0–26 on GPU and block 27 on ANE; it is not a standalone
ANE client. The separate full-Qwen ANE trainer and federated proposal experiments
remain missing. The corrected hybrid requires the passing real-block validation
report at the same sequence cap. Do not substitute the failed historical gate.

Run from the project root after the [repository setup](../../README.md):

```sh
# Read-only readiness and plan.
PYTHONPATH=src .venv/bin/python main.py --config configs/M1_8GB/gpu_m4_comparison.yaml --preflight
.venv/bin/python tools/run_baselines.py --scope comparison

# Fresh sequential GPU and hybrid runs, using the selected 5 GiB disk policy.
.venv/bin/python tools/run_baselines.py --scope comparison --run \
  --no-swap-growth-guard --min-free-disk-gib 5 \
  --output runs/baseline-suites/m1-comparison-new
```

No rerun is necessary to inspect the completed measurements. Follow the
[standalone baseline guide](../../wiki/guides/baseline-suite.md) for imports,
resumes, progress and reporting. It uses `caffeinate -i`; keep the lid open and
AC connected. Only one supervised arm runs at a time.

Comparison defaults disable the swap-growth stop at the user's request and
retain a 10 GiB disk reserve; the command above explicitly lowers it to 5 GiB.
Critical-pressure, low-available-memory, battery and time checks remain enabled.
The 2 GiB MLX allocator/wired policy is not a hard total-memory cap: measured
hybrid MLX peak was about 4.26 GB. These checks cannot guarantee no memory pressure.
The original 15 GiB admission and failed 512 MiB swap-growth attempts are recorded
in [historical evidence](../../evidence/m1-preflight-2026-11-10/README.md).

Outputs, model weights, datasets and caches remain under ignored `runs/`; never
commit weights or dataset content. Complete local checkpoints remain available
for recovery, but published evidence contains metadata only. Earlier failed
hybrid results are superseded by the [numerical repair](../../evidence/m1-hybrid-validation-2026-11-10/README.md)
and completed comparison, not relabeled as successes.
