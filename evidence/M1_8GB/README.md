# Qwen3-0.6B QLoRA on an M1 MacBook Air, 8 GB

Audited 2026-11-16. **Measured:** both local comparison arms completed 12,288
updates, three epochs on the first 4,096 nonempty WikiText-2 records, batch 1,
accumulation 1, gradient checkpointing off, sequence cap 512, and full validation
(2,461 records) and test (2,891 records). Both supervisors exited 0 and both final
checkpoints record step 12,288. Earlier interrupted attempts remain historical
evidence; this report supersedes their incomplete status for these two arms.

## Results

| Measured | GPU | GPU + ANE |
|---|---:|---:|
| Initial full test perplexity | 64.233305 | 64.233305 |
| Final full test perplexity | 23.323357 | 23.323607 |
| Validation PPL, updates 4095 / 8191 / 12287 | 21.2919 / 22.0342 / 23.7330 | 21.3033 / 22.0447 / 23.7351 |
| Final reported train loss | 1.972738 | 1.969868 |
| Peak MLX allocation, bytes | 2,185,616,352 | 4,259,897,919 |
| Reported median step latency, ms | 448.337 | 572.477 |
| This invocation, seconds | 7,353.949 | 9,431.140 |
| Restored checkpoint | 2,000 | 500 |
| New updates in this invocation | 10,288 | 11,788 |

Sources: [GPU result](gpu/2026-11-10T15-15-52/result.json),
[hybrid result](gpu_ane/2026-11-10T22-28-06/result.json),
[suite summary](summary.json), [CSV](summary.csv).
Validation is logged immediately before the last update of each epoch by the
existing trainer. Validation perplexity rises after epoch one in both arms;
successful completion does not mean three epochs improve held-out quality.

Timings cover unequal resumed segments, repeated initial test evaluation, and
different background conditions. They are **not** comparable end-to-end training
times or a controlled hardware speedup. Peak MLX allocation is not total system
memory; the configured 2 GiB MLX budget is not a hard process cap.

## Placement and proposal coverage

The GPU arm records `placement: gpu`, `ane_lora: false`, and no ANE engine.
The hybrid uses GPU blocks 0–26 and one ANE suffix block (27), with 19,171 forward
calls and 11,788 backward calls in its resumed invocation. The backend checks
device mask `0x4` and fails rather than falling back. Its
[real-model numerical gate](../m1-hybrid-validation-2026-11-10/README.md) covers
forward output, LoRA gradients and the gradient returned to the GPU prefix.

**Unknown:** the saved M1 mactop ANE channel reports zero. This is not evidence
of zero ANE energy, and the mask/call counters are not an independently captured
Core ML compute plan or positive power trace. The proposal's strict power/compute
plan acceptance criterion remains unestablished. Raw result energy fields are
preserved, but accepted summary energy remains null; no energy ranking is made.

| Proposal requirement | This M1 contribution |
|---|---|
| Local GPU and combined GPU + ANE curves | Completed with the agreed adapted recipe |
| First 4,096 records, three epochs, full evaluation | Completed |
| Batch 8, accumulation 2 | Adapted to 1 / 1; not the exact proposal |
| One-GPU FedAvg, star, uncompressed delta | Not measured; these are local AdamW curves |
| Combined federated client | Executor integration missing |
| Separate full-Qwen GPU and ANE clients | Independent full-Qwen ANE trainer missing |
| Per-step all-reduce, ring/tree, int8/error feedback, FedNova | Not measured; federated client integration/schedules missing |
| Multiple physical Macs | Not measured; no second connected machine |

## Relationship to the M4 PR

Following [PR #1](https://github.com/Gautam-Diwan/MacReduce/pull/1), configs live
in `configs/M1_8GB/` and completed measurements in this machine directory. Inspected
PR head `30471852690b55476b25f8dafb7f8b6635ba7c4e`: both machine configs and its
evidence README (not every raw M4 measurement).
The [pinned M4 GPU config](https://github.com/Gautam-Diwan/MacReduce/blob/30471852690b55476b25f8dafb7f8b6635ba7c4e/configs/M4_16GB/qlora_gpu.yaml)
and [hybrid config](https://github.com/Gautam-Diwan/MacReduce/blob/30471852690b55476b25f8dafb7f8b6635ba7c4e/configs/M4_16GB/qlora_gpu_ane.yaml)
confirm the same model/data revisions, prefix, seed, LoRA settings, optimizer,
learning-rate schedule, batch, accumulation and checkpointing setting.

Differences beyond hardware: M1 has an explicit MLX budget, resource supervision,
30-second telemetry, reports every 100 rather than 10 steps, saves every 500 rather
than every epoch, and resumed runs. This branch also repairs ANE RMSNorm/factor
numerics, isolates batch RNG from evaluation/backend setup, and verifies pinned
dataset parquet hashes. The M4 PR changes configs/evidence without those code
repairs. Thus matching configuration does **not** establish a hardware-only
comparison. Re-run the M4 with this implementation for that claim.

## Reproduce and inspect

Use the [setup instructions](../../README.md) and
[baseline guide](../../wiki/guides/baseline-suite.md). Host used Python 3.12,
MLX 0.32.4, mlx-lm 0.32.2 and ANEForge 0.4.0; historical preflight records macOS
27.2.1. Shared code is portable; machine policy is in the configs.

```sh
# Read-only plan; no training.
.venv/bin/python tools/run_baselines.py --scope comparison

# Fresh sequential pair. Explicitly selects the last hybrid run's disk/swap policy.
.venv/bin/python tools/run_baselines.py --scope comparison --run \
  --no-swap-growth-guard --min-free-disk-gib 5 \
  --output runs/baseline-suites/m1-comparison-new
```

The runner uses `caffeinate -i`, prevents overlapping supervised jobs, and retains
critical memory-pressure, available-memory, battery and deadline checks. Default
configs retain a 10 GiB disk reserve; the command explicitly lowers it to 5 GiB.
Keep the lid open and AC connected. This command starts fresh and is not needed
to inspect the completed results. Resume commands are in the baseline guide.

## Files and validation

Each completed arm contains its original result, resolved config, history,
final checkpoint metadata, selected figures and supervisor exit/policy records.
[Artifact hashes](artifact-manifest.json) identify exact copies from local runs.
Model/optimizer weights, dataset content and process telemetry are excluded.
The repository copies alone cannot resume training; local ignored checkpoints
remain intact. Energy figures are omitted because that channel is unvalidated.

[Validation](validation.txt): 135 default tests passed, 7 opt-in tests skipped;
14 explicitly enabled ANE/GPU/checkpoint tests passed. Maintained Python code
passes Ruff. Whole-tree Ruff has 13 existing errors in immutable source snapshots.
No long training was rerun during this completion audit.
