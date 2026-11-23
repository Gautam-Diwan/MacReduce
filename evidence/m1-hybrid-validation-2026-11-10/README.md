# M1 hybrid numerical repair and baseline launch, 2026-11-10

Measured on M1 8GB, macOS 27.2.1, MLX 0.32.4, MLX-LM 0.32.2,
ANEForge 0.4.0. This extends the [failed pilot](../m1-training-2026-11-10/README.md)
and supersedes its launch blocker for the newly validated implementation only.
Earlier measurements remain unchanged. Inspected scope: real saved activations,
the installed ANEForge RMSNorm emitter/VJP, the suffix bridge, saved adapters,
and the actual probe/pilot reports linked below.

## Numerical changes and evidence

The original ANE RMSNorm native L2 forward returned zero on the real large
activations. [Scaling probe](normalization-scaling.json) compared forward and
backward with float64 RMSNorm, including zeros: dividing the input by 128 fixed
the large-input forward, but scaling the automatic backward also degraded the
zero-input result. [Forward-only lowering](normalization-forward-fix.json)
retains the original backward input/epsilon. Large-input forward relative L2
error was 0.000464 and backward error 0.000511; zero-input backward error 0.00651.
The original backend's epsilon-floor approximation is retained; this is not a
claim of exact RMSNorm equivalence for all very small inputs.

The new local `macreduce_scaled_rms_norm` operator uses the pinned ANEForge
emitter/VJP registration APIs. It rescales the forward L2 reduction by 128 and
adjusts the forward epsilon floor, while differentiating in the original
coordinates. It does not patch the installed package or globally replace
Tensor.rms_norm. The normalization-only candidate improved full-block forward
agreement but [still failed small adapter gradients](norm-only-parity-failed.json).

LoRA upload now uses A/16 and B*16, preserving A@B. Returned gradients use the
corresponding chain rule to the unchanged master parameters and checkpoint
coordinates. This lifts small B factors before fp16 conversion. The combined
candidate [passed the original 128-token gradient gate](factor-parity-128.json).
The evidence supports this correction on tested cases, not a universal account
of every precision error or a measurement on M4.

The integrated production path passed stronger checks on three partial-length
sequences at adapter steps 0, 2,500 and 12,288: [512-token program](parity-512.json).
It also passed on full 512-token sequences and a padded 165-token sequence:
[boundary gate](boundary-parity-512.json). Every nonzero adapter gradient and
the gradient back to the GPU prefix must have cosine >=0.99 and relative L2
error <=0.15; forward requires cosine >=0.99 and relative L2 <=0.02. The actual
boundary forward errors were 0.00366–0.00458 and hidden-gradient errors
0.01010–0.01091. All compiled forward/backward programs were required to have
device mask 0x4. Reference: the MLX quantized block, not an independent float64
whole-decoder oracle. [Implementation hashes](implementation-sha256.json).

One boundary probe using an unpadded very short heading was stopped by the MLX
reference attention VJP (vector kernel lacks logsumexp), before reporting parity.
The successful retry used a 165-token record padded to 512 plus full sequences.
That unpadded short reference case is not counted as passing. An initial small
diagnostic also used the wrong input-list attribute; its corrected run used
the installed Model._input_tensors. Diagnostic scripts and all attempt logs
are retained under `runs/M1_8GB/ane-normalization-retry`.

## End-to-end pilot

[Config](pilot.yaml), [result](pilot-result.json), [history](pilot-history.json),
[supervisor exit](pilot-exit.json): 20 updates, batch/accumulation 1, checkpointing
off, 512-token cap, first 4,096 records, same pinned model/data and LoRA rank/scale.
The short smoke schedule uses warmup 2 and only two validation/test batches;
it is not the long comparison recipe. Sampled test PPL 117.1804 -> 59.5148,
peak MLX 1,845,908,234 bytes, 24 ANE forwards and 20 backwards, 27 GPU blocks
and block 27 on ANE. Exit 0, no resource stop, 24.19 seconds supervised elapsed.
This establishes bounded learning/execution, not a full-dataset quality result.

Default suite: [134 passed, 7 skipped](tests.txt); separate device tests:
[6 passed](hardware-tests.txt), including normalization range/zero gradients
and the existing decoder reference test. CPU tests establish the factor
rescaling/chain-rule invariant. Project-code Ruff and whitespace checks pass;
[whole-tree Ruff](ruff-all.txt) retains 13 preexisting errors in immutable
source snapshots. The existing SciPy/NumPy warning remains. Structural wiki
lint cannot certify the numerical interpretation.

## Long comparison launch

Started at 17:33:30 America/New_York with launcher PID 57790, independent of
Codex. It imports the completed GPU run and starts a fresh ANE+GPU arm from the
base model: first 4,096 records, three epochs / 12,288 updates, batch 1,
accumulation 1, checkpointing off, 512-token cap, full validation/test splits,
original warmup 300/cosine schedule. [Machine config](../../configs/M1_8GB/gpu_ane_m4_comparison.yaml).
Swap-growth stop is disabled per user choice; disk/memory checks remain.

```sh
.venv/bin/python tools/run_baselines.py --scope comparison --run \
  --no-swap-growth-guard \
  --existing-gpu runs/baseline-suites/comparison-resume-2/local_gpu/logs/2026-11-10T15-15-52 \
  --output runs/baseline-suites/comparison-hybrid-1
```

Launcher log: `runs/baseline-suites/launch-hybrid-20261110T173330/stdout.log`.
Suite: `runs/baseline-suites/comparison-hybrid-1`; final summaries will combine
the saved GPU result and hybrid result. Launch/preflight and the initial
evaluation were observed, not completion of the long run. The GPU duration is
from a resumed segment and cannot be compared as full wall time without its
earlier segment. ANE-only, federated training and multi-Mac cases remain blocked.
Energy channels require validation before an efficiency claim.

See the [standalone guide](../../wiki/guides/baseline-suite.md) and
[current project state](../../wiki/project/state.md).
