# Core ML compute-plan probe, 2026-10-25

One M4 (Mac16,12), 24 GiB, macOS 26.5. `coremltools==9.0`, `torch==2.14.1`. Command:

```sh
.venv/bin/python -m macreduce ane --output evidence/ane-probe/result.json
```

The network is a float16 3-to-64 convolution with a 3×3 kernel and a ReLU, input shape `1×3×256×256`. It is not the Qwen3 draft. `coremltools` reports Torch 2.7.0 as the newest version it has tested; conversion still produced an mlprogram on Torch 2.14.1.

**Measured:** with `CPU_AND_NE`, the compute plan assigns `ios17.conv` and `ios16.relu` to `MLNeuralEngineComputeDevice`. With `CPU_ONLY`, the same operators are assigned to `MLCPUComputeDevice`. Constant operators have no preferred device. Twenty predictions took 0.0377 s under `CPU_AND_NE` and 0.0375 s under `CPU_ONLY`. Those wall times do not show which device ran.

`powermetrics -s ane_power` returned `powermetrics must be invoked as the superuser`. `ane_power_watts` is null. `placement_verified` stays false because the serving draft was not converted.
