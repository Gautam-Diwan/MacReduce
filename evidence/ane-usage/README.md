# Neural Engine activity, 2026-10-25

One M4 (Mac16,12), 24 GiB, macOS 26.5. `coremltools==9.0`, `torch==2.14.1`, `mactop` v2.1.5. Command:

```sh
.venv/bin/python -m macreduce ane --output evidence/ane-usage/result.json
```

The network is two float16 convolutions, 3 to 128 and 128 to 128, kernel 3, each followed by ReLU. Input shape `1×3×512×512`. It is not the Qwen draft. `coremltools` reports Torch 2.7.0 as the newest version it has tested; conversion still produced an mlprogram.

**Plan:** under `CPU_AND_NE`, both `ios17.conv` operators and both `ios16.relu` operators are assigned to `MLNeuralEngineComputeDevice`. Under `CPU_ONLY`, the same operators are assigned to `MLCPUComputeDevice`.

**mactop, system-wide estimates.** `powermetrics` still returns `powermetrics must be invoked as the superuser`. Each phase sampled for about 8 seconds at 200 ms.

- Idle peaked at 0.0095 W and 0.042 GB/s.
- `CPU_ONLY` ran 198 predictions in 8.04 s. All 40 samples were 0 W and 0 GB/s.
- `CPU_AND_NE` ran 424 predictions in 8.00 s. Nine consecutive samples, from 01:49:56 to 01:49:59 local time, read 0.581–0.802 W and 10.23–11.89 GB/s. Later samples in that same capture fell back to the idle floor. `ane_active` stayed 0 in every sample.

Twenty predictions outside that capture took 0.368 s under `CPU_AND_NE` and 0.805 s under `CPU_ONLY`. Wall time is not residency. The bandwidth and watt samples are. They are not attributed to this process, and they do not place the serving draft on the Neural Engine. `placement_verified` stays false for that draft.
