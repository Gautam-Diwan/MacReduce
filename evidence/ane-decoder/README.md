# Decoder placement, 2026-10-25

One M4 (Mac16,12), 24 GiB, macOS 26.5. `coremltools==9.0`, `torch==2.14.1`, `mactop` v2.1.5. Command:

```sh
macreduce ane-workload --output evidence/ane-decoder/result.json
```

The workload is one decoder block: batch 1, 128 tokens, width 256, four heads, MLP width 1024, float16. The largest absolute difference from the float32 PyTorch forward was 0.00282.

**Seconds per forward.** CPU-only Core ML 0.000176 s. CPU+Neural Engine 0.000170 s. GPU via PyTorch MPS 0.000370 s.

**Why the ranking omitted the Neural Engine.** The compute plan assigns the linear and matmul ops to `MLNeuralEngineComputeDevice` under `CPU_AND_NE`, and to `MLCPUComputeDevice` under `CPU_ONLY`. mactop during the Neural Engine loop had two samples at or above 1 GB/s (14.15 GB/s at 1.51 W, then 14.30 GB/s at 1.52 W). The later samples in that capture fell to about 0.04 GB/s. The CPU-only loop stayed at 0 W and 0 GB/s. The gate that wrote this file required three such samples, so `placement_verified` is false and `choose_device` ranked CPU against GPU only. The Neural Engine time was kept in `cpu_and_ne` and left out of `choice.measured`.

The gate now accepts two samples. This file is the run from before that change.
