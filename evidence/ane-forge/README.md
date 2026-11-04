# ANEForge linear step

`macreduce ane-forge` imports `aneforge` as `af` (0.4.0) and trains a 16-by-1 linear map for four Adam steps. The forward, backward, and update programs are compiled with device mask `0x4`, which that package defines as the Apple Neural Engine. A 256-wide matmul then runs for the mactop window.

On this M4 the window showed Neural Engine power up to about 0.79 W and combined Neural Engine bandwidth up to about 2.9 GB/s. `ane_active` stayed 0. mactop is system-wide, not per-process. The numbers are in `result.json`. This is not the Qwen3 QLoRA client.
