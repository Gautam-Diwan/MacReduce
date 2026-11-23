"""Tiny explicit GPU execution probe; no model weights or dataset."""

import json
import platform
import sys
from importlib.metadata import version

import mlx.core as mx

if not mx.metal.is_available():
    raise RuntimeError("Metal GPU is unavailable; no CPU fallback")
mx.set_memory_limit(256 * 1024**2)
mx.set_cache_limit(32 * 1024**2)
mx.set_default_device(mx.gpu)
result = (mx.ones((16, 16)) @ mx.ones((16, 16))).sum().item()
if result != 4096.0:
    raise RuntimeError(f"unexpected matrix result: {result}")
print(json.dumps({
    "status": "ok",
    "macos": platform.mac_ver()[0],
    "architecture": platform.machine(),
    "python": platform.python_version(),
    "mlx": version("mlx"),
    "mlx_metal": version("mlx-metal"),
    "device": str(mx.default_device()),
    "dtype": "float32",
    "shape": [16, 16],
    "result": result,
    "peak_mlx_bytes": mx.get_peak_memory(),
    "scope": "tiny GPU execution only; no QLoRA, ANE, throughput, or power measurement",
}, indent=2))
sys.stdout.flush()
