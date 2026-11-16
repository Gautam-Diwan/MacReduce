"""Read-only checks; never import accelerator runtimes or download model weights.

Readiness is permission to attempt a bounded smoke run, not proof of fit or
ANE residency. The memory allowance covers MLX allocations, not the process,
ANE compiler, OS, or other applications.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

GIB = 1024**3
PINS = {"mlx": "0.32.4", "mlx-lm": "0.32.2"}


def inspect_host(output: Path) -> dict:
    """Collect only hardware fields needed for admission; omit machine identifiers."""

    memory = None
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            memory = int(result.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        pass
    if memory is None and platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType", "-json"],
                capture_output=True, text=True, timeout=10,
            )
            hardware = json.loads(result.stdout)["SPHardwareDataType"][0]
            size, unit = hardware["physical_memory"].split()
            memory = int(float(size) * {"GB": GIB, "TB": 1024 * GIB}[unit])
        except (OSError, ValueError, KeyError, IndexError, subprocess.TimeoutExpired):
            pass
    target = output.resolve()
    while not target.exists():
        target = target.parent
    packages = {}
    for name in (*PINS, "aneforge"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {
        "system": platform.system(),
        "architecture": platform.machine(),
        "macos": platform.mac_ver()[0],
        "python": list(sys.version_info[:3]),
        "memory_total_bytes": memory,
        "disk_free_bytes": shutil.disk_usage(target).free,
        "packages": packages,
        "mactop": shutil.which("mactop") is not None,
    }


def check_config(config: dict, host: dict | None = None) -> dict:
    """Return blockers explicitly; missing telemetry or hardware is never a pass."""

    output = Path(config["logging"]["dir"])
    host = inspect_host(output) if host is None else host
    blockers = []
    mode = config.get("placement", {}).get("mode", "gpu")
    if mode not in {"gpu", "gpu_ane"}:
        blockers.append("ANE-only/split Qwen3 training is not implemented in this runner")
    if host["system"] != "Darwin" or host["architecture"] != "arm64":
        blockers.append("MLX training requires a native arm64 Apple Silicon Mac")
    try:
        os_version = tuple(int(part) for part in host["macos"].split("."))
    except ValueError:
        os_version = ()
    # Minimum in the MLX installation guide; exact pinned wheels may need newer macOS.
    if os_version < (14,):
        blockers.append("macOS 14 or newer is required; exact pinned wheels may require newer")
    if tuple(host["python"]) < (3, 12):
        blockers.append("Python 3.12 or newer is required")
    required = {**PINS, **({"aneforge": "0.4.0"} if mode == "gpu_ane" else {})}
    for name, pin in required.items():
        if host["packages"].get(name) != pin:
            blockers.append(f"{name}=={pin} is required (found {host['packages'].get(name)})")
    if mode == "gpu_ane":
        report = config.get("placement", {}).get("validation_report")
        try:
            validation = json.loads(Path(report).read_text()) if report else {}
        except (OSError, ValueError, TypeError):
            validation = {}
        if validation.get("status") != "pass":
            blockers.append("hybrid Qwen3 training requires a passing real-block gradient report")
        elif validation.get("sequence") != config["model"].get("max_seq_length"):
            blockers.append("hybrid gradient report must match the configured sequence length")
    if not host["mactop"]:
        blockers.append("mactop is missing; this runner requires it for system telemetry")
    safety = config.get("safety") or {}
    minimum_disk = int(safety.get("min_free_disk_bytes", 15 * GIB))
    reserve = int(safety.get("reserve_memory_bytes", 4 * GIB))
    limit = config["infrastructure"].get("memory_limit_bytes")
    if minimum_disk < 1 or reserve < 1:
        blockers.append("disk and memory reserves must be positive")
    if host["disk_free_bytes"] < minimum_disk:
        blockers.append(f"free disk is below the configured {minimum_disk / GIB:g} GiB reserve")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        blockers.append("preflight requires an explicit positive MLX memory budget")
    elif host["memory_total_bytes"] is None:
        blockers.append("physical memory is unknown; cannot check the memory reserve")
    elif limit + reserve > host["memory_total_bytes"]:
        blockers.append("MLX budget leaves less than the configured memory reserve")
    if config["training"].get("batch_size") != 1:
        blockers.append("bounded smoke preflight requires batch_size: 1 (no automatic sweep)")
    if not isinstance(config["training"].get("grad_checkpoint"), bool):
        blockers.append("grad_checkpoint must be explicitly true or false")
    steps = config["training"].get("iters")
    profile = safety.get("profile", "smoke")
    if profile == "baseline":
        if steps is not None or config["training"].get("epochs", 0) < 1:
            blockers.append("baseline requires positive epochs and iters: null")
        if config["training"].get("val_batches") != -1 or config["training"].get("test_batches") != -1:
            blockers.append("baseline requires full validation and test evaluation (-1)")
    elif profile != "smoke":
        blockers.append(f"unknown safety profile {profile}")
    elif not isinstance(steps, int) or isinstance(steps, bool) or not 1 <= steps <= 20:
        blockers.append("bounded smoke preflight requires 1 to 20 explicit training steps")
    return {
        "status": "blocked" if blockers else "ready",
        "placement": mode,
        "scope": f"{profile} readiness only; no accelerator execution or residency verified",
        "host": host,
        "blockers": blockers,
    }
