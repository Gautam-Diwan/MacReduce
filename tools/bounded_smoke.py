"""Run one configured experiment with time, disk, and memory stop conditions.

This supervisor is a best-effort stop mechanism, not an OS memory sandbox.
All files and subprocesses belong to this invocation; it never stops other work.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import yaml


def swap_growth_limit(config):
    value = config.get("safety", {}).get("max_swap_growth_bytes", 512 * 1024**2)
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("max_swap_growth_bytes must be a positive integer or null")
    return value


def swap_stop_reason(initial, current, limit):
    if limit is not None and current - initial > limit:
        return f"system swap grew more than {limit / 1024**2:g} MiB"
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "supervisor.pid").write_text(str(os.getpid()) + "\n")

    def interrupted(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    initial_swap = psutil.swap_memory().used
    config = yaml.safe_load(Path(args.config).read_text())
    maximum_swap_growth = swap_growth_limit(config)
    minimum_disk = int(config["safety"]["min_free_disk_bytes"])
    if minimum_disk < 1:
        raise ValueError("disk reserve must be positive")
    (output / "guard_policy.json").write_text(json.dumps({
        "max_swap_growth_bytes": maximum_swap_growth,
        "swap_growth_guard_enabled": maximum_swap_growth is not None,
        "min_free_disk_bytes": minimum_disk,
        "min_available_memory_bytes": 256 * 1024**2,
        "max_seconds": args.seconds,
    }, indent=2) + "\n")
    started = time.monotonic()
    reason = None
    command = [sys.executable, "main.py", "--config", args.config]
    (output / "command.json").write_text(json.dumps(command) + "\n")
    with (output / "stdout.log").open("w") as log, (output / "resources.jsonl").open("w") as trace:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                memory = psutil.virtual_memory()
                swap = psutil.swap_memory().used
                battery = psutil.sensors_battery()
                pressure = None
                try:
                    result = subprocess.run(
                        ["sysctl", "-n", "kern.memorystatus_vm_pressure_level"],
                        capture_output=True, text=True, timeout=3,
                    )
                    if result.returncode == 0:
                        pressure = int(result.stdout.strip())
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    pass
                elapsed = time.monotonic() - started
                free = shutil.disk_usage(output).free
                trace.write(json.dumps({
                    "elapsed_s": elapsed, "available_bytes": memory.available,
                    "swap_used_bytes": swap, "pressure_level": pressure,
                    "disk_free_bytes": free,
                    "battery_percent": None if battery is None else battery.percent,
                    "power_plugged": None if battery is None else battery.power_plugged,
                }) + "\n")
                trace.flush()
                if pressure is not None and pressure & 4:
                    reason = "critical system memory pressure"
                elif memory.available < 256 * 1024**2:
                    reason = "available system memory below 256 MiB"
                elif swap_stop_reason(initial_swap, swap, maximum_swap_growth):
                    reason = swap_stop_reason(initial_swap, swap, maximum_swap_growth)
                elif free < minimum_disk:
                    reason = f"free disk below {minimum_disk / 1024**3:g} GiB"
                elif battery is not None and not battery.power_plugged and battery.percent < 20:
                    reason = "unplugged battery below 20 percent"
                elif elapsed > args.seconds:
                    reason = "wall time limit"
                if reason:
                    break
                time.sleep(2)
        except KeyboardInterrupt:
            reason = "interrupted"
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    summary = {"returncode": process.returncode, "stop_reason": reason,
               "elapsed_s": time.monotonic() - started}
    (output / "exit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    raise SystemExit(1 if reason else process.returncode)


if __name__ == "__main__":
    main()
