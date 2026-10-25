"""Structured timings and optional system telemetry with explicit provenance."""

import contextlib
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil


def probe(command, timeout=10):
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        return {
            "status": "available" if r.returncode == 0 else "unavailable",
            "value": r.stdout.strip() if r.returncode == 0 else None,
            "reason": r.stderr.strip()[:1200] if r.returncode else None,
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "unavailable", "value": None, "reason": str(exc)}


def capabilities(smoke=False):
    chip = probe(["sysctl", "-n", "machdep.cpu.brand_string"])
    # system_profiler often remains readable when sysctl is sandbox restricted.
    info = probe(["system_profiler", "SPHardwareDataType", "-json"])
    model = None
    if info["value"]:
        try:
            hw = json.loads(info["value"])["SPHardwareDataType"][0]
            model = hw.get("machine_model")
            if chip["value"] is None and hw.get("chip_type"):
                chip = {
                    "status": "available",
                    "value": hw["chip_type"],
                    "reason": None,
                    "source": "system_profiler",
                }
        except (ValueError, KeyError, IndexError):
            pass
    match = re.search(r"Apple M(\d+)", chip["value"] or "")
    generation = int(match[1]) if match else None
    features = {
        n: probe(["sysctl", "-n", "hw.optional.arm." + n])
        for n in ("FEAT_SME", "FEAT_SME2", "FEAT_BF16", "FEAT_DotProd")
    }
    try:
        memory = {
            "total_bytes": psutil.virtual_memory().total,
            "available_bytes": psutil.virtual_memory().available,
            "source": "psutil",
        }
    except (OSError, psutil.Error) as exc:
        memory = {"total_bytes": None, "available_bytes": None, "reason": str(exc)}
    mlx = {"installed": importlib.util.find_spec("mlx") is not None, "execution": "not_tested"}
    if smoke and mlx["installed"]:
        mlx["probe"] = probe(
            [
                sys.executable,
                "-c",
                "import mlx.core as m; "
                "m.set_default_device(m.gpu); "
                "print((m.ones((16,16))@m.ones((16,16))).sum().item())",
            ]
        )
        mlx["execution"] = mlx["probe"]["status"]
    return {
        "schema": 1,
        "os": platform.platform(),
        "architecture": platform.machine(),
        "model": model,
        "chip": chip,
        "memory": memory,
        "cpu_features": features,
        "accelerators": {
            "cpu_simd": {
                "hardware": "inferred" if platform.machine() == "arm64" else "unknown",
                "operator_dispatch": "not_verified",
            },
            "cpu_matrix": {
                "hardware": "probe cpu_features SME/SME2",
                "access": "Accelerate/SME; no dedicated backend in v0.1",
                "operator_dispatch": "not_verified",
            },
            "gpu": {"hardware": "inferred" if generation else "unknown", "mlx": mlx},
            "ane": {
                "hardware": "inferred" if generation else "unknown",
                "basis": "Apple M-series specification; hardware presence is not execution",
                "training_backend": "not_implemented",
                "execution": "not_tested",
            },
            "gpu_neural_accelerators": {
                "hardware": (
                    "inferred_present"
                    if generation == 5
                    else "not_present"
                    if generation in (1, 2, 3, 4)
                    else "unknown"
                ),
                "basis": "Apple M5 GPU specification; future generations require verification",
                "operator_dispatch": "not_verified",
                "execution": "not_tested",
            },
            "media_engines": {
                "hardware": "inferred" if generation else "unknown",
                "use": "future video preprocessing; not Transformer training",
            },
        },
        "tools": {
            "mactop": shutil.which("mactop"),
            "powermetrics": shutil.which("powermetrics"),
            "xctrace": probe(["xcrun", "--find", "xctrace"]),
        },
    }


class Events:
    def __init__(self, directory, node, run_id=None):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", node):
            raise ValueError("invalid node name")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / f"{node}.jsonl"
        self.node, self.run_id = node, run_id
        self.lock = threading.Lock()

    def emit(self, event, **fields):
        item = {
            "schema": 1,
            "event": event,
            "node": self.node,
            "run_id": self.run_id,
            "wall_ns": time.time_ns(),
            "monotonic_ns": time.monotonic_ns(),
            "pid": os.getpid(),
            **fields,
        }
        with self.lock, self.path.open("a") as f:
            f.write(json.dumps(item, allow_nan=False) + "\n")

    @contextlib.contextmanager
    def span(self, name, **fields):
        start = time.monotonic_ns()
        error = None
        try:
            yield
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            self.emit(
                "span",
                phase=name,
                start_ns=start,
                duration_ns=time.monotonic_ns() - start,
                error=error,
                **fields,
            )


def process_sample():
    try:
        p = psutil.Process()
        times = p.cpu_times()
        return {
            "status": "available",
            "rss_bytes": p.memory_info().rss,
            "cpu_user_seconds": times.user,
            "cpu_system_seconds": times.system,
            "source": "psutil",
            "scope": "process",
        }
    except (OSError, psutil.Error) as exc:
        return {
            "status": "unavailable",
            "rss_bytes": None,
            "reason": str(exc),
            "source": "psutil",
            "scope": "process",
        }


class Sampler:
    def __init__(self, events, interval=1.0):
        self.events, self.interval = events, interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)

    def run(self):
        while not self.stop_event.is_set():
            self.events.emit("process_sample", **process_sample())
            self.stop_event.wait(self.interval)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop_event.set()
        self.thread.join(timeout=2)


def parse_mactop(text):
    """Accept one JSON array, pretty JSON, concatenated JSON, or JSON Lines."""
    decoder = json.JSONDecoder()
    while text.strip():
        text = text.lstrip()
        value, end = decoder.raw_decode(text)
        yield from value if isinstance(value, list) else [value]
        text = text[end:]


def normalize_mactop(sample):
    soc = sample.get("soc_metrics", {})
    # Preserve missing values as null. A reported zero remains a sensor observation.
    return {
        "source": "mactop/IOReport",
        "scope": "system",
        "estimated": True,
        "attribution": "not attributable to this training process",
        "sensor_timestamp": sample.get("timestamp"),
        "cpu_power_w": soc.get("cpu_power"),
        "gpu_power_w": soc.get("gpu_power"),
        "ane_power_w": soc.get("ane_power"),
        "dram_power_w": soc.get("dram_power"),
        "gpu_usage_percent": sample.get("gpu_usage"),
        "cpu_temperature_c": soc.get("cpu_temp"),
        "gpu_temperature_c": soc.get("gpu_temp"),
        "raw": sample,
    }


def collect_mactop(directory, seconds=10, interval_ms=1000):
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        XDG_STATE_HOME=str(path.resolve() / "state"),
        XDG_CONFIG_HOME=str(path.resolve() / "config"),
    )
    command = [
        "mactop",
        "--headless",
        "--format",
        "json",
        "--count",
        str(max(1, int(seconds * 1000 / interval_ms))),
        "--interval",
        str(interval_ms),
    ]
    try:
        r = subprocess.run(command, capture_output=True, text=True, env=env, timeout=seconds + 20)
        (path / "mactop.raw.json").write_text(r.stdout)
        (path / "mactop.stderr.txt").write_text(r.stderr)
        status = {
            "status": "available" if r.returncode == 0 else "unavailable",
            "returncode": r.returncode,
            "reason": r.stderr[:2000],
            "command": command,
            "version": probe(["mactop", "--version"]),
        }
        if r.returncode == 0:
            with (path / "mactop.normalized.jsonl").open("w") as f:
                for sample in parse_mactop(r.stdout):
                    f.write(json.dumps(normalize_mactop(sample)) + "\n")
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        status = {"status": "unavailable", "reason": str(exc), "command": command}
    (path / "mactop.status.json").write_text(json.dumps(status, indent=2))
    return status


def report(directory):
    root = Path(directory)
    events = []
    for path in sorted(root.rglob("*.jsonl")):
        for line in path.read_text().splitlines():
            try:
                item = json.loads(line)
                if "event" in item:
                    events.append(item)
            except ValueError:
                continue  # A crashed writer can leave a partial final line.
    spans = [e for e in events if e["event"] == "span"]
    groups = {}
    for e in spans:
        groups.setdefault((e["node"], e["phase"]), []).append(e["duration_ns"] / 1e6)
    import numpy as np

    lines = [
        "# MacReduce run report",
        "",
        "Timings are local monotonic durations. Cross-node clocks "
        "are not synchronized. Transfer timings include request and server overhead; byte counts "
        "are application payload, not radio traffic.",
        "Nested phases overlap (for example HTTP polling inside barrier wait); do not add all rows "
        "to estimate total elapsed time. First-step timings include warmup effects.",
        "",
        "| Node | Phase | Count | Total s | p50 ms | p95 ms |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (node, phase), values in sorted(groups.items()):
        lines.append(
            f"| {node} | {phase} | {len(values)} | {sum(values) / 1000:.3f} | "
            f"{np.percentile(values, 50):.3f} | {np.percentile(values, 95):.3f} |"
        )
    rounds = [e for e in events if e["event"] == "round_committed"]
    transfers = [e for e in events if e["event"] == "transfer"]
    retries = [e for e in events if e["event"] == "request_retry"]
    summary = {
        "committed_tokens": sum(e["tokens"] for e in rounds),
        "completed_rounds": len(rounds),
        "successful_request_payload_sent_bytes": sum(e["sent_payload_bytes"] for e in transfers),
        "successful_request_payload_received_bytes": sum(
            e["received_payload_bytes"] for e in transfers
        ),
        "retry_events": len(retries),
        "node_training": {},
    }
    lines.extend(
        [
            "",
            "| Node | Local training tokens | Completed step time s | Tokens per training second |",
            "|---|---:|---:|---:|",
        ]
    )
    for node in sorted({e["node"] for e in spans if e["phase"] == "train_step"}):
        steps = [
            e
            for e in spans
            if e["node"] == node and e["phase"] == "train_step" and not e.get("error")
        ]
        seconds = sum(e["duration_ns"] for e in steps) / 1e9
        tokens = sum(e["tokens"] for e in steps)
        rate = tokens / seconds if seconds else None
        summary["node_training"][node] = {
            "tokens": tokens,
            "seconds": seconds,
            "tokens_per_training_second": rate,
        }
        rate_text = f"{rate:.1f}" if rate is not None else "unavailable"
        lines.append(f"| {node} | {tokens} | {seconds:.6f} | {rate_text} |")
    lines.extend(
        [
            "",
            "These are compute-only rates, including warmup, not end-to-end cluster throughput.",
            f"Committed tokens: {summary['committed_tokens']}. Retry events: {len(retries)}.",
            f"Successful request payloads: {summary['successful_request_payload_sent_bytes']} "
            f"bytes sent and {summary['successful_request_payload_received_bytes']} bytes received.",
            "Failed-attempt bytes are not included because delivery is unknown.",
        ]
    )
    if rounds:
        lines.extend(["", "| Version | Validation loss | Tokens in round |", "|---:|---:|---:|"])
        for e in rounds:
            lines.append(f"| {e['version']} | {e['validation_loss']:.5f} | {e['tokens']} |")
    rss = [e["rss_bytes"] for e in events if e["event"] == "process_sample" and e.get("rss_bytes")]
    lines += [
        "",
        f"Peak observed single-process RSS: {max(rss) if rss else 'unavailable'} bytes.",
        "This is sampled, not the allocator peak or summed cluster RAM.",
        "",
        "System power is only available if separately captured. ANE activity and M5 tensor "
        "dispatch cannot be inferred from these training spans.",
        "",
    ]
    (root / "report.md").write_text("\n".join(lines))
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    # Each node gets its own zero. Do not imply clock synchronization in a combined trace.
    for node in sorted({e["node"] for e in spans}):
        local = [e for e in spans if e["node"] == node]
        origin = min(e["start_ns"] for e in local)
        trace = [
            {
                "name": e["phase"],
                "ph": "X",
                "pid": e["pid"],
                "tid": 1,
                "ts": (e["start_ns"] - origin) / 1000,
                "dur": e["duration_ns"] / 1000,
                "args": {k: v for k, v in e.items() if k not in ("phase",)},
            }
            for e in local
        ]
        (root / f"{node}.trace.json").write_text(json.dumps({"traceEvents": trace}))
    return root / "report.md"
