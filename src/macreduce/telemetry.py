"""System power, memory, and energy from mactop.

Headless JSON is the series used for plots. Prometheus scrapes of
``mactop_power_watts`` are stored beside it. Both are system-wide.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from pathlib import Path

_POWER = re.compile(r'^mactop_power_watts\{component="([^"]+)"\}\s+(\S+)')
_RAILS = ("gpu", "gpu_sram", "ane", "cpu", "dram")


def parse_prometheus(text: str) -> dict[str, float]:
    """Read mactop power gauges from a Prometheus text scrape."""

    power: dict[str, float] = {}
    for line in text.splitlines():
        match = _POWER.match(line)
        if match:
            power[match.group(1)] = float(match.group(2))
    return power


def integrate_joules(samples: list[dict], key: str) -> float | None:
    """Trapezoid integral of one power series. None when two stamps are missing."""

    total = 0.0
    used = 0
    for left, right in zip(samples, samples[1:]):
        start, end = left.get("t"), right.get("t")
        if start is None or end is None or end <= start:
            continue
        first, second = left.get(key), right.get(key)
        if first is None or second is None:
            continue
        total += 0.5 * (float(first) + float(second)) * (end - start)
        used += 1
    if used == 0:
        return None
    return total


def cumulative_joules(samples: list[dict], key: str) -> list[tuple[float, float]]:
    """Elapsed seconds and cumulative joules. The first point is zero."""

    stamped = [row for row in samples if row.get("t") is not None and row.get(key) is not None]
    if len(stamped) < 2:
        return []
    origin = float(stamped[0]["t"])
    total = 0.0
    points = [(0.0, 0.0)]
    for left, right in zip(stamped, stamped[1:]):
        dt = float(right["t"]) - float(left["t"])
        if dt <= 0:
            continue
        total += 0.5 * (float(left[key]) + float(right[key])) * dt
        points.append((float(right["t"]) - origin, total))
    return points


def headless_series(text: str) -> list[dict]:
    """Parse mactop JSON, stopping before a trailing partial write."""

    decoder = json.JSONDecoder()
    rows: list[dict] = []
    remaining = text.lstrip()
    while remaining.strip():
        try:
            value, end = decoder.raw_decode(remaining)
        except json.JSONDecodeError:
            break
        samples = value if isinstance(value, list) else [value]
        for sample in samples:
            if isinstance(sample, dict):
                rows.append(_sample_row(sample))
        remaining = remaining[end:].lstrip()
    return rows


def _sample_row(sample: dict) -> dict:
    soc = sample.get("soc_metrics") or {}
    memory = sample.get("memory") or {}
    stamp = sample.get("timestamp")
    return {
        "t": datetime.fromisoformat(stamp).timestamp() if stamp else None,
        "gpu": soc.get("gpu_power"),
        "gpu_sram": soc.get("gpu_sram_power"),
        "ane": soc.get("ane_power"),
        "ane_bw_gbs": soc.get("ane_bw_combined_gbs"),
        "ane_active": soc.get("ane_active"),
        "cpu": soc.get("cpu_power"),
        "dram": soc.get("dram_power"),
        "memory_used_bytes": memory.get("used"),
        "memory_total_bytes": memory.get("total"),
    }


def energy_summary(samples: list[dict], prometheus: list[dict] | None = None) -> dict:
    """Joules and average watts over the mactop sample span."""

    prometheus = prometheus or []
    summary = {
        "source": "mactop headless soc_metrics, trapezoid over sample timestamps",
        "scope": "system",
        "gpu_joules": integrate_joules(samples, "gpu"),
        "gpu_sram_joules": integrate_joules(samples, "gpu_sram"),
        "ane_joules": integrate_joules(samples, "ane"),
        "cpu_joules": integrate_joules(samples, "cpu"),
        "dram_joules": integrate_joules(samples, "dram"),
        "prometheus_gpu_joules": integrate_joules(prometheus, "gpu"),
        "prometheus_ane_joules": integrate_joules(prometheus, "ane"),
    }
    stamps = [row["t"] for row in samples if row.get("t") is not None]
    span = None
    if len(stamps) >= 2 and stamps[-1] > stamps[0]:
        span = stamps[-1] - stamps[0]
        summary["sample_span_s"] = span
    for rail in _RAILS:
        joules = summary[f"{rail}_joules"]
        if joules is not None and span:
            summary[f"{rail}_watts_avg"] = joules / span
    return summary


class SystemTrace:
    """Headless mactop plus a Prometheus scrape until the block exits."""

    def __init__(self, directory: Path, port: int = 9096, interval_ms: int = 1000):
        self.directory = Path(directory)
        self.port = port
        self.interval_ms = interval_ms
        self.samples: list[dict] = []
        self.prometheus: list[dict] = []
        self.status = "not_started"
        self._process: subprocess.Popen | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stdout = None

    def __enter__(self) -> SystemTrace:
        self.directory.mkdir(parents=True, exist_ok=True)
        self._stdout = (self.directory / "mactop.raw.json").open("w")
        command = [
            "mactop",
            "--prometheus",
            f":{self.port}",
            "--headless",
            "--format",
            "json",
            "--count",
            "0",
            "--interval",
            str(self.interval_ms),
        ]
        self._process = subprocess.Popen(
            command,
            stdout=self._stdout,
            stderr=subprocess.PIPE,
            text=True,
        )
        self._thread = threading.Thread(target=self._scrape, daemon=True)
        self._thread.start()
        self.status = "running"
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._thread is not None:
            self._thread.join(timeout=3)
        if self._stdout is not None:
            self._stdout.close()
        self.samples = read_series(self.directory / "mactop.raw.json")
        (self.directory / "prometheus.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in self.prometheus)
        )
        code = self._process.returncode if self._process is not None else None
        self.status = "available" if self.samples else "unavailable"
        (self.directory / "mactop.status.json").write_text(
            json.dumps(
                {
                    "status": self.status,
                    "returncode": code,
                    "prometheus_port": self.port,
                    "prometheus_samples": len(self.prometheus),
                    "headless_samples": len(self.samples),
                },
                indent=2,
            )
            + "\n"
        )

    def _scrape(self) -> None:
        url = f"http://127.0.0.1:{self.port}/metrics"
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    power = parse_prometheus(response.read().decode())
            except (OSError, TimeoutError, ValueError):
                self._stop.wait(0.2)
                continue
            if power:
                self.prometheus.append({"t": time.time(), **power})
            self._stop.wait(self.interval_ms / 1000)


def read_series(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return headless_series(path.read_text())
