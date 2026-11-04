"""Logged QLoRA curve for the pinned Qwen3-0.6B.

``macreduce lora`` calls this. The run writes mlx-lm loss reports, a mactop
trace, and matplotlib figures for loss, perplexity, GPU and Neural Engine
power, memory, and latency. Figures refresh once a minute while training
is still going. The memory budget defaults to the GPU's recommended working set.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from .finetune import MetricLog, run_lora
from .memory import apply_memory_limit, resolve_memory_limit
from .plots import write_run_figures
from .telemetry import SystemTrace, energy_summary, read_series


def allocate_run(root: Path, now: datetime | None = None) -> Path:
    """A new directory named by the local timestamp. A collision gains a suffix."""

    stamp = (now or datetime.now()).strftime("%Y-%m-%dT%H-%M-%S")
    candidate = root / stamp
    suffix = 2
    while candidate.exists():
        candidate = root / f"{stamp}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data) -> None:
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def bind_run_log(run_dir: Path) -> Path:
    """Copy stdout and stderr into ``run_dir/run.log`` for this launch only."""

    path = run_dir / "run.log"
    handle = path.open("w", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
    (run_dir.parent / "latest").write_text(str(run_dir) + "\n")
    return path


def latest_checkpoint(root: Path) -> Path:
    """The newest timestamped run under ``root`` that has a checkpoint."""

    if not root.is_dir():
        raise FileNotFoundError(root)
    runs = sorted((path for path in root.iterdir() if path.is_dir()), key=lambda path: path.name)
    for run in reversed(runs):
        pointer = run / "checkpoints" / "latest.json"
        if not pointer.is_file():
            continue
        target = Path(json.loads(pointer.read_text())["path"])
        if (target / "state.json").is_file():
            return target
    raise FileNotFoundError(f"no checkpoint under {root}")


def run_curve(
    output: Path,
    data_dir: Path,
    port: int = 9096,
    refresh_s: float = 60,
    memory_limit_bytes: int | None = None,
    **lora_kwargs,
) -> dict:
    """Train the curve, trace the machine, and write figures under ``output``."""

    output.mkdir(parents=True, exist_ok=True)
    memory_limit_bytes = resolve_memory_limit(memory_limit_bytes)
    placement = lora_kwargs.pop("placement", "gpu")
    if placement not in {"gpu", "gpu_ane"}:
        raise ValueError(f"unknown placement {placement}")
    figures = output / "figures"
    print(f"qwen3 curve: figures in {figures}, refresh every {refresh_s:.0f}s", flush=True)
    log = MetricLog(output / "history.jsonl")
    stop = threading.Event()

    def _refresh() -> None:
        while not stop.wait(refresh_s):
            write_run_figures(figures, log.snapshot(), read_series(output / "mactop.raw.json"))

    refresher = threading.Thread(target=_refresh, daemon=True)
    started = time.perf_counter()
    with SystemTrace(output, port=port) as trace:
        refresher.start()
        _limit_memory(memory_limit_bytes)
        kwargs = dict(lora_kwargs)
        kwargs.setdefault("steps_per_report", 10)
        if placement == "gpu_ane":
            kwargs["ane_lora"] = True
            print(
                "qwen training: early blocks on the gpu, later blocks on the neural engine",
                flush=True,
            )
        body = run_lora(output, data_dir, metric_log=log, **kwargs)
        body["mlx_peak_bytes"] = _peak_memory()
    stop.set()
    refresher.join(timeout=2)
    samples = trace.samples
    body["seconds_wall"] = time.perf_counter() - started
    body["energy"] = energy_summary(samples, trace.prometheus)
    body["mactop_status"] = trace.status
    body["headless_samples"] = len(samples)
    body["prometheus_samples"] = len(trace.prometheus)
    body["prometheus_port"] = port
    body["memory_limit_bytes"] = memory_limit_bytes
    body["placement"] = placement
    body["log"] = str(output / "run.log")
    body["figures"] = [str(path) for path in write_run_figures(figures, body.get("history"), samples)]
    body["latency"] = _latency_summary(body.get("history") or {})
    return body


def render_saved(directory: Path) -> list[Path]:
    """Redraw figures from a saved curve directory or a feasibility result."""

    report_path = directory / "result.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"no result.json in {directory}")
    report = json.loads(report_path.read_text())
    written: list[Path] = []
    if "arms" in report:
        from .plots import plot_energy_comparison

        for name, arm in report["arms"].items():
            history = (arm.get("work") or {}).get("history")
            written.extend(write_run_figures(directory / "figures" / name, history, arm.get("series")))
        comparison = plot_energy_comparison(directory / "figures" / "energy-cases.png", report.get("cases") or {})
        if comparison is not None:
            written.append(comparison)
        return written
    samples = read_series(directory / "mactop.raw.json")
    return write_run_figures(directory / "figures", report.get("history"), samples)


def _limit_memory(memory_limit_bytes: int) -> None:
    apply_memory_limit(memory_limit_bytes)


def _peak_memory() -> int | None:
    try:
        import mlx.core as mx

        return int(mx.get_peak_memory())
    except Exception:
        return None


def _latency_summary(history: dict) -> dict:
    rates = [
        float(row["iterations_per_second"])
        for row in history.get("train") or []
        if row.get("iterations_per_second")
    ]
    tokens = [
        float(row["tokens_per_second"])
        for row in history.get("train") or []
        if row.get("tokens_per_second")
    ]
    val_times = [
        float(row["val_time"]) for row in history.get("validation") or [] if row.get("val_time") is not None
    ]
    return {
        "median_iteration_ms": _median([1000.0 / rate for rate in rates]),
        "median_tokens_per_second": _median(tokens),
        "median_validation_s": _median(val_times),
        "reports": len(rates),
    }


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])
