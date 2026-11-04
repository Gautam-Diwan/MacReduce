"""One-Mac feasibility traces for the three client modes.

The GPU client is a short QLoRA run of the pinned Qwen3-0.6B. The Neural
Engine client is an ANEForge linear step plus a matmul held on device mask
0x4. Those are different models. This module does not average their deltas
and does not call the engine arm a Qwen perplexity.

Power comes from mactop. The Prometheus scrape is stored beside the headless
JSON. Energy is the trapezoid of system power, which is not process-scoped.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from .federated import payload_bytes
from .finetune import run_lora
from .plots import plot_energy_comparison, write_run_figures
from .telemetry import SystemTrace, energy_summary

MEMORY_LIMIT_BYTES = 8 * 1024**3


def run_ane_window(
    seconds: float | None = None, stop: threading.Event | None = None, seed: int = 0
) -> dict:
    """Keep the Neural Engine busy after four linear Adam steps.

    The matmul loop ends at ``seconds`` or when ``stop`` is set. One of the
    two is required.
    """

    if seconds is None and stop is None:
        raise ValueError("the Neural Engine window needs a duration or a stop event")
    if seconds is not None and seconds <= 0:
        raise ValueError("the Neural Engine window must be positive")
    import aneforge as af
    import numpy as np

    from .ane_forge import run_linear

    trained = run_linear(steps=4, seed=seed)
    rng = np.random.default_rng(seed)
    width = 256
    features = af.input((32, width))
    weight = rng.standard_normal((width, width)).astype(np.float32)
    program = af.compile(features @ weight)
    batch = rng.standard_normal((32, width)).astype(np.float32)
    calls = 0
    deadline = None if seconds is None else time.perf_counter() + seconds
    while True:
        if stop is not None and stop.is_set():
            break
        if deadline is not None and time.perf_counter() >= deadline:
            break
        program(batch)
        calls += 1
    return {
        "status": trained["status"],
        "model": "ANEForge linear 16x1 Adam, then a 256-wide matmul",
        "qwen_lora": False,
        "aneforge": trained["aneforge"],
        "device_mask": trained["device_mask"],
        "programs": trained["programs"],
        "linear_steps": trained["steps"],
        "linear_losses": trained["losses"],
        "weight_max_abs_move": trained["weight_max_abs_move"],
        "matmul_calls": calls,
        "window_seconds": seconds,
    }


def run_feasibility(
    output: Path,
    data_dir: Path,
    iters: int = 4,
    batch_size: int = 2,
    max_seq_length: int = 256,
    train_examples: int = 64,
    eval_batches: int = 2,
    port: int = 9096,
) -> dict:
    """Measure GPU-only, Neural Engine-only, and both at once on this Mac."""

    if iters < 1 or eval_batches < 1:
        raise ValueError("iters and eval batches must be positive")
    output.mkdir(parents=True, exist_ok=True)
    gpu = _arm(
        output / "gpu",
        port,
        lambda: _gpu(
            output / "gpu",
            data_dir,
            iters,
            batch_size,
            max_seq_length,
            train_examples,
            eval_batches,
        ),
    )
    window = max(float(gpu.get("seconds") or 0), 8.0)
    ane = _arm(output / "ane", port, lambda: run_ane_window(window))
    combined = _arm(
        output / "gpu-ane",
        port,
        lambda: _combined(
            output / "gpu-ane",
            data_dir,
            iters,
            batch_size,
            max_seq_length,
            train_examples,
            eval_batches,
        ),
    )
    report = {
        "status": "ok"
        if all(arm.get("status") == "ok" for arm in (gpu, ane, combined))
        else "partial",
        "machine": "one Mac",
        "memory_limit_bytes": MEMORY_LIMIT_BYTES,
        "attribution": "mactop power is system-wide, not per process",
        "ane_active_note": "ane_active stayed unused as a residency bit; power and bandwidth are the trace",
        "qwen_on_ane": False,
        "arms": {"gpu": gpu, "ane": ane, "gpu_ane": combined},
        "cases": _cases(gpu, ane, combined),
    }
    _write_plots(output, report)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _gpu(directory, data_dir, iters, batch_size, max_seq_length, train_examples, eval_batches):
    import mlx.core as mx

    mx.set_memory_limit(MEMORY_LIMIT_BYTES)
    mx.reset_peak_memory()
    mx.random.seed(0)
    body = run_lora(
        directory / "lora",
        data_dir,
        iters=iters,
        epochs=1,
        test_batches=eval_batches,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        train_examples=train_examples,
        grad_accumulation_steps=1,
        val_batches=eval_batches,
    )
    trainable = int(body["trainable_parameters"])
    body["adapter_bytes_fp32"] = payload_bytes(trainable, quantized=False)
    body["adapter_bytes_int8"] = payload_bytes(trainable, quantized=True, n_tensors=1)
    body["mlx_peak_bytes"] = int(mx.get_peak_memory())
    body["mlx_active_bytes"] = int(mx.get_active_memory())
    body["qwen_lora"] = True
    return body


def _combined(
    directory, data_dir, iters, batch_size, max_seq_length,     train_examples, eval_batches
):
    holder: dict = {}
    stop = threading.Event()

    def _engine():
        try:
            holder["ane"] = run_ane_window(stop=stop)
        except Exception as exc:
            holder["ane"] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}

    thread = threading.Thread(target=_engine)
    thread.start()
    try:
        gpu = _gpu(
            directory, data_dir, iters, batch_size, max_seq_length, train_examples, eval_batches
        )
    finally:
        stop.set()
        thread.join()
    gpu["ane_client"] = holder.get("ane")
    gpu["combined_means"] = "both devices were busy; their parameters were not averaged"
    return gpu


def _arm(directory: Path, port: int, work) -> dict:
    print(f"feasibility arm {directory.name}", flush=True)
    started = time.perf_counter()
    with SystemTrace(directory, port=port) as trace:
        try:
            body = work()
            status = "ok" if body.get("status") == "ok" else body.get("status", "ok")
            error = None
        except Exception as exc:
            body = {}
            status = "error"
            error = f"{type(exc).__name__}: {exc}"
    elapsed = time.perf_counter() - started
    samples = trace.samples
    prom = trace.prometheus
    energy = energy_summary(samples, prom)
    return {
        "status": status,
        "error": error,
        "seconds": elapsed,
        "mactop_status": trace.status,
        "headless_samples": len(samples),
        "prometheus_samples": len(prom),
        "prometheus_port": port,
        "energy": energy,
        "power_max": _peaks(samples),
        "series": [
            {
                key: row.get(key)
                for key in ("t", "gpu", "ane", "ane_bw_gbs", "memory_used_bytes")
            }
            for row in samples
        ],
        "work": _compact(body),
    }


def _peaks(samples: list[dict]) -> dict:
    def _max(key: str):
        values = [row[key] for row in samples if row.get(key) is not None]
        return max(values) if values else None

    return {
        "gpu_w": _max("gpu"),
        "ane_w": _max("ane"),
        "ane_bw_gbs": _max("ane_bw_gbs"),
        "ane_active_max": _max("ane_active"),
    }


def _compact(body: dict) -> dict:
    if not body:
        return {}
    keep = (
        "status",
        "error",
        "model",
        "iters",
        "batch_size",
        "max_seq_length",
        "train_examples",
        "seconds",
        "trainable_parameters",
        "adapter_bytes_fp32",
        "adapter_bytes_int8",
        "baseline_test",
        "adapted_test",
        "history",
        "mlx_peak_bytes",
        "mlx_active_bytes",
        "qwen_lora",
        "qwen_on_ane",
        "linear_losses",
        "linear_steps",
        "weight_max_abs_move",
        "matmul_calls",
        "device_mask",
        "aneforge",
        "combined_means",
        "window_seconds",
    )
    compact = {key: body[key] for key in keep if key in body}
    ane = body.get("ane_client")
    if isinstance(ane, dict):
        compact["ane_client"] = _compact(ane)
    return compact


def _cases(gpu: dict, ane: dict, combined: dict) -> dict:
    gpu_work = gpu.get("work") or {}
    return {
        "1_gpu": {
            "client": "one GPU QLoRA client",
            "seconds": gpu.get("seconds"),
            "energy": gpu.get("energy"),
            "power_max": gpu.get("power_max"),
            "validation": _perplexity(gpu_work),
            "adapter_bytes_fp32": gpu_work.get("adapter_bytes_fp32"),
            "adapter_bytes_int8": gpu_work.get("adapter_bytes_int8"),
            "wire_bytes": 0,
            "status": gpu.get("status"),
        },
        "1_gpu_plus_ane": {
            "client": "one combined client, GPU QLoRA and Neural Engine graph concurrent",
            "same_adapter": False,
            "seconds": combined.get("seconds"),
            "energy": combined.get("energy"),
            "power_max": combined.get("power_max"),
            "validation": _perplexity(combined.get("work") or {}),
            "ane": (combined.get("work") or {}).get("ane_client"),
            "status": combined.get("status"),
        },
        "1_gpu_and_1_ane": {
            "client": "two clients, local work not overlapped",
            "same_adapter": False,
            "seconds": _add(gpu.get("seconds"), ane.get("seconds")),
            "gpu_joules": _add(_nested(gpu, "gpu_joules"), _nested(ane, "gpu_joules")),
            "ane_joules": _add(_nested(gpu, "ane_joules"), _nested(ane, "ane_joules")),
            "gpu_client": _perplexity(gpu_work),
            "ane_client": ane.get("work"),
            "status": "ok" if gpu.get("status") == "ok" and ane.get("status") == "ok" else "partial",
        },
    }


def _perplexity(work: dict) -> dict:
    history = work.get("history") or {}
    return {
        "baseline_test": work.get("baseline_test"),
        "adapted_test": work.get("adapted_test"),
        "validation_during_training": history.get("validation"),
        "scope": "subset of batches named by eval_batches, not the full WikiText-2 splits",
    }


def _nested(arm: dict, key: str):
    energy = arm.get("energy") or {}
    return energy.get(key)


def _add(left, right):
    if left is None or right is None:
        return None
    return left + right


def _write_plots(directory: Path, report: dict) -> None:
    arms = report["arms"]
    for name in ("gpu", "ane", "gpu_ane"):
        arm = arms[name]
        history = (arm.get("work") or {}).get("history")
        write_run_figures(directory / "figures" / name, history, arm.get("series") or [])
    plot_energy_comparison(directory / "figures" / "energy-cases.png", report["cases"])

