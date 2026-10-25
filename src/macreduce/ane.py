"""Core ML compute-plan probe for a small convolution.

The serving drafter is not converted here. A compute-unit flag is a constraint.
``placement_verified`` stays false until a draft model is converted and its plan
or a power sample says the Neural Engine ran it. A missing power sample stays
null.
"""

from __future__ import annotations

import importlib.metadata as metadata
import platform
import subprocess
import time
from pathlib import Path

from .observe import normalize_mactop, parse_mactop


def _assignments(plan) -> list[dict]:
    rows: list[dict] = []

    def walk(block) -> None:
        for operation in block.operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            device = None
            if usage is not None and usage.preferred_compute_device is not None:
                device = type(usage.preferred_compute_device).__name__
            rows.append({"operator": operation.operator_name, "device": device})
            for child in operation.blocks:
                walk(child)

    for function in plan.model_structure.program.functions.values():
        walk(function.block)
    return rows


def _convert(destination: Path):
    import coremltools as ct
    import numpy as np
    import torch
    import torch.nn as nn

    class Net(nn.Module):
        def forward(self, x):
            weight = torch.ones(64, 3, 3, 3, dtype=x.dtype)
            return torch.relu(torch.nn.functional.conv2d(x, weight, padding=1))

    example = torch.zeros(1, 3, 256, 256)
    traced = torch.jit.trace(Net().eval(), example)
    model = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=example.shape, dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    package = destination / "probe.mlpackage"
    model.save(str(package))
    return package


def _mactop_while(during, seconds: float = 2.0, interval_ms: int = 200) -> dict:
    """Sample system power while ``during`` runs. A zero reading stays zero."""

    count = max(1, int(seconds * 1000 / interval_ms))
    command = [
        "mactop",
        "--headless",
        "--format",
        "json",
        "--count",
        str(count),
        "--interval",
        str(interval_ms),
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except OSError as exc:
        return {"status": "unavailable", "reason": str(exc), "samples": []}
    try:
        during()
        stdout, stderr = process.communicate(timeout=seconds + 15)
    except Exception:
        process.kill()
        process.communicate()
        raise
    if process.returncode != 0:
        return {
            "status": "unavailable",
            "reason": (stderr or "").strip()[:500],
            "samples": [],
        }
    rows = []
    for raw in parse_mactop(stdout):
        soc = raw.get("soc_metrics", {})
        normalized = normalize_mactop(raw)
        rows.append(
            {
                "ane_power_w": normalized["ane_power_w"],
                "ane_active": soc.get("ane_active"),
                "ane_bw_combined_gbs": soc.get("ane_bw_combined_gbs"),
                "gpu_power_w": normalized["gpu_power_w"],
                "cpu_power_w": normalized["cpu_power_w"],
            }
        )
    return {
        "status": "available",
        "source": "mactop/IOReport",
        "scope": "system",
        "estimated": True,
        "attribution": "not attributable to this process",
        "samples": rows,
    }


def _peak(reading: dict, key: str):
    values = [row[key] for row in reading.get("samples", []) if row.get(key) is not None]
    if not values:
        return None
    return max(values)


def placement_probe(work: Path) -> dict:
    """Convert one fixed-shape network and read plans for two compute-unit constraints."""

    import coremltools as ct
    import numpy as np
    from coremltools.models import CompiledMLModel
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.utils import compile_model

    work.mkdir(parents=True, exist_ok=True)
    package = _convert(work)
    compiled = Path(compile_model(str(package), destination_path=str(work / "probe.mlmodelc")))
    plans = {}
    for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_ONLY):
        plan = MLComputePlan.load_from_path(str(compiled), compute_units=units)
        plans[units.name] = _assignments(plan)

    neural = [
        row
        for row in plans["CPU_AND_NE"]
        if row["device"] == "MLNeuralEngineComputeDevice"
    ]
    cpu_only_neural = [
        row
        for row in plans["CPU_ONLY"]
        if row["device"] == "MLNeuralEngineComputeDevice"
    ]
    sample = np.zeros((1, 3, 256, 256), dtype=np.float16)
    timings = {}
    loaded_models = {}
    for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_ONLY):
        loaded = CompiledMLModel(str(compiled), compute_units=units)
        loaded.predict({"x": sample})
        started = time.perf_counter()
        for _ in range(20):
            loaded.predict({"x": sample})
        timings[units.name] = time.perf_counter() - started
        loaded_models[units.name] = loaded

    def sustain(model) -> None:
        deadline = time.perf_counter() + 2.0
        while time.perf_counter() < deadline:
            model.predict({"x": sample})

    mactop = {
        "idle": _mactop_while(lambda: time.sleep(2.0)),
        "CPU_ONLY": _mactop_while(lambda: sustain(loaded_models["CPU_ONLY"])),
        "CPU_AND_NE": _mactop_while(lambda: sustain(loaded_models["CPU_AND_NE"])),
    }
    ane_power = _peak(mactop["CPU_AND_NE"], "ane_power_w")
    power = subprocess.run(
        ["powermetrics", "-s", "ane_power", "-n", "1", "-i", "200"],
        capture_output=True,
        text=True,
    )
    power_text = (power.stderr or power.stdout).strip()
    model_name = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True)
    memsize = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
    return {
        "coremltools": metadata.version("coremltools"),
        "torch": metadata.version("torch"),
        "hw_model": model_name.stdout.strip() or None,
        "memory_bytes": int(memsize.stdout.strip()) if memsize.stdout.strip().isdigit() else None,
        "torch_note": (
            "coremltools 9 reports Torch 2.7.0 as the newest version it has tested. "
            "Conversion still produced an mlprogram on the installed Torch."
        ),
        "os": platform.platform(),
        "machine": platform.machine(),
        "network": "conv2d 3x64 kernel 3, relu, float16, input 1x3x256x256",
        "draft_model_converted": False,
        "placement_verified": False,
        "plan_places_conv_on_neural_engine": bool(neural) and not cpu_only_neural,
        "plans": plans,
        "predict_seconds_20": timings,
        "mactop": {
            name: {
                "status": reading["status"],
                "ane_power_w_max": _peak(reading, "ane_power_w"),
                "ane_active_max": _peak(reading, "ane_active"),
                "ane_bw_combined_gbs_max": _peak(reading, "ane_bw_combined_gbs"),
                "gpu_power_w_max": _peak(reading, "gpu_power_w"),
                "cpu_power_w_max": _peak(reading, "cpu_power_w"),
                "samples": reading.get("samples", []),
                "reason": reading.get("reason"),
            }
            for name, reading in mactop.items()
        },
        "ane_power_watts": ane_power,
        "ane_power_source": "mactop" if ane_power is not None else None,
        "ane_power_reason": (
            f"powermetrics: {power_text or power.returncode}. "
            "mactop readings are system-wide estimates and are not attributed to this process."
        ),
        "claim": (
            "Under CPU_AND_NE the compute plan assigns the convolution and ReLU to "
            "MLNeuralEngineComputeDevice. Under CPU_ONLY the same operators are not "
            "assigned there. mactop samples power while those predictions run. "
            "A mactop watt reading is a system estimate, not proof that this process "
            "occupied the Neural Engine. The serving draft model was not converted."
        ),
    }
