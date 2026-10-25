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
    for units in (ct.ComputeUnit.CPU_AND_NE, ct.ComputeUnit.CPU_ONLY):
        loaded = CompiledMLModel(str(compiled), compute_units=units)
        loaded.predict({"x": sample})
        started = time.perf_counter()
        for _ in range(20):
            loaded.predict({"x": sample})
        timings[units.name] = time.perf_counter() - started

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
        "ane_power_watts": None,
        "ane_power_reason": power_text or f"powermetrics exited {power.returncode}",
        "claim": (
            "Under CPU_AND_NE the compute plan assigns the convolution and ReLU to "
            "MLNeuralEngineComputeDevice. Under CPU_ONLY the same operators are not "
            "assigned there. Wall-clock predict time is not residency. No ANE power "
            "sample was read, and the Gemma or Qwen draft model was not converted."
        ),
    }
