"""Core ML placement measurements.

``placement_probe`` converts a convolution. ``decoder_workload`` converts one
fixed-shape decoder block. ``qwen_forward_workload`` converts a 28-layer
network with the Qwen3-0.6B widths and a vocabulary projection. Both time the
same forward on CPU-only Core ML, GPU, and CPU+Neural Engine. The Qwen draft weights
are not converted. A compute-unit flag is a constraint. ``placement_verified``
for the decoder requires the plan to put its matmul and linear ops on the
Neural Engine and mactop to show Neural Engine bandwidth during that run and
not during the CPU-only run. A missing power sample stays null.
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
            weight = torch.ones(128, 3, 3, 3, dtype=x.dtype)
            hidden = torch.relu(torch.nn.functional.conv2d(x, weight, padding=1))
            projected = torch.ones(128, 128, 3, 3, dtype=x.dtype)
            return torch.relu(torch.nn.functional.conv2d(hidden, projected, padding=1))

    example = torch.zeros(1, 3, 512, 512)
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
        during_result = during()
        stdout, stderr = process.communicate(timeout=seconds + 20)
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
                "timestamp": raw.get("timestamp"),
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
        "during": during_result,
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
    sample = np.zeros((1, 3, 512, 512), dtype=np.float16)
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

    def sustain(model) -> dict:
        started = time.perf_counter()
        deadline = started + 8.0
        predicts = 0
        while time.perf_counter() < deadline:
            model.predict({"x": sample})
            predicts += 1
        return {"predicts": predicts, "seconds": time.perf_counter() - started}

    mactop = {
        "idle": _mactop_while(lambda: time.sleep(8.0), seconds=8.0),
        "CPU_ONLY": _mactop_while(lambda: sustain(loaded_models["CPU_ONLY"]), seconds=8.0),
        "CPU_AND_NE": _mactop_while(lambda: sustain(loaded_models["CPU_AND_NE"]), seconds=8.0),
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
        "network": "two conv2d 3x128 and 128x128, kernel 3, relu, float16, input 1x3x512x512",
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
                "during": reading.get("during"),
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


def _decoder():
    import torch
    import torch.nn as nn

    class Block(nn.Module):
        def forward(self, x):
            hidden = self.norm1(x)
            query = self.query(hidden).reshape(1, 128, 4, 64).transpose(1, 2)
            key = self.key(hidden).reshape(1, 128, 4, 64).transpose(1, 2)
            value = self.value(hidden).reshape(1, 128, 4, 64).transpose(1, 2)
            scores = torch.matmul(query, key.transpose(-2, -1)) * (64**-0.5)
            weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights, value).transpose(1, 2).reshape(1, 128, 256)
            residual = x + self.output(context)
            updated = self.norm2(residual)
            return residual + self.down(torch.relu(self.up(updated)))

    block = Block()
    block.norm1 = nn.LayerNorm(256)
    block.query = nn.Linear(256, 256, bias=False)
    block.key = nn.Linear(256, 256, bias=False)
    block.value = nn.Linear(256, 256, bias=False)
    block.output = nn.Linear(256, 256, bias=False)
    block.norm2 = nn.LayerNorm(256)
    block.up = nn.Linear(256, 1024, bias=False)
    block.down = nn.Linear(1024, 256, bias=False)
    return block


def _compile_decoder(destination: Path, module, example):
    import coremltools as ct
    import numpy as np
    import torch
    from coremltools.models.utils import compile_model

    traced = torch.jit.trace(module.eval(), example)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="x", shape=(1, 128, 256), dtype=np.float16)],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
    )
    package = destination / "decoder.mlpackage"
    converted.save(str(package))
    return compile_model(str(package), destination_path=str(destination / "decoder.mlmodelc"))


def _time_predicts(model, features: dict, repeats: int) -> float:
    for _ in range(3):
        model.predict(features)
    started = time.perf_counter()
    for _ in range(repeats):
        model.predict(features)
    return (time.perf_counter() - started) / repeats


def _hot(samples: list[dict], key: str, floor: float) -> int:
    return sum((sample.get(key) or 0) >= floor for sample in samples)


def decoder_workload() -> dict:
    """Time one decoder forward on CPU, GPU, and the Neural Engine.

    The block is batch 1, 128 tokens, width 256, four heads, float16. It is a
    serving-shaped attention step. It is not the Qwen draft.
    """

    import tempfile

    import coremltools as ct
    import numpy as np
    import torch
    from coremltools.models import CompiledMLModel
    from coremltools.models.compute_plan import MLComputePlan

    module = _decoder().eval()
    example = torch.randn(1, 128, 256)
    with torch.no_grad():
        reference = module(example).detach().numpy()
    with tempfile.TemporaryDirectory() as directory:
        compiled = _compile_decoder(Path(directory), module, example)
        plans = {
            name: _assignments(
                MLComputePlan.load_from_path(compiled, compute_units=getattr(ct.ComputeUnit, name))
            )
            for name in ("CPU_AND_NE", "CPU_ONLY")
        }
        features = {"x": np.ascontiguousarray(example.numpy().astype(np.float16))}
        compiled_path = str(compiled)
        ane_model = CompiledMLModel(compiled_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        cpu_model = CompiledMLModel(compiled_path, compute_units=ct.ComputeUnit.CPU_ONLY)
        predicted = np.array(next(iter(ane_model.predict(features).values())), dtype=np.float32)
        cpu_seconds = _time_predicts(cpu_model, features, 30)
        ane_seconds = _time_predicts(ane_model, features, 30)
        gpu = _gpu_forward(module, example)

        def sustain(model) -> int:
            completed = 0
            deadline = time.perf_counter() + 3.0
            while time.perf_counter() < deadline:
                model.predict(features)
                completed += 1
            return completed

        idle = _mactop_while(lambda: time.sleep(2.0), seconds=2.0)
        cpu_trace = _mactop_while(lambda: sustain(cpu_model), seconds=3.0)
        ane_trace = _mactop_while(lambda: sustain(ane_model), seconds=3.0)
        hybrid = _hybrid(ane_model, features, module, example)
    ne_compute = [
        row
        for row in plans["CPU_AND_NE"]
        if row["device"] == "MLNeuralEngineComputeDevice"
        and row["operator"] in {"ios17.linear", "ios17.matmul"}
    ]
    cpu_ne = [
        row for row in plans["CPU_ONLY"] if row["device"] == "MLNeuralEngineComputeDevice"
    ]
    ane_hot = _hot(ane_trace["samples"], "ane_bw_combined_gbs", 1.0)
    cpu_hot = _hot(cpu_trace["samples"], "ane_bw_combined_gbs", 1.0)
    verified = bool(ne_compute) and not cpu_ne and ane_hot >= 2 and cpu_hot == 0
    return {
        "status": "ok",
        "coremltools": metadata.version("coremltools"),
        "host": platform.platform(),
        "workload": {
            "kind": "decoder_block",
            "batch": 1,
            "tokens": 128,
            "width": 256,
            "heads": 4,
            "mlp_width": 1024,
            "precision": "float16",
            "qwen_draft_weights": False,
        },
        "plans": plans,
        "parity_max_abs": float(np.max(np.abs(reference - predicted))),
        "cpu_only": {"seconds_per_forward": cpu_seconds},
        "cpu_and_ne": {"seconds_per_forward": ane_seconds},
        "gpu": gpu,
        "mactop": {
            "idle": idle,
            "CPU_ONLY": cpu_trace,
            "CPU_AND_NE": ane_trace,
            "hybrid": hybrid,
        },
        "ane_bandwidth_samples_at_least_1gbs": {"CPU_AND_NE": ane_hot, "CPU_ONLY": cpu_hot},
        "placement_verified": verified,
        "claim": (
            "The measured forward is one decoder block, not the Qwen draft. "
            "placement_verified requires Neural Engine matmul and linear ops in "
            "the CPU_AND_NE plan, none in the CPU_ONLY plan, and at least two "
            "mactop samples at or above 1 GB/s during that run only."
        ),
    }


def _gpu_forward(module, example) -> dict:
    import torch

    if not torch.backends.mps.is_available():
        return {"available": False, "seconds_per_forward": None, "reason": "mps unavailable"}
    model = _decoder().eval()
    model.load_state_dict(module.state_dict())
    model = model.to("mps")
    data = example.to("mps")
    with torch.no_grad():
        for _ in range(3):
            model(data)
        torch.mps.synchronize()
        started = time.perf_counter()
        for _ in range(30):
            model(data)
        torch.mps.synchronize()
    return {
        "available": True,
        "device": "mps",
        "seconds_per_forward": (time.perf_counter() - started) / 30,
    }


def _hybrid(ane_model, features: dict, module, example) -> dict:
    """Run GPU forwards while Core ML predicts, and sample both powers."""

    import threading

    import torch

    if not torch.backends.mps.is_available():
        return {"available": False, "reason": "mps unavailable"}
    model = _decoder().eval()
    model.load_state_dict(module.state_dict())
    model = model.to("mps")
    data = example.to("mps")
    stop = threading.Event()
    counts = {"ane": 0}

    def predict() -> None:
        while not stop.is_set():
            ane_model.predict(features)
            counts["ane"] += 1

    thread = threading.Thread(target=predict)
    thread.start()

    def gpu_forwards() -> int:
        completed = 0
        with torch.no_grad():
            deadline = time.perf_counter() + 3.0
            while time.perf_counter() < deadline:
                model(data)
                torch.mps.synchronize()
                completed += 1
        return completed

    try:
        observed = _mactop_while(gpu_forwards, seconds=3.0)
    finally:
        stop.set()
        thread.join()
    samples = observed["samples"]
    overlap = sum(
        (sample.get("ane_bw_combined_gbs") or 0) >= 1
        and (sample.get("gpu_power_w") or 0) >= 0.05
        for sample in samples
    )
    return {
        "available": True,
        "ane_predicts": counts["ane"],
        "gpu_forwards": observed.get("during"),
        "samples": samples,
        "overlap_samples": overlap,
        "both_active": overlap >= 3,
        "scope": "system-wide estimated power during concurrent GPU and Core ML calls",
    }


def _rank(measured: dict) -> dict:
    """Rank measured forwards. The Neural Engine stays out until placement is verified."""

    from .placement import choose_device

    seconds = {
        "cpu": measured["cpu_only"]["seconds_per_forward"],
        "gpu": measured["gpu"].get("seconds_per_forward"),
    }
    if measured["placement_verified"]:
        seconds["ane"] = measured["cpu_and_ne"]["seconds_per_forward"]
    measured["choice"] = choose_device(seconds)
    if not measured["placement_verified"]:
        hot = measured["ane_bandwidth_samples_at_least_1gbs"]["CPU_AND_NE"]
        measured["choice"]["ane_excluded"] = (
            f"ANE forward was measured and left out of the ranking "
            f"({hot} samples at or above 1 GB/s; 2 required)"
        )
    return measured


def rank_decoder() -> dict:
    """Measure the decoder and keep the Neural Engine out of the ranking until verified."""

    try:
        return _rank(decoder_workload())
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}


def _qwen_shaped():
    """28-layer network with the Qwen3-0.6B widths. Weights are random."""

    import torch
    import torch.nn as nn

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n1 = nn.RMSNorm(1024, eps=1e-6)
            self.n2 = nn.RMSNorm(1024, eps=1e-6)
            self.query = nn.Linear(1024, 2048, bias=False)
            self.key = nn.Linear(1024, 1024, bias=False)
            self.value = nn.Linear(1024, 1024, bias=False)
            self.output = nn.Linear(2048, 1024, bias=False)
            self.gate = nn.Linear(1024, 3072, bias=False)
            self.up = nn.Linear(1024, 3072, bias=False)
            self.down = nn.Linear(3072, 1024, bias=False)

        def forward(self, hidden):
            normed = self.n1(hidden)
            query = self.query(normed).reshape(1, 64, 16, 128).transpose(1, 2)
            key = self.key(normed).reshape(1, 64, 8, 128).transpose(1, 2)
            value = self.value(normed).reshape(1, 64, 8, 128).transpose(1, 2)
            key = torch.cat((key, key), dim=1)
            value = torch.cat((value, value), dim=1)
            scores = torch.matmul(query, key.transpose(-2, -1)) * (128**-0.5)
            weights = torch.softmax(scores, dim=-1)
            context = torch.matmul(weights, value).transpose(1, 2).reshape(1, 64, 2048)
            residual = hidden + self.output(context)
            normed = self.n2(residual)
            activated = torch.nn.functional.silu(self.gate(normed))
            return residual + self.down(activated * self.up(normed))

    class Network(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([Layer() for _ in range(28)])
            self.norm = nn.RMSNorm(1024, eps=1e-6)
            self.head = nn.Linear(1024, 151936, bias=False)

        def forward(self, hidden):
            for layer in self.layers:
                hidden = layer(hidden)
            last = self.norm(hidden)[:, 63, :]
            return self.head(last)

    torch.manual_seed(0)
    network = Network().eval()
    with torch.no_grad():
        for name, param in network.named_parameters():
            if "norm" not in name:
                param.mul_(0.02)
    return network


def _brief(rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    cpu_ops: list[str] = []
    ne_compute = 0
    for row in rows:
        device = row["device"] or "unassigned"
        counts[device] = counts.get(device, 0) + 1
        if row["device"] == "MLCPUComputeDevice":
            cpu_ops.append(row["operator"])
        if row["device"] == "MLNeuralEngineComputeDevice" and row["operator"] in {
            "ios17.linear",
            "ios17.matmul",
        }:
            ne_compute += 1
    return {"counts": counts, "cpu_operators": cpu_ops, "ne_linear_or_matmul": ne_compute}


def _time_mps(module, example, repeats: int) -> dict:
    import copy

    import torch

    if not torch.backends.mps.is_available():
        return {"available": False, "seconds_per_forward": None, "reason": "mps unavailable"}
    model = copy.deepcopy(module).to("mps").eval()
    data = example.to("mps")
    with torch.no_grad():
        model(data)
        torch.mps.synchronize()
        started = time.perf_counter()
        for _ in range(repeats):
            model(data)
        torch.mps.synchronize()
    seconds = (time.perf_counter() - started) / repeats
    del model, data
    return {"available": True, "device": "mps", "seconds_per_forward": seconds}


def qwen_forward_workload() -> dict:
    """Time one full-depth forward at the Qwen3-0.6B widths.

    Input is 64 hidden states of width 1024. Output is the next-token logits
    over the 151936-word vocabulary. The weights are random. The pinned Qwen3
    checkpoint is measured separately by ``macreduce infer``.
    """

    import shutil
    import tempfile

    import coremltools as ct
    import numpy as np
    import torch
    from coremltools.models import CompiledMLModel
    from coremltools.models.compute_plan import MLComputePlan
    from coremltools.models.utils import compile_model

    free = shutil.disk_usage(Path.cwd()).free
    if free < 3_000_000_000:
        raise OSError(
            f"{free} bytes free; the 28-layer package needs about 3 GB "
            "for the compiler's copy of the weights"
        )
    network = _qwen_shaped()
    example = torch.randn(1, 64, 1024)
    with torch.no_grad():
        reference = network(example).detach().numpy()
    parameters = sum(param.numel() for param in network.parameters())
    gpu = _time_mps(network, example, 8)
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory)
        traced = torch.jit.trace(network.eval(), example)
        converted = ct.convert(
            traced,
            inputs=[ct.TensorType(name="x", shape=(1, 64, 1024), dtype=np.float16)],
            compute_precision=ct.precision.FLOAT16,
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.macOS14,
        )
        package = destination / "qwen.mlpackage"
        converted.save(str(package))
        compiled = str(
            compile_model(str(package), destination_path=str(destination / "qwen.mlmodelc"))
        )
        shutil.rmtree(package)
        plans = {
            name: _brief(
                _assignments(
                    MLComputePlan.load_from_path(
                        compiled, compute_units=getattr(ct.ComputeUnit, name)
                    )
                )
            )
            for name in ("CPU_AND_NE", "CPU_ONLY")
        }
        features = {"x": np.ascontiguousarray(example.numpy().astype(np.float16))}
        ane_model = CompiledMLModel(compiled, compute_units=ct.ComputeUnit.CPU_AND_NE)
        cpu_model = CompiledMLModel(compiled, compute_units=ct.ComputeUnit.CPU_ONLY)
        predicted = np.array(next(iter(ane_model.predict(features).values())), dtype=np.float32)
        cpu_seconds = _time_predicts(cpu_model, features, 8)
        ane_seconds = _time_predicts(ane_model, features, 8)

        def sustain(model) -> int:
            completed = 0
            deadline = time.perf_counter() + 4.0
            while time.perf_counter() < deadline:
                model.predict(features)
                completed += 1
            return completed

        cpu_trace = _mactop_while(lambda model=cpu_model: sustain(model), seconds=4.0)
        ane_trace = _mactop_while(lambda model=ane_model: sustain(model), seconds=4.0)
        del ane_model, cpu_model
    ane_hot = _hot(ane_trace["samples"], "ane_bw_combined_gbs", 1.0)
    cpu_hot = _hot(cpu_trace["samples"], "ane_bw_combined_gbs", 1.0)
    verified = (
        plans["CPU_AND_NE"]["ne_linear_or_matmul"] > 0
        and plans["CPU_ONLY"]["ne_linear_or_matmul"] == 0
        and ane_hot >= 2
        and cpu_hot == 0
    )
    return {
        "status": "ok",
        "coremltools": metadata.version("coremltools"),
        "host": platform.platform(),
        "workload": {
            "kind": "qwen3_0_6b_shaped_forward",
            "layers": 28,
            "tokens": 64,
            "width": 1024,
            "heads": 16,
            "kv_heads": 8,
            "head_dim": 128,
            "mlp_width": 3072,
            "vocab": 151936,
            "parameters": parameters,
            "precision": "float16",
            "weights": "random, scaled by 0.02",
            "qwen_draft_weights": False,
        },
        "plans": plans,
        "parity_max_abs": float(np.max(np.abs(reference - predicted))),
        "cpu_only": {"seconds_per_forward": cpu_seconds},
        "cpu_and_ne": {"seconds_per_forward": ane_seconds},
        "gpu": gpu,
        "mactop": {"CPU_ONLY": cpu_trace, "CPU_AND_NE": ane_trace},
        "ane_bandwidth_samples_at_least_1gbs": {"CPU_AND_NE": ane_hot, "CPU_ONLY": cpu_hot},
        "placement_verified": verified,
        "claim": (
            "One forward is 64 positions through 28 Qwen3-0.6B-width layers "
            "plus the vocabulary projection. Weights are random. "
            "placement_verified requires Neural Engine linear or matmul ops, "
            "none of those on the CPU-only plan, and at least two mactop samples "
            "at or above 1 GB/s during the Neural Engine run only."
        ),
    }


def rank_qwen_forward() -> dict:
    """Measure the full-depth forward and rank the devices that were verified."""

    try:
        return _rank(qwen_forward_workload())
    except Exception as exc:
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
