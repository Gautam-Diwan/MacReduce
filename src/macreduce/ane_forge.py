"""A linear step on the Neural Engine through ANEForge.

``import aneforge as af`` builds the graph. The compiled forward, backward,
and Adam programs use device mask ``0x4``, which ANEForge defines as the
Apple Neural Engine. This is not the Qwen3 LoRA client.
"""

from __future__ import annotations

import json
import tempfile
import threading
from pathlib import Path

import numpy as np

ANE_MASK = 0x4


def prove_engine(seconds: int = 4, seed: int = 0) -> dict:
    """Compile a matmul for the Neural Engine and keep it busy while sampling.

    The result is a failure when every mactop sample reports zero Neural
    Engine power and zero Neural Engine bandwidth. ``ane_active`` is recorded
    and is not treated as residency.
    """

    if seconds < 1:
        raise ValueError("sample window must be positive")
    trained = run_linear(steps=4, seed=seed)
    import aneforge as af

    rng = np.random.default_rng(seed)
    width = 256
    features = af.input((32, width))
    weight = rng.standard_normal((width, width)).astype(np.float32)
    program = af.compile(features @ weight)
    batch = rng.standard_normal((32, width)).astype(np.float32)
    calls = {"n": 0}
    stop = threading.Event()

    def burn():
        while not stop.is_set():
            program(batch)
            calls["n"] += 1

    thread = threading.Thread(target=burn)
    thread.start()
    from .observe import collect_mactop

    with tempfile.TemporaryDirectory(prefix="macreduce-ane-forge-") as directory:
        status = collect_mactop(directory, seconds=seconds, interval_ms=500)
        raw_path = Path(directory) / "mactop.raw.json"
        raw = raw_path.read_text() if raw_path.is_file() else ""
    stop.set()
    thread.join()
    samples = [_ane_sample(row) for row in _mactop_rows(raw)]
    power = [row["ane_power_w"] for row in samples if row["ane_power_w"] is not None]
    bandwidth = [row["ane_bw_gbs"] for row in samples if row["ane_bw_gbs"] is not None]
    if not samples or (max(power or [0]) <= 0 and max(bandwidth or [0]) <= 0):
        raise RuntimeError("mactop showed no Neural Engine power or bandwidth during the run")
    trained["matmul_calls"] = calls["n"]
    trained["mactop_status"] = status.get("status")
    trained["mactop"] = samples
    trained["ane_power_w_max"] = max(power)
    trained["ane_bw_gbs_max"] = max(bandwidth)
    return trained


def _mactop_rows(text: str) -> list:
    decoder = json.JSONDecoder()
    rows = []
    remaining = text.lstrip()
    while remaining.strip():
        value, end = decoder.raw_decode(remaining)
        rows.extend(value if isinstance(value, list) else [value])
        remaining = remaining[end:].lstrip()
    return rows


def _ane_sample(row: dict) -> dict:
    soc = row.get("soc_metrics", {})
    return {
        "ane_power_w": soc.get("ane_power"),
        "ane_bw_gbs": soc.get("ane_bw_combined_gbs"),
        "ane_active": soc.get("ane_active"),
        "gpu_power_w": soc.get("gpu_power"),
    }


def run_linear(steps: int = 20, seed: int = 0, sample_seconds: int = 0) -> dict:
    """Train a tiny linear map for ``steps`` and return the engine masks.

    ``sample_seconds`` collects system mactop while the steps run. A missing
    sampler does not turn a finished engine step into a failure.
    """

    if steps < 1:
        raise ValueError("steps must be positive")
    from importlib.metadata import version

    import aneforge as af

    rng = np.random.default_rng(seed)
    batch, width = 8, 16
    features = af.input((batch, width))
    weight = af.parameter(rng.standard_normal((width, 1)).astype(np.float32) * 0.1)
    prediction = features @ weight
    target = af.input((batch, 1))
    residual = prediction - target
    loss = (residual * residual).mean((0, 1))
    before = np.array(weight.attrs["value"], copy=True)
    trainer = af.Trainer(loss, [weight], lr=0.05, optimizer="adam", device_optimizer=True)
    inputs = rng.standard_normal((64, width)).astype(np.float32)
    slope = np.linspace(-0.2, 0.2, width).astype(np.float32).reshape(width, 1)
    labels = inputs @ slope
    trainer.set_dataset(features, inputs, target, labels)
    losses = []
    samples = []
    if sample_seconds > 0:
        holder = {}

        def _sample():
            from .observe import collect_mactop

            with tempfile.TemporaryDirectory(prefix="macreduce-ane-forge-") as directory:
                holder["status"] = collect_mactop(directory, seconds=sample_seconds)
                path = Path(directory) / "mactop.normalized.jsonl"
                if path.is_file():
                    holder["rows"] = [json.loads(line) for line in path.read_text().splitlines() if line]

        thread = threading.Thread(target=_sample)
        thread.start()
        while thread.is_alive():
            trainer.step()
            losses.append(float(trainer.loss()))
        thread.join()
        samples = [
            {"ane_power_w": row.get("ane_power_w"), "gpu_power_w": row.get("gpu_power_w")}
            for row in holder.get("rows", [])
        ]
    else:
        for _ in range(steps):
            trainer.step()
            losses.append(float(trainer.loss()))
    after = np.array(weight.attrs["value"], copy=True)
    masks = {
        "forward": int(trainer._fwd._prog._device_mask),
        "backward": [int(item._prog._device_mask) for item in trainer._bwd],
        "update": [int(item._prog._device_mask) for item in trainer._upd],
    }
    flat = [masks["forward"], *masks["backward"], *masks["update"]]
    if any(mask != ANE_MASK for mask in flat):
        raise RuntimeError(f"a program was compiled off the Neural Engine: {masks}")
    trainer.release()
    return {
        "status": "ok",
        "import": "aneforge as af",
        "aneforge": version("aneforge"),
        "device_mask": hex(ANE_MASK),
        "device_mask_meaning": "Apple Neural Engine",
        "programs": masks,
        "steps": len(losses),
        "losses": losses,
        "weight_max_abs_move": float(np.max(np.abs(after - before))),
        "mactop": samples,
    }


class ResidentEngine:
    """Keep a compiled Neural Engine graph running until :meth:`stop`.

    The Qwen adapter stays on the GPU. This graph is the ANEForge linear
    Adam step plus two 512-by-2048 matmuls in one program. Both must carry
    device mask ``0x4``. A compile that lands anywhere else raises.
    """

    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        self.error: str | None = None
        self.summary: dict | None = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, timeout: float = 180) -> None:
        self._thread = threading.Thread(target=self._run, name="ane-resident")
        self._thread.start()
        if not self._ready.wait(timeout):
            self._stop.set()
            self._thread.join(timeout=5)
            raise RuntimeError(self.error or "Neural Engine graph was not ready")
        if self.error:
            self._stop.set()
            self._thread.join()
            raise RuntimeError(self.error)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        if self.error:
            raise RuntimeError(self.error)
        return self.summary or {}

    def _run(self) -> None:
        try:
            self.summary = hold_engine(self._stop, seed=self.seed, ready=self._ready)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._ready.set()


def hold_engine(stop: threading.Event, seed: int = 0, ready: threading.Event | None = None) -> dict:
    """Run the engine graph until ``stop`` is set. Signal ``ready`` after the mask check."""

    import aneforge as af

    trained = run_linear(steps=4, seed=seed)
    rng = np.random.default_rng(seed)
    batch_rows, width = 512, 2048
    features = af.input((batch_rows, width))
    weight = rng.standard_normal((width, width)).astype(np.float32) * 0.01
    program = af.compile((features @ weight) @ weight)
    mask = int(program._prog._device_mask)
    if mask != ANE_MASK:
        program.release()
        raise RuntimeError(f"matmul compiled off the Neural Engine: {hex(mask)}")
    if ready is not None:
        ready.set()
    batch = rng.standard_normal((batch_rows, width)).astype(np.float32)
    calls = 0
    try:
        while not stop.is_set():
            program(batch)
            calls += 1
    finally:
        program.release()
    return {
        "status": "ok",
        "qwen_lora": False,
        "model": "ANEForge linear 16x1 Adam, then two resident 512x2048 matmuls",
        "aneforge": trained["aneforge"],
        "device_mask": hex(ANE_MASK),
        "programs": trained["programs"],
        "matmul_device_mask": hex(mask),
        "linear_steps": trained["steps"],
        "linear_losses": trained["losses"],
        "weight_max_abs_move": trained["weight_max_abs_move"],
        "matmul_calls": calls,
    }
