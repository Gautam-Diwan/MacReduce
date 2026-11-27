"""
Measure how recorded federated LoRA deltas respond to wire codecs.

Reads client ``delta.npz`` files and consecutive coordinator adapters, then
reports energy layout and the error of fp16, bf16 and affine int8/int4/int2
(group 64 along the long axis, the MLX ``mx.quantize`` layout). Error feedback
is simulated across the recorded rounds. This is an offline codec study of
already-trained deltas; it says nothing about convergence under compression.

    uv run python tools/delta_codecs.py runs/federation-gpu --out evidence/<dir>/codecs.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
CODECS = {"fp16": 16, "bf16": 16, "int8": 8.5, "int4": 4.5, "int2": 2.5}


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].astype(np.float64) for key in data.files}


def affine(x: np.ndarray, bits: int, group: int = 64) -> np.ndarray:
    tall = x.shape[-1] < x.shape[0]
    t = x.T if tall else x
    flat = t.reshape(-1)
    pad = (-flat.size) % group
    rows = np.concatenate([flat, np.zeros(pad)]).reshape(-1, group)
    low, high = rows.min(1, keepdims=True), rows.max(1, keepdims=True)
    scale = np.where(high > low, (high - low) / (2**bits - 1), 1.0)
    out = (np.round((rows - low) / scale) * scale + low).reshape(-1)[:flat.size].reshape(t.shape)
    return out.T if tall else out


def bf16(x: np.ndarray) -> np.ndarray:
    bits = x.astype(np.float32).view(np.uint32)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000
    return rounded.astype(np.uint32).view(np.float32).astype(np.float64)


def encode(x: np.ndarray, codec: str) -> np.ndarray:
    if codec == "fp16":
        return x.astype(np.float16).astype(np.float64)
    if codec == "bf16":
        return bf16(x)
    return affine(x, int(codec[3:]))


def error(reference: dict, approx: dict) -> dict:
    num = sum(((approx[k] - reference[k]) ** 2).sum() for k in reference)
    den = sum((reference[k] ** 2).sum() for k in reference)
    dot = sum((approx[k] * reference[k]).sum() for k in reference)
    norm = sum((approx[k] ** 2).sum() for k in reference)
    return {"relative_l2": float(np.sqrt(num / den)), "cosine": float(dot / np.sqrt(den * norm))}


def layout(delta: dict) -> dict:
    total = sum((v**2).sum() for v in delta.values())
    factor, module, layer = defaultdict(float), defaultdict(float), defaultdict(float)
    for name, value in delta.items():
        energy = (value**2).sum() / total
        factor["B" if name.endswith("lora_b") else "A"] += energy
        module[next(m for m in MODULES if m in name)] += energy
        layer[int(re.search(r"layers\.(\d+)\.", name).group(1))] += energy
    squares = np.sort(np.concatenate([v.ravel() for v in delta.values()]) ** 2)[::-1]
    shares = [layer[i] for i in sorted(layer)]
    return {
        "l2": float(np.sqrt(total)), "max_abs": float(max(np.abs(v).max() for v in delta.values())),
        "top1pct_energy": float(squares[:squares.size // 100].sum() / squares.sum()),
        "top10pct_energy": float(squares[:squares.size // 10].sum() / squares.sum()),
        "factor_energy": dict(factor), "module_energy": dict(module),
        "layer_energy_min": float(min(shares)), "layer_energy_max": float(max(shares)),
    }


def feedback(deltas: list[dict], codec: str) -> dict:
    residual = {k: np.zeros_like(v) for k, v in deltas[0].items()}
    true, sent, plain = (dict.fromkeys(deltas[0], 0.0) for _ in range(3))
    for delta in deltas:
        for key, value in delta.items():
            target = value + residual[key]
            coded = encode(target, codec)
            residual[key] = target - coded
            true[key] = true[key] + value
            sent[key] = sent[key] + coded
            plain[key] = plain[key] + encode(value, codec)
    return {"with_feedback": error(true, sent)["relative_l2"],
            "without_feedback": error(true, plain)["relative_l2"]}


def study(deltas: list[dict]) -> dict:
    return {
        "rounds": [layout(d) for d in deltas],
        "codecs": {c: [error(d, {k: encode(v, c) for k, v in d.items()}) for d in deltas]
                   for c in CODECS},
        "cumulative_error_feedback": {c: feedback(deltas, c) for c in ("int8", "int4", "int2")},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("runs", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    clients = {path.name: [load(p) for p in sorted(path.glob("round-*/delta.npz"))]
               for path in sorted(args.runs.iterdir()) if any(path.glob("round-*/delta.npz"))}
    adapters = [load(p) for p in sorted((args.runs / "coordinator").glob("adapter-*.npz"))]
    merged = [{k: b[k] - a[k] for k in a} for a, b in itertools.pairwise(adapters)]
    params = sum(v.size for v in adapters[0].values())
    report = {
        "parameters": int(params),
        "wire_mib": {c: params * bits / 8 / 2**20 for c, bits in {"fp32": 32, **CODECS}.items()},
        "clients": {name: study(deltas) for name, deltas in clients.items()},
        "merged": study(merged),
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
