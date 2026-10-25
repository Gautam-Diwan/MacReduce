"""One-block causal byte Transformer with a shared MLX/PyTorch parameter layout.

The CPU backend is an explicit reference backend, never a silent GPU fallback.
All model tensors and checkpoint arithmetic use float32 in this first version.
"""

import hashlib
import io
import zipfile
from pathlib import Path

import numpy as np

MODEL = {"width": 32, "heads": 4, "sequence": 32, "vocab": 256}


def initial_weights(config=MODEL, seed=7):
    rng = np.random.default_rng(seed)
    d = config["width"]
    shapes = {
        "embedding": (256, d),
        "position": (config["sequence"], d),
        "q": (d, d),
        "k": (d, d),
        "v": (d, d),
        "out": (d, d),
        "ff1": (d, 4 * d),
        "ff2": (4 * d, d),
        "head": (d, 256),
    }
    p = {
        k: (
            rng.standard_normal(s) * (0.02 if k in ("embedding", "position") else s[0] ** -0.5)
        ).astype(np.float32)
        for k, s in shapes.items()
    }
    for n in ("norm1", "norm2", "norm3"):
        p[n + "_g"] = np.ones(d, dtype=np.float32)
        p[n + "_b"] = np.zeros(d, dtype=np.float32)
    return p


def encode_weights(weights):
    out = io.BytesIO()
    np.savez(out, **weights)  # No pickle; deliberately uncompressed for a measured baseline.
    return out.getvalue()


def decode_weights(raw, reference):
    limit = sum(x.nbytes for x in reference.values()) + 4096 * len(reference)
    if len(raw) > limit:
        raise ValueError("checkpoint exceeds model size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            entries = z.infolist()
            if len(entries) != len(reference) or sum(i.file_size for i in entries) > limit:
                raise ValueError("invalid checkpoint archive")
            if len({i.filename for i in entries}) != len(entries):
                raise ValueError("duplicate tensors")
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            if set(z.files) != set(reference):
                raise ValueError("parameter names do not match")
            result = {}
            for k, expected in reference.items():
                v = z[k]
                if v.shape != expected.shape or v.dtype != np.float32:
                    raise ValueError(f"invalid shape or dtype: {k}")
                if not np.isfinite(v).all():
                    raise ValueError(f"non-finite parameter: {k}")
                result[k] = v.copy()
        return result
    except (zipfile.BadZipFile, OSError, EOFError) as exc:
        raise ValueError("invalid checkpoint") from exc


def weighted_average(updates):
    """updates: sequence of (validated parameter dict, actual processed token count)."""
    if not updates or any(type(n) is not int or n <= 0 for _, n in updates):
        raise ValueError("positive integer token counts required")
    total = sum(n for _, n in updates)
    reference = updates[0][0]
    for p, _ in updates:
        if set(p) != set(reference) or any(p[k].shape != reference[k].shape for k in p):
            raise ValueError("incompatible parameter layouts")
        if any(v.dtype != np.float32 or not np.isfinite(v).all() for v in p.values()):
            raise ValueError("finite float32 parameters required")
    return {
        k: sum(p[k].astype(np.float64) * (n / total) for p, n in updates).astype(np.float32)
        for k in reference
    }


def corpus(seed=1, path=None):
    if path:
        raw = Path(path).read_bytes()
    else:
        rng = np.random.default_rng(seed)
        subjects = ["the cat", "a bird", "the robot", "a child", "the dog", "a fox"]
        verbs = ["sees", "finds", "likes", "follows"]
        places = ["near the tree", "in the park", "by the river", "at home"]
        raw = "".join(
            f"{rng.choice(subjects)} {rng.choice(subjects)} {rng.choice(verbs)} "
            f"{rng.choice(places)}.\n"
            for _ in range(2000)
        ).encode()
    if len(raw) < 1024:
        raise ValueError("training/validation text must contain at least 1024 bytes")
    return np.frombuffer(raw, dtype=np.uint8), hashlib.sha256(raw).hexdigest()


def batch(data, rng, batch_size, sequence):
    starts = rng.integers(0, len(data) - sequence, size=batch_size)
    indices = starts[:, None] + np.arange(sequence + 1)[None, :]
    values = data[indices].astype(np.int64)
    return values[:, :-1], values[:, 1:]


class TorchBackend:
    name = "torch-cpu"

    def __init__(self, config, weights):
        import torch

        self.t = torch
        torch.set_num_threads(1)
        self.config = config
        self.load(weights)

    def load(self, weights):
        self.p = {k: self.t.tensor(v, requires_grad=True) for k, v in weights.items()}

    def forward(self, x):
        t, p = self.t, self.p
        b, s = x.shape
        d, h = self.config["width"], self.config["heads"]

        def norm(v, n):
            return t.nn.functional.layer_norm(v, (d,), p[n + "_g"], p[n + "_b"], 1e-5)

        x = p["embedding"][x] + p["position"][:s]
        z = norm(x, "norm1")
        q, k, v = [(z @ p[n]).reshape(b, s, h, d // h).transpose(1, 2) for n in ("q", "k", "v")]
        mask = t.triu(t.full((s, s), float("-inf")), diagonal=1)
        a = t.softmax((q @ k.transpose(-1, -2)) / (d // h) ** 0.5 + mask, dim=-1)
        x = x + (a @ v).transpose(1, 2).reshape(b, s, d) @ p["out"]
        x = x + t.relu(norm(x, "norm2") @ p["ff1"]) @ p["ff2"]
        return norm(x, "norm3") @ p["head"]

    def loss(self, x, y):
        logits = self.forward(self.t.tensor(x))
        return self.t.nn.functional.cross_entropy(
            logits.reshape(-1, 256), self.t.tensor(y).reshape(-1)
        )

    def step(self, x, y, lr):
        loss = self.loss(x, y)
        loss.backward()
        with self.t.no_grad():
            for v in self.p.values():
                v -= lr * v.grad
                v.grad = None
        return loss.item()

    def evaluate(self, x, y):
        with self.t.no_grad():
            return self.loss(x, y).item()

    def export(self):
        return {k: v.detach().numpy().copy() for k, v in self.p.items()}

    def memory(self):
        return {
            "allocator_active_bytes": None,
            "allocator_peak_bytes": None,
            "allocator_source": "torch CPU allocator not instrumented",
        }


class MLXBackend:
    def __init__(self, config, weights, device="gpu"):
        import mlx.core as mx
        import mlx.nn as nn

        self.mx, self.nn, self.config = mx, nn, config
        self.name = "mlx-" + device
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        self.load(weights)
        self.grad = mx.value_and_grad(self.loss)

    def load(self, weights):
        self.p = {k: self.mx.array(v) for k, v in weights.items()}
        self.mx.eval(self.p)

    def loss(self, p, x, y):
        mx = self.mx
        b, s = x.shape
        d, h = self.config["width"], self.config["heads"]

        def norm(v, n):
            mean = mx.mean(v, axis=-1, keepdims=True)
            var = mx.mean(mx.square(v - mean), axis=-1, keepdims=True)
            return (v - mean) * mx.rsqrt(var + 1e-5) * p[n + "_g"] + p[n + "_b"]

        x = p["embedding"][x] + p["position"][:s]
        z = norm(x, "norm1")
        q, k, v = [
            (z @ p[n]).reshape(b, s, h, d // h).transpose(0, 2, 1, 3) for n in ("q", "k", "v")
        ]
        mask = mx.triu(mx.full((s, s), -float("inf")), k=1)
        a = mx.softmax((q @ k.transpose(0, 1, 3, 2)) / (d // h) ** 0.5 + mask, axis=-1)
        x = x + (a @ v).transpose(0, 2, 1, 3).reshape(b, s, d) @ p["out"]
        x = x + mx.maximum(norm(x, "norm2") @ p["ff1"], 0) @ p["ff2"]
        logits = norm(x, "norm3") @ p["head"]
        return self.nn.losses.cross_entropy(logits, y, reduction="mean")

    def step(self, x, y, lr):
        loss, grad = self.grad(self.p, self.mx.array(x), self.mx.array(y))
        self.p = {k: v - lr * grad[k] for k, v in self.p.items()}
        self.mx.eval(loss, self.p)  # Include completed work, not just lazy dispatch, in timings.
        return loss.item()

    def evaluate(self, x, y):
        return self.loss(self.p, self.mx.array(x), self.mx.array(y)).item()

    def export(self):
        self.mx.eval(self.p)
        return {k: np.array(v) for k, v in self.p.items()}

    def memory(self):
        return {
            "allocator_active_bytes": self.mx.get_active_memory(),
            "allocator_peak_bytes": self.mx.get_peak_memory(),
            "allocator_source": "MLX process allocator, not total system memory",
        }


def backend(name, config, weights):
    if name == "torch-cpu":
        return TorchBackend(config, weights)
    if name in ("mlx-cpu", "mlx-gpu"):
        return MLXBackend(config, weights, name.split("-")[1])
    raise ValueError(f"backend {name!r} is not implemented; no implicit fallback")
