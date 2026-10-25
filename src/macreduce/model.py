"""Causal byte Transformer with a shared MLX/PyTorch parameter layout.

The default configuration is the original one-block, 29,888-parameter model.
The CPU backend is an explicit reference backend, never a silent GPU fallback.
Checkpoints are NumPy archives. Float32 is the only implemented dtype.
"""

import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CONFIG_KEY = "__model_config__"
# Legacy field names accepted by ModelConfig.from_mapping. Call sites that need
# the active run must pass that run's ModelConfig rather than this constant.
MODEL = {"width": 32, "heads": 4, "sequence": 32, "vocab": 256}


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab: int = 256
    width: int = 32
    heads: int = 4
    layers: int = 1
    context: int = 32
    dtype: str = "float32"
    tokenizer: str = "utf-8-bytes"

    def __post_init__(self):
        checks = {
            "vocab": (self.vocab, 2, 1024),
            "width": (self.width, 1, 256),
            "heads": (self.heads, 1, 256),
            "layers": (self.layers, 1, 8),
            "context": (self.context, 1, 256),
        }
        for name, (value, low, high) in checks.items():
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer from {low} to {high}")
        if self.width % self.heads != 0:
            raise ValueError("width must be divisible by heads")
        if self.dtype != "float32":
            raise ValueError("only float32 checkpoints are implemented")
        if self.tokenizer != "utf-8-bytes":
            raise ValueError("tokenizer must be utf-8-bytes")

    def to_dict(self):
        return {
            "vocab": self.vocab,
            "width": self.width,
            "heads": self.heads,
            "layers": self.layers,
            "context": self.context,
            "dtype": self.dtype,
            "tokenizer": self.tokenizer,
        }

    @staticmethod
    def from_mapping(value=None):
        if value is None:
            return ModelConfig()
        if isinstance(value, ModelConfig):
            return value
        if not isinstance(value, dict):
            raise ValueError("model config must be a mapping")
        data = dict(value)
        if "sequence" in data:
            sequence = data.pop("sequence")
            if "context" in data and data["context"] != sequence:
                raise ValueError("context and sequence disagree")
            data.setdefault("context", sequence)
        allowed = {"vocab", "width", "heads", "layers", "context", "dtype", "tokenizer"}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown model config fields: {sorted(unknown)}")
        defaults = ModelConfig()
        return ModelConfig(**{key: data.get(key, getattr(defaults, key)) for key in allowed})


DEFAULT_CONFIG = ModelConfig()


def layer_param(config, layer, name):
    config = ModelConfig.from_mapping(config)
    if config.layers == 1 or name in ("norm3_g", "norm3_b"):
        return name
    return f"b{layer}_{name}"


def parameter_specs(config):
    """Deterministic parameter order shared by initialization and checkpoints."""
    config = ModelConfig.from_mapping(config)
    width = config.width
    specs = [
        ("embedding", (config.vocab, width), "embed"),
        ("position", (config.context, width), "embed"),
    ]

    def projections(prefix):
        specs.extend(
            [
                (prefix + "q", (width, width), "proj"),
                (prefix + "k", (width, width), "proj"),
                (prefix + "v", (width, width), "proj"),
                (prefix + "out", (width, width), "proj"),
                (prefix + "ff1", (width, 4 * width), "proj"),
                (prefix + "ff2", (4 * width, width), "proj"),
            ]
        )

    if config.layers == 1:
        projections("")
    else:
        for layer in range(config.layers):
            projections(f"b{layer}_")
    specs.append(("head", (width, config.vocab), "proj"))

    def norms(prefix, names):
        for name in names:
            specs.append((prefix + name + "_g", (width,), "one"))
            specs.append((prefix + name + "_b", (width,), "zero"))

    if config.layers == 1:
        norms("", ("norm1", "norm2", "norm3"))
    else:
        for layer in range(config.layers):
            norms(f"b{layer}_", ("norm1", "norm2"))
        norms("", ("norm3",))
    return specs


def initial_weights(config=None, seed=7):
    config = ModelConfig.from_mapping(config)
    rng = np.random.default_rng(seed)
    weights = {}
    for name, shape, kind in parameter_specs(config):
        if kind == "one":
            weights[name] = np.ones(shape, dtype=np.float32)
        elif kind == "zero":
            weights[name] = np.zeros(shape, dtype=np.float32)
        else:
            scale = 0.02 if kind == "embed" else shape[0] ** -0.5
            weights[name] = (rng.standard_normal(shape) * scale).astype(np.float32)
    return weights


def encode_weights(weights, config=None):
    payload = dict(weights)
    if config is not None:
        raw = json.dumps(
            ModelConfig.from_mapping(config).to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        payload[CONFIG_KEY] = np.frombuffer(raw, dtype=np.uint8).copy()
    out = io.BytesIO()
    np.savez(out, **payload)  # No pickle; deliberately uncompressed for a measured baseline.
    return out.getvalue()


def decode_weights(raw, reference=None, config=None):
    if config is not None:
        config = ModelConfig.from_mapping(config)
        expected = {name: shape for name, shape, _kind in parameter_specs(config)}
    elif reference is not None:
        expected = {key: tuple(np.asarray(value).shape) for key, value in reference.items()}
    else:
        raise ValueError("checkpoint reference required")
    if CONFIG_KEY in expected:
        raise ValueError("invalid parameter name")
    limit = sum(int(np.prod(shape, dtype=np.int64)) * 4 for shape in expected.values()) + 4096 * (
        len(expected) + 1
    )
    if len(raw) > limit:
        raise ValueError("checkpoint exceeds model size limit")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            entries = archive.infolist()
            if sum(info.file_size for info in entries) > limit:
                raise ValueError("invalid checkpoint archive")
            filenames = [info.filename for info in entries]
            if len(set(filenames)) != len(filenames):
                raise ValueError("duplicate tensors")
            keys = []
            for name in filenames:
                if not name.endswith(".npy"):
                    raise ValueError("invalid checkpoint archive")
                keys.append(name[:-4])
            params = [key for key in keys if key != CONFIG_KEY]
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            embedded = None
            if CONFIG_KEY in archive.files:
                embedded = ModelConfig.from_mapping(
                    json.loads(archive[CONFIG_KEY].tobytes().decode())
                )
            if config is not None and embedded is not None and embedded != config:
                raise ValueError("checkpoint model config mismatch")
            if set(params) != set(expected):
                raise ValueError("parameter names do not match")
            result = {}
            for key, shape in expected.items():
                value = archive[key]
                if value.shape != shape or value.dtype != np.float32:
                    raise ValueError(f"invalid shape or dtype: {key}")
                if not np.isfinite(value).all():
                    raise ValueError(f"non-finite parameter: {key}")
                result[key] = value.copy()
        return result
    except (zipfile.BadZipFile, OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid checkpoint") from exc


def weighted_average(updates):
    """updates: sequence of (validated parameter dict, actual processed token count)."""
    if not updates or any(type(n) is not int or n <= 0 for _, n in updates):
        raise ValueError("positive integer token counts required")
    total = sum(n for _, n in updates)
    reference = updates[0][0]
    for params, _count in updates:
        if set(params) != set(reference) or any(
            params[key].shape != reference[key].shape for key in params
        ):
            raise ValueError("incompatible parameter layouts")
        if any(value.dtype != np.float32 or not np.isfinite(value).all() for value in params.values()):
            raise ValueError("finite float32 parameters required")
    return {
        key: sum(params[key].astype(np.float64) * (count / total) for params, count in updates).astype(
            np.float32
        )
        for key in reference
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
        self.config = ModelConfig.from_mapping(config)
        self.load(weights)

    def load(self, weights):
        self.p = {key: self.t.tensor(value, requires_grad=True) for key, value in weights.items()}

    def _norm(self, value, gain, bias):
        width = self.config.width
        return self.t.nn.functional.layer_norm(value, (width,), gain, bias, 1e-5)

    def forward(self, tokens):
        torch, params = self.t, self.p
        batch_size, steps = tokens.shape
        config = self.config
        width, heads = config.width, config.heads
        head_dim = width // heads
        hidden = params["embedding"][tokens] + params["position"][:steps]
        for layer in range(config.layers):

            def name(suffix, layer=layer):
                return params[layer_param(config, layer, suffix)]

            normalized = self._norm(hidden, name("norm1_g"), name("norm1_b"))
            query, key, value = [
                (normalized @ name(part)).reshape(batch_size, steps, heads, head_dim).transpose(1, 2)
                for part in ("q", "k", "v")
            ]
            mask = torch.triu(torch.full((steps, steps), float("-inf")), diagonal=1)
            scores = torch.softmax(
                (query @ key.transpose(-1, -2)) / head_dim**0.5 + mask, dim=-1
            )
            attended = (scores @ value).transpose(1, 2).reshape(batch_size, steps, width)
            hidden = hidden + attended @ name("out")
            feed = self._norm(hidden, name("norm2_g"), name("norm2_b"))
            hidden = hidden + torch.relu(feed @ name("ff1")) @ name("ff2")
        hidden = self._norm(hidden, params["norm3_g"], params["norm3_b"])
        return hidden @ params["head"]

    def loss(self, tokens, target):
        logits = self.forward(self.t.tensor(tokens))
        return self.t.nn.functional.cross_entropy(
            logits.reshape(-1, self.config.vocab), self.t.tensor(target).reshape(-1)
        )

    def step(self, tokens, target, lr):
        loss = self.loss(tokens, target)
        loss.backward()
        with self.t.no_grad():
            for value in self.p.values():
                value -= lr * value.grad
                value.grad = None
        return loss.item()

    def evaluate(self, tokens, target):
        with self.t.no_grad():
            return self.loss(tokens, target).item()

    def export(self):
        return {key: value.detach().numpy().copy() for key, value in self.p.items()}

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

        self.mx, self.nn = mx, nn
        self.config = ModelConfig.from_mapping(config)
        self.name = "mlx-" + device
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
        self.load(weights)
        self.grad = mx.value_and_grad(self.loss)

    def load(self, weights):
        self.p = {key: self.mx.array(value) for key, value in weights.items()}
        self.mx.eval(self.p)

    def _norm(self, params, value, gain, bias):
        mean = self.mx.mean(value, axis=-1, keepdims=True)
        var = self.mx.mean(self.mx.square(value - mean), axis=-1, keepdims=True)
        return (value - mean) * self.mx.rsqrt(var + 1e-5) * params[gain] + params[bias]

    def loss(self, params, tokens, target):
        mx = self.mx
        batch_size, steps = tokens.shape
        config = self.config
        width, heads = config.width, config.heads
        head_dim = width // heads
        hidden = params["embedding"][tokens] + params["position"][:steps]
        for layer in range(config.layers):
            def name(suffix, layer=layer):
                return layer_param(config, layer, suffix)

            normalized = self._norm(params, hidden, name("norm1_g"), name("norm1_b"))
            query, key, value = [
                (normalized @ params[name(part)])
                .reshape(batch_size, steps, heads, head_dim)
                .transpose(0, 2, 1, 3)
                for part in ("q", "k", "v")
            ]
            mask = mx.triu(mx.full((steps, steps), -float("inf")), k=1)
            scores = mx.softmax(
                (query @ key.transpose(0, 1, 3, 2)) / head_dim**0.5 + mask, axis=-1
            )
            attended = (scores @ value).transpose(0, 2, 1, 3).reshape(batch_size, steps, width)
            hidden = hidden + attended @ params[name("out")]
            feed = self._norm(params, hidden, name("norm2_g"), name("norm2_b"))
            hidden = hidden + mx.maximum(feed @ params[name("ff1")], 0) @ params[name("ff2")]
        logits = self._norm(params, hidden, "norm3_g", "norm3_b") @ params["head"]
        return self.nn.losses.cross_entropy(logits, target, reduction="mean")

    def step(self, tokens, target, lr):
        loss, grad = self.grad(self.p, self.mx.array(tokens), self.mx.array(target))
        self.p = {key: value - lr * grad[key] for key, value in self.p.items()}
        self.mx.eval(loss, self.p)  # Include completed work, not just lazy dispatch, in timings.
        return loss.item()

    def evaluate(self, tokens, target):
        return self.loss(self.p, self.mx.array(tokens), self.mx.array(target)).item()

    def export(self):
        self.mx.eval(self.p)
        return {key: np.array(value) for key, value in self.p.items()}

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
