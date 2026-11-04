"""Qwen LoRA branch on the Neural Engine.

The frozen 4-bit base stays on the GPU. The trainable product
``(x @ A) @ B`` and its gradients run as ANEForge programs on device mask
``0x4``. A program that compiles anywhere else raises. Dropout must be 0.
"""

from __future__ import annotations

import threading

import mlx.core as mx
import numpy as np

from .ane_forge import ANE_MASK

_CACHE: dict[tuple, object] = {}
_LOCK = threading.Lock()
CALLS = {"forward": 0, "backward": 0}


def ane_stats() -> dict:
    return {
        "status": "ok",
        "qwen_lora_branch": True,
        "device_mask": hex(ANE_MASK),
        "forward_calls": CALLS["forward"],
        "backward_calls": CALLS["backward"],
        "programs": len(_CACHE),
    }


def enable_ane_lora() -> None:
    """Send every LoRA branch through the Neural Engine. The base linear stays on the GPU."""

    from mlx_lm.tuner.lora import LoRALinear

    if getattr(LoRALinear, "_macreduce_ane", False):
        return

    def _call(self, x):
        keep = float(getattr(self.dropout, "_p_1", 0.0))
        if keep != 1.0:
            raise RuntimeError("the Neural Engine LoRA path requires dropout 0")
        y = self.linear(x)
        z = lora_product(x, self.lora_a, self.lora_b)
        return y + (self.scale * z).astype(x.dtype)

    LoRALinear.__call__ = _call
    LoRALinear._macreduce_ane = True
    _check_mask()


@mx.custom_function
def lora_product(x, left, right):
    """``(x @ left) @ right`` on the Neural Engine."""

    return _forward(x, left, right)


@lora_product.vjp
def _lora_vjp(primals, cotangent, output):
    del output
    x, left, right = primals
    dx, d_left, d_right = _backward(x, left, right, cotangent)
    return dx, d_left, d_right


def _forward(x, left, right):
    flat, shape = _flatten(x)
    rows, in_dim = flat.shape
    rank = left.shape[1]
    out_dim = right.shape[1]
    program = _program("forward", rows, in_dim, rank, out_dim)
    result = program(_numpy(flat), _numpy(left), _numpy(right))
    CALLS["forward"] += 1
    return mx.array(result).astype(x.dtype).reshape(*shape[:-1], out_dim)


def _backward(x, left, right, cotangent):
    flat, shape = _flatten(x)
    rows, in_dim = flat.shape
    rank = left.shape[1]
    out_dim = right.shape[1]
    upstream = _flatten(cotangent)[0]
    program = _program("backward", rows, in_dim, rank, out_dim)
    values = program(_numpy(flat), _numpy(left), _numpy(right), _numpy(upstream))
    dx, d_left, d_right = [values[name] for _, name in program.output_ports]
    CALLS["backward"] += 1
    return (
        mx.array(dx).astype(x.dtype).reshape(shape),
        mx.array(d_left).astype(left.dtype),
        mx.array(d_right).astype(right.dtype),
    )


def _program(kind: str, rows: int, in_dim: int, rank: int, out_dim: int):
    key = (kind, rows, in_dim, rank, out_dim)
    with _LOCK:
        cached = _CACHE.get(key)
        if cached is not None:
            return cached
        import aneforge as af
        from aneforge._compile import compile_multi

        features = af.input((rows, in_dim))
        left = af.input((in_dim, rank))
        right = af.input((rank, out_dim))
        if kind == "forward":
            program = af.compile((features @ left) @ right)
        elif kind == "backward":
            upstream = af.input((rows, out_dim))
            hidden = features @ left
            d_hidden = upstream @ right.transpose((1, 0))
            d_left = features.transpose((1, 0)) @ d_hidden
            d_right = hidden.transpose((1, 0)) @ upstream
            d_features = d_hidden @ left.transpose((1, 0))
            program = compile_multi([d_features, d_left, d_right])
        else:
            raise ValueError(f"unknown LoRA program {kind}")
        mask = _device_mask(program)
        if mask != ANE_MASK:
            program.release()
            raise RuntimeError(f"{kind} LoRA compiled off the Neural Engine: {hex(mask)}")
        _CACHE[key] = program
        return program


def _check_mask() -> None:
    """Compile one tiny program so a bad device mask fails before the long run."""

    features = mx.zeros((2, 4))
    left = mx.zeros((4, 2))
    right = mx.zeros((2, 4))
    mx.eval(_forward(features, left, right))


def _device_mask(program) -> int:
    compiled = getattr(program, "_prog", None)
    if compiled is None:
        compiled = program.prog
    return int(compiled._device_mask)


def _flatten(array):
    mx.eval(array)
    if array.ndim == 1:
        raise ValueError("a LoRA input needs a feature dimension")
    in_dim = int(array.shape[-1])
    rows = int(np.prod(array.shape[:-1]))
    return mx.reshape(array, (rows, in_dim)), tuple(array.shape)


def _numpy(array) -> np.ndarray:
    mx.eval(array)
    return np.array(array, copy=True).astype(np.float32)
