"""
Quantized adapter transport.

Master weights stay float32; only the gRPC payload is compressed. Affine int8
matches MLX ``mx.quantize`` grouping along the long axis (group 64). Error
feedback carries the leftover into the next delta.
"""

from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np

from macreduce.cluster.artifacts import MAX_ARTIFACT_BYTES

CODECS = ("fp32", "bf16", "int8")
GROUP = 64
BITS = 8


def layout(schema: dict, codec: str) -> SimpleNamespace:
    """Names, shapes and dtypes of the tensors Flower should send."""
    if codec not in CODECS:
        msg = f"wire codec must be one of {CODECS}"
        raise ValueError(msg)
    names, shapes, dtypes = [], {}, {}
    for name, shape in schema.items():
        if codec == "fp32":
            names.append(name)
            shapes[name], dtypes[name] = list(shape), np.dtype(np.float32)
            continue
        if codec == "bf16":
            key = f"{name}::bf16"
            names.append(key)
            shapes[key], dtypes[key] = list(shape), np.dtype(np.uint16)
            continue
        work, _, _ = _aligned(np.zeros(shape, np.float32))
        packed = f"{name}::q"
        scale = f"{name}::scale"
        zero = f"{name}::zero"
        groups = int(np.prod(work.shape) // GROUP)
        names.extend((packed, scale, zero))
        shapes[packed], dtypes[packed] = list(work.shape), np.dtype(np.uint8)
        shapes[scale], dtypes[scale] = [groups], np.dtype(np.float32)
        shapes[zero], dtypes[zero] = [groups], np.dtype(np.float32)
    return SimpleNamespace(names=names, shapes=shapes, dtypes=dtypes)


def encode(tensors: dict, codec: str) -> dict:
    if codec == "fp32":
        return {name: np.asarray(value, dtype=np.float32) for name, value in tensors.items()}
    if codec == "bf16":
        return {f"{name}::bf16": _to_bf16(value) for name, value in tensors.items()}
    encoded = {}
    for name, value in tensors.items():
        work, _, _ = _aligned(np.asarray(value, dtype=np.float32))
        grouped = work.reshape(-1, GROUP)
        low = grouped.min(axis=1, keepdims=True)
        high = grouped.max(axis=1, keepdims=True)
        scale = np.where(high > low, (high - low) / (2**BITS - 1), 1.0).astype(np.float32)
        quantized = np.round((grouped - low) / scale).clip(0, 2**BITS - 1).astype(np.uint8)
        encoded[f"{name}::q"] = quantized.reshape(work.shape)
        encoded[f"{name}::scale"] = scale.reshape(-1)
        encoded[f"{name}::zero"] = low.astype(np.float32).reshape(-1)
    return encoded


def decode(encoded: dict, codec: str, schema: dict) -> dict:
    if codec == "fp32":
        return {name: np.asarray(encoded[name], dtype=np.float32) for name in schema}
    if codec == "bf16":
        return {name: _from_bf16(encoded[f"{name}::bf16"]).reshape(schema[name]) for name in schema}
    result = {}
    for name, shape in schema.items():
        work, transposed, original = _aligned(np.zeros(shape, np.float32))
        grouped = encoded[f"{name}::q"].reshape(-1, GROUP)
        scale = encoded[f"{name}::scale"].reshape(-1, 1)
        zero = encoded[f"{name}::zero"].reshape(-1, 1)
        restored = (grouped.astype(np.float32) * scale + zero).reshape(work.shape)
        last = original.shape[0] if original.ndim == 1 or transposed else original.shape[-1]
        restored = restored[:, :last]
        if original.ndim == 1:
            restored = restored.reshape(-1)
        elif transposed:
            restored = restored.T
        result[name] = np.asarray(restored, dtype=np.float32).reshape(shape)
    return result


def quantize_delta(delta: dict, residual: dict | None, codec: str) -> tuple[dict, dict]:
    """Encode ``delta + residual`` and return the leftover for the next round."""
    schema = {name: list(np.asarray(value).shape) for name, value in delta.items()}
    if residual is None:
        residual = {name: np.zeros(shape, np.float32) for name, shape in schema.items()}
    target = {name: np.asarray(delta[name], dtype=np.float32) + residual[name] for name in schema}
    encoded = encode(target, codec)
    reconstructed = decode(encoded, codec, schema)
    leftover = {name: np.asarray(target[name] - reconstructed[name], dtype=np.float32)
                for name in schema}
    return encoded, leftover


def pack_encoded(encoded: dict) -> bytes:
    out = io.BytesIO()
    np.savez(out, **encoded)
    raw = out.getvalue()
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("adapter exceeds the 64 MiB transfer cap")
    return raw


def unpack_encoded(raw: bytes, names: list[str]) -> dict:
    if not 1 <= len(raw) <= MAX_ARTIFACT_BYTES:
        raise ValueError("invalid adapter size")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        encoded = {name: archive[name] for name in archive.files}
    if set(encoded) != set(names):
        raise ValueError("encoded adapter keys differ from the wire layout")
    return encoded


def _aligned(array: np.ndarray) -> tuple[np.ndarray, bool, np.ndarray]:
    original = np.asarray(array, dtype=np.float32)
    if original.ndim == 1:
        work, transposed = original.reshape(1, -1), False
    elif original.shape[-1] < original.shape[0]:
        work, transposed = original.T, True
    else:
        work, transposed = original, False
    pad = (-work.shape[-1]) % GROUP
    if pad:
        work = np.pad(work, ((0, 0), (0, pad)))
    return work, transposed, original


def _to_bf16(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(
        0xFFFF0000
    )
    return (rounded >> np.uint32(16)).astype(np.uint16)


def _from_bf16(value: np.ndarray) -> np.ndarray:
    bits = np.asarray(value, dtype=np.uint16).astype(np.uint32) << np.uint32(16)
    return bits.view(np.float32)
