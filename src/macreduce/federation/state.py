"""Bounded float32 adapter files and atomic coordinator commits."""

from __future__ import annotations

import hashlib
import io
import json
import uuid
import zipfile
from pathlib import Path

import numpy as np

from macreduce.cluster.artifacts import MAX_ARTIFACT_BYTES
from macreduce.cluster.coordinator import atomic_write

from .plan import fingerprint


def pack_adapter(state: dict) -> bytes:
    arrays = {}
    for name, value in state.items():
        array = np.asarray(value, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError("adapter contains nonfinite values")
        arrays[name] = array
    if not arrays:
        raise ValueError("adapter is empty")
    out = io.BytesIO()
    np.savez(out, **arrays)
    raw = out.getvalue()
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("adapter exceeds the 64 MiB transfer cap")
    return raw


def unpack_adapter(raw: bytes, schema: dict | None = None) -> dict:
    if not 1 <= len(raw) <= MAX_ARTIFACT_BYTES:
        raise ValueError("invalid adapter size")
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        files = archive.infolist()
        if not files or len(files) > 1024 or sum(f.file_size for f in files) > MAX_ARTIFACT_BYTES:
            raise ValueError("invalid or oversized decompressed adapter")
        if len({f.filename for f in files}) != len(files):
            raise ValueError("duplicate adapter tensor names")
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    if schema is not None and set(result) != set(schema):
        raise ValueError("adapter keys differ from the common adapter")
    for name, array in result.items():
        if array.dtype != np.float32 or not np.isfinite(array).all():
            raise ValueError("wire tensors must be finite float32")
        if schema is not None and list(array.shape) != schema[name]:
            raise ValueError("adapter shape differs from the common adapter")
    return result


def tensor_digest(names, arrays):
    digest = hashlib.sha256()
    for name, array in zip(names, arrays, strict=True):
        tensor = np.asarray(array)
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode())
        digest.update(tensor.tobytes(order="C"))
    return digest.hexdigest()


class RunState:
    def __init__(self, root: Path, plan: dict, data_hash: str, initial=None, *, resume=False):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.plan = plan
        self.plan_hash = fingerprint(plan)
        manifest = self.root / "state.json"
        if manifest.exists():
            if not resume:
                raise ValueError("run exists; use --resume or a new directory")
            self.meta = json.loads(manifest.read_text())
            if self.meta["plan_hash"] != self.plan_hash or self.meta["data_hash"] != data_hash:
                raise ValueError("resume plan/dataset differs")
            raw = (self.root / self.meta["file"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.meta["file_sha256"]:
                raise ValueError("checkpoint checksum mismatch")
            self.adapter = unpack_adapter(raw, self.meta["schema"])
        else:
            if resume or initial is None:
                raise ValueError("new run needs an adapter; resume needs a saved checkpoint")
            self.adapter = unpack_adapter(pack_adapter(initial))
            self.meta = {"run_id": str(uuid.uuid4()), "plan_hash": self.plan_hash,
                         "data_hash": data_hash, "version": 0,
                         "names": list(self.adapter),
                         "schema": {k: list(v.shape) for k, v in self.adapter.items()},
                         "plan": plan, "receipts": {}}
            self._publish(self.adapter, self.meta)

    @property
    def version(self):
        return self.meta["version"]

    @property
    def arrays(self):
        return [self.adapter[key] for key in self.meta["names"]]

    @property
    def digest(self):
        return tensor_digest(self.meta["names"], self.arrays)

    def _publish(self, adapter, meta):
        raw = pack_adapter(adapter)
        filename = f"adapter-{meta['version']:06d}.npz"
        candidate = {**meta, "file": filename, "file_sha256": hashlib.sha256(raw).hexdigest()}
        atomic_write(self.root / filename, raw)
        atomic_write(self.root / "state.json", json.dumps(candidate, allow_nan=False).encode())
        self.meta, self.adapter = candidate, adapter

    def commit(self, updates):
        names = {row["id"] for row in self.plan["clients"]}
        if set(updates) != names:
            raise ValueError("fixed cohort is incomplete")
        merged = {k: v.astype(np.float64) for k, v in self.adapter.items()}
        receipts = {}
        for row in self.plan["clients"]:
            delta, receipt = updates[row["id"]]
            # Validate shape, dtype, byte cap, and finiteness even for direct callers.
            delta = unpack_adapter(pack_adapter(delta), self.meta["schema"])
            for key in merged:
                merged[key] += row["weight"] * delta[key]
            receipts[row["id"]] = receipt
        merged = unpack_adapter(pack_adapter(merged), self.meta["schema"])
        candidate = {**self.meta, "version": self.version + 1,
                     "receipts": {**self.meta["receipts"], str(self.version): receipts}}
        self._publish(merged, candidate)
