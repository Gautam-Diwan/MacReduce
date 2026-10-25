"""Bounded chunked artifacts. Hash checks stay separate from the optimizer step."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

from .model import ModelConfig

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_CHUNK_SIZE = 256 * 1024
MAX_CHUNKS = 4096
CONTROL_RESPONSE_BYTES = 1024 * 1024


def plan_sha256(config):
    payload = json.dumps(
        ModelConfig.from_mapping(config).to_dict(), sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def chunk_length(manifest, index):
    start = index * manifest["chunk_size"]
    if type(index) is not int or index < 0 or start >= manifest["length"]:
        raise ValueError("chunk index out of range")
    return min(manifest["chunk_size"], manifest["length"] - start)


def validate_manifest(manifest):
    """Reject an oversized or inconsistent manifest before any artifact allocation."""
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if manifest.get("schema") != 1:
        raise ValueError("unsupported artifact schema")
    if manifest.get("role") not in ("checkpoint", "update"):
        raise ValueError("invalid artifact role")
    if manifest.get("dtype") != "float32":
        raise ValueError("artifact dtype must be float32")
    length, chunk_size = manifest.get("length"), manifest.get("chunk_size")
    hashes = manifest.get("chunk_hashes")
    if type(length) is not int or type(chunk_size) is not int:
        raise ValueError("invalid manifest lengths")
    if length > MAX_ARTIFACT_BYTES:
        raise ValueError("artifact exceeds size cap")
    if length < 1:
        raise ValueError("artifact is empty")
    if not 1 <= chunk_size <= MAX_CHUNK_SIZE:
        raise ValueError("invalid chunk size")
    if not isinstance(hashes, list) or not 1 <= len(hashes) <= MAX_CHUNKS:
        raise ValueError("invalid chunk hash list")
    if len(hashes) != (length + chunk_size - 1) // chunk_size:
        raise ValueError("chunk hash count does not match length")
    if any(type(item) is not str or len(item) != 64 for item in hashes):
        raise ValueError("invalid chunk hash")
    if type(manifest.get("sha256")) is not str or len(manifest["sha256"]) != 64:
        raise ValueError("invalid content hash")
    if type(manifest.get("run_id")) is not str or not manifest["run_id"]:
        raise ValueError("manifest run id required")
    if type(manifest.get("version")) is not int:
        raise ValueError("manifest version required")
    if type(manifest.get("plan_sha256")) is not str or len(manifest["plan_sha256"]) != 64:
        raise ValueError("plan hash required")
    schema = manifest.get("parameter_schema")
    if not isinstance(schema, list) or not schema or any(type(name) is not str for name in schema):
        raise ValueError("parameter schema required")
    return manifest


def build_manifest(
    raw,
    *,
    run_id,
    version,
    role,
    parameter_names,
    plan,
    node=None,
    chunk_size=MAX_CHUNK_SIZE,
    dtype="float32",
):
    if type(raw) is not bytes:
        raise ValueError("artifact must be bytes")
    chunks = [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)] if raw else []
    manifest = validate_manifest(
        {
            "schema": 1,
            "run_id": run_id,
            "version": version,
            "role": role,
            "node": node,
            "dtype": dtype,
            "parameter_schema": list(parameter_names),
            "plan_sha256": plan,
            "length": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "chunk_size": chunk_size,
            "chunk_hashes": [hashlib.sha256(part).hexdigest() for part in chunks],
        }
    )
    return manifest


def assemble_chunks(manifest, read_chunk):
    """Write chunks to a temp file and return bytes only after every hash matches."""
    validate_manifest(manifest)
    descriptor, name = tempfile.mkstemp(prefix="macreduce-", suffix=".partial")
    os.close(descriptor)
    final = name[: -len(".partial")] + ".complete"
    try:
        with open(name, "wb") as out:
            for index, expected in enumerate(manifest["chunk_hashes"]):
                chunk = read_chunk(index)
                if type(chunk) is not bytes or len(chunk) != chunk_length(manifest, index):
                    raise ValueError("corrupted chunk")
                if hashlib.sha256(chunk).hexdigest() != expected:
                    raise ValueError("corrupted chunk")
                out.write(chunk)
            out.flush()
            os.fsync(out.fileno())
        if os.path.getsize(name) != manifest["length"] or file_sha256(name) != manifest["sha256"]:
            raise ValueError("content hash mismatch")
        os.replace(name, final)
        return Path(final).read_bytes()
    finally:
        for candidate in (name, final):
            if os.path.exists(candidate):
                os.remove(candidate)


class IncomingArtifact:
    """Server-side upload. Completing it validates bytes and does not apply an update."""

    def __init__(self, directory, upload_id, manifest):
        validate_manifest(manifest)
        self.manifest = manifest
        self.upload_id = upload_id
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.partial = self.directory / f"{upload_id}.partial"
        self.final = self.directory / f"{upload_id}.npz"
        self.received = set()
        self.completed = False
        self.partial.write_bytes(b"")

    def write_chunk(self, index, data):
        if type(data) is not bytes:
            raise ValueError("corrupted chunk")
        expected_len = chunk_length(self.manifest, index)
        digest = hashlib.sha256(data).hexdigest()
        if digest != self.manifest["chunk_hashes"][index] or len(data) != expected_len:
            raise ValueError("corrupted chunk")
        if self.completed:
            return
        with self.partial.open("r+b") as handle:
            handle.seek(index * self.manifest["chunk_size"])
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        self.received.add(index)

    def finish(self):
        if self.completed and self.final.is_file():
            return self.final.read_bytes()
        count = len(self.manifest["chunk_hashes"])
        if (
            self.received != set(range(count))
            or self.partial.stat().st_size != self.manifest["length"]
        ):
            raise ValueError("incomplete artifact")
        digest = hashlib.sha256()
        with self.partial.open("rb") as handle:
            for index, expected in enumerate(self.manifest["chunk_hashes"]):
                data = handle.read(chunk_length(self.manifest, index))
                if hashlib.sha256(data).hexdigest() != expected:
                    self._reset()
                    raise ValueError("corrupted chunk")
                digest.update(data)
            if handle.read(1):
                self._reset()
                raise ValueError("artifact longer than manifest")
        if digest.hexdigest() != self.manifest["sha256"]:
            self._reset()
            raise ValueError("content hash mismatch")
        os.replace(self.partial, self.final)
        self.completed = True
        return self.final.read_bytes()

    def _reset(self):
        self.partial.unlink(missing_ok=True)
        self.partial.write_bytes(b"")
        self.received.clear()
