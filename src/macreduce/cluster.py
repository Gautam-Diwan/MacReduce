"""Fixed-cohort local SGD over HTTP, with versioned, idempotent model updates."""

import hashlib
import hmac
import json
import os
import re
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .artifacts import (
    CONTROL_RESPONSE_BYTES,
    MAX_CHUNK_SIZE,
    IncomingArtifact,
    build_manifest,
    plan_sha256,
    validate_manifest,
)
from .model import (
    ModelConfig,
    backend,
    batch,
    corpus,
    decode_weights,
    encode_weights,
    initial_weights,
    weighted_average,
)
from .observe import Events


class Conflict(ValueError):
    pass


def atomic_write(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class Coordinator:
    def __init__(
        self,
        directory,
        nodes,
        rounds=8,
        steps=10,
        lr=0.05,
        evaluator="torch-cpu",
        round_timeout=120,
        resume=False,
        validation_text=None,
        model_config=None,
    ):
        if (
            not nodes
            or "coordinator" in nodes
            or len(set(nodes)) != len(nodes)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]{1,40}", n) for n in nodes)
        ):
            raise ValueError("worker names must be unique, 1–40 letters, digits, _ or -")
        if rounds < 1 or steps < 1 or lr <= 0 or round_timeout <= 0:
            raise ValueError("rounds, steps, learning rate, and timeout must be positive")
        self.root = Path(directory)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.pending = {}
        self.uploads = {}
        self.uploads_by_id = {}
        self.chunk_size = MAX_CHUNK_SIZE
        self.failure = None
        self.round_timeout = round_timeout
        self.started = None  # First registration starts the admission deadline.
        self.model = ModelConfig.from_mapping(model_config)
        self.config = {
            "model": self.model.to_dict(),
            "rounds": rounds,
            "steps": steps,
            "lr": lr,
            "nodes": nodes,
            "optimizer": "SGD without momentum",
            "algorithm": "token-weighted local SGD",
            "seed": 7,
        }
        data, digest = corpus(seed=999, path=validation_text)
        self.config["validation_sha256"] = digest
        self.validation = batch(data, np.random.default_rng(1234), 32, self.model.context)
        manifest = self.root / "state.json"
        self.weights = initial_weights(self.model)
        self.version, self.registered, self.receipts = 0, {}, {}
        self.run_id = str(uuid.uuid4())
        if manifest.exists():
            if not resume:
                raise ValueError("run directory already has state; use --resume or a new directory")
            state = json.loads(manifest.read_text())
            if state.get("schema") != 2:
                raise ValueError("unsupported run schema")
            if self.config != state["config"]:
                raise ValueError("resume configuration does not match saved run")
            raw = (self.root / state["checkpoint"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != state["checkpoint_sha256"]:
                raise ValueError("checkpoint checksum mismatch")
            self.weights = decode_weights(raw, config=self.model)
            self.version, self.run_id = state["version"], state["run_id"]
            self.registered, self.receipts = state["registered"], state["receipts"]
        self.events = Events(self.root / "events", "coordinator", self.run_id)
        with self.events.span("evaluator_init"):
            self.evaluator = backend(evaluator, self.model, self.weights)
        self.validation_loss = self.evaluator.evaluate(*self.validation)
        self.raw = encode_weights(self.weights, self.model)
        self.save()
        self.events.emit(
            "run_started",
            config=self.config,
            resumed=resume,
            version=self.version,
            validation_loss=self.validation_loss,
            parameters=sum(v.size for v in self.weights.values()),
            checkpoint_bytes=len(self.raw),
            evaluator_backend=evaluator,
        )

    def save(self):
        filename = f"checkpoint-{self.version:05}.npz"
        atomic_write(self.root / filename, self.raw)
        state = {
            "schema": 2,
            "run_id": self.run_id,
            "version": self.version,
            "config": self.config,
            "checkpoint": filename,
            "checkpoint_sha256": hashlib.sha256(self.raw).hexdigest(),
            "registered": self.registered,
            "receipts": self.receipts,
            "validation_loss": self.validation_loss,
        }
        atomic_write(self.root / "state.json", json.dumps(state, indent=2).encode())

    def status(self):
        with self.lock:
            if (
                self.version < self.config["rounds"]
                and self.started is not None
                and time.monotonic() - self.started > self.round_timeout
                and not self.failure
            ):
                self.failure = "round deadline exceeded; fixed cohort is not silently reduced"
                self.events.emit(
                    "run_failed",
                    reason=self.failure,
                    version=self.version,
                    missing=[n for n in self.config["nodes"] if n not in self.pending],
                )
            state = (
                "failed"
                if self.failure
                else ("done" if self.version >= self.config["rounds"] else "running")
            )
            return {
                "state": state,
                "run_id": self.run_id,
                "version": self.version,
                "registered": list(self.registered),
                "pending": list(self.pending),
                "nodes": self.config["nodes"],
                "failure": self.failure,
                "validation_loss": self.validation_loss,
            }

    def register(self, node, info):
        with self.lock:
            if node not in self.config["nodes"]:
                raise ValueError("worker is not in the configured cohort")
            if type(info.get("batch_size")) is not int or not 1 <= info["batch_size"] <= 64:
                raise ValueError("batch_size must be an integer between 1 and 64")
            if info.get("backend") not in ("mlx-gpu", "mlx-cpu", "torch-cpu"):
                raise ValueError("unsupported backend")
            if node in self.registered:
                previous = self.registered[node]
                for key in ("batch_size", "backend", "data_sha256"):
                    if previous.get(key) != info.get(key):
                        raise Conflict(f"worker {key} cannot change during a run")
            else:
                self.registered[node] = info
                self.save()
            if self.started is None:
                self.started = time.monotonic()
            self.events.emit("worker_registered", worker=node, info=info)
            return {
                "run_id": self.run_id,
                "config": self.config,
                "rank": self.config["nodes"].index(node),
            }

    def job(self, node):
        with self.lock:
            if node not in self.registered:
                raise Conflict("register before requesting a job")
            status = self.status()
            if status["state"] != "running":
                return status
            if set(self.registered) != set(self.config["nodes"]) or node in self.pending:
                return {**status, "state": "wait"}
            return {
                **status,
                "state": "train",
                "config": self.config,
                "checkpoint_sha256": hashlib.sha256(self.raw).hexdigest(),
            }

    def checkpoint(self, version):
        with self.lock:
            if version != self.version:
                raise Conflict("checkpoint version has changed")
            return self.raw

    def submit(self, node, version, tokens, raw, run_id):
        with self.lock:
            if run_id != self.run_id:
                raise Conflict("update belongs to another run")
            if self.failure:
                raise Conflict(self.failure)
            digest = hashlib.sha256(raw).hexdigest()
            key = f"{version}:{node}"
            receipt = {"sha256": digest, "tokens": tokens}
            if key in self.receipts:
                if self.receipts[key] != receipt:
                    raise Conflict("conflicting retry for an already committed update")
                return {"accepted": True, "duplicate": True, "version": self.version}
            if self.status()["state"] != "running":
                raise Conflict("run is not accepting updates")
            if node not in self.registered or version != self.version:
                raise Conflict("unregistered worker or stale/future update")
            expected = (
                self.registered[node]["batch_size"] * self.config["steps"] * self.model.context
            )
            if tokens != expected:
                raise ValueError("processed token count differs from assigned work")
            if node in self.pending:
                if self.pending[node][2] != receipt:
                    raise Conflict("conflicting retry for pending update")
                return {"accepted": True, "duplicate": True, "version": self.version}
            with self.events.span("decode_update", worker=node, version=version):
                weights = decode_weights(raw, config=self.model)
            self.pending[node] = (weights, tokens, receipt)
            self.events.emit(
                "update_received",
                worker=node,
                version=version,
                payload_bytes=len(raw),
                tokens=tokens,
            )
            if set(self.pending) == set(self.config["nodes"]):
                try:
                    with self.events.span("reduce", version=version):
                        merged = weighted_average(
                            [self.pending[n][:2] for n in self.config["nodes"]]
                        )
                    with self.events.span("validation", version=version + 1):
                        self.evaluator.load(merged)
                        loss = self.evaluator.evaluate(*self.validation)
                        if not np.isfinite(loss):
                            raise ValueError("non-finite validation loss")
                    self.weights, self.raw = merged, encode_weights(merged, self.model)
                    self.version += 1
                    self.validation_loss = loss
                    for n, (_, _, r) in self.pending.items():
                        self.receipts[f"{version}:{n}"] = r
                    with self.events.span("checkpoint_commit", version=self.version):
                        self.save()
                    self.events.emit(
                        "round_committed",
                        version=self.version,
                        tokens=sum(v[1] for v in self.pending.values()),
                        validation_loss=loss,
                        round_seconds=time.monotonic() - self.started,
                    )
                    self.pending.clear()
                    self.started = time.monotonic()
                except Exception as exc:
                    self.failure = f"round commit failed: {type(exc).__name__}: {exc}"
                    self.events.emit("run_failed", reason=self.failure)
                    raise
            return {"accepted": True, "duplicate": False, "version": self.version}

    def checkpoint_manifest(self, version, run_id):
        with self.lock:
            if run_id != self.run_id:
                raise Conflict("checkpoint belongs to another run")
            if version != self.version:
                raise Conflict("checkpoint version has changed")
            return build_manifest(
                self.raw,
                run_id=self.run_id,
                version=self.version,
                role="checkpoint",
                parameter_names=list(self.weights),
                plan=plan_sha256(self.model),
                chunk_size=self.chunk_size,
                dtype=self.model.dtype,
            )

    def checkpoint_chunk(self, version, index, run_id):
        with self.lock:
            manifest = self.checkpoint_manifest(version, run_id)
            start = index * manifest["chunk_size"]
            if type(index) is not int or index < 0 or start >= manifest["length"]:
                raise ValueError("chunk index out of range")
            return self.raw[start : start + manifest["chunk_size"]]

    def _check_update_request(self, node, version, tokens, run_id, manifest):
        validate_manifest(manifest)
        if run_id != self.run_id or manifest["run_id"] != self.run_id:
            raise Conflict("update belongs to another run")
        if manifest["role"] != "update" or manifest["version"] != version:
            raise ValueError("update manifest does not match the request")
        if manifest.get("node") != node:
            raise Conflict("manifest node does not match")
        if manifest["dtype"] != self.model.dtype or manifest["plan_sha256"] != plan_sha256(
            self.model
        ):
            raise ValueError("update manifest does not match the run")
        if manifest["parameter_schema"] != list(self.weights):
            raise ValueError("parameter schema does not match the run")
        if node not in self.registered:
            raise Conflict("unregistered worker or stale/future update")
        expected = self.registered[node]["batch_size"] * self.config["steps"] * self.model.context
        if tokens != expected:
            raise ValueError("processed token count differs from assigned work")
        if self.failure:
            raise Conflict(self.failure)
        return {"sha256": manifest["sha256"], "tokens": tokens}

    def begin_update(self, node, version, tokens, run_id, manifest):
        with self.lock:
            receipt = self._check_update_request(node, version, tokens, run_id, manifest)
            key = f"{version}:{node}"
            if key in self.receipts:
                if self.receipts[key] != receipt:
                    raise Conflict("conflicting retry for an already committed update")
                return {
                    "accepted": True,
                    "duplicate": True,
                    "version": self.version,
                    "upload_id": None,
                }
            if version == self.version and node in self.pending:
                if self.pending[node][2] != receipt:
                    raise Conflict("conflicting retry for pending update")
                return {
                    "accepted": True,
                    "duplicate": True,
                    "version": self.version,
                    "upload_id": None,
                }
            if self.status()["state"] != "running" or version != self.version:
                raise Conflict("run is not accepting updates")
            existing = self.uploads.get(key)
            if existing is not None:
                if existing.manifest["sha256"] != manifest["sha256"]:
                    raise Conflict("conflicting upload for the same update")
                return {
                    "accepted": True,
                    "duplicate": existing.completed,
                    "version": self.version,
                    "upload_id": existing.upload_id,
                }
            upload_id = uuid.uuid4().hex
            artifact = IncomingArtifact(self.root / "incoming", upload_id, manifest)
            artifact.tokens = tokens
            self.uploads[key] = artifact
            self.uploads_by_id[upload_id] = artifact
            return {
                "accepted": True,
                "duplicate": False,
                "version": self.version,
                "upload_id": upload_id,
            }

    def receive_chunk(self, upload_id, index, data):
        with self.lock:
            artifact = self.uploads_by_id.get(upload_id)
            if artifact is None:
                raise ValueError("unknown upload")
            artifact.write_chunk(index, data)
            return {"accepted": True, "duplicate": artifact.completed}

    def complete_update(self, upload_id):
        with self.lock:
            artifact = self.uploads_by_id.get(upload_id)
            if artifact is None:
                raise ValueError("unknown upload")
            if artifact.completed and hasattr(artifact, "result"):
                return {**artifact.result, "duplicate": True}
            raw = artifact.finish()
            result = self.submit(
                artifact.manifest["node"],
                artifact.manifest["version"],
                artifact.tokens,
                raw,
                artifact.manifest["run_id"],
            )
            artifact.result = result
            return result


def make_server(coordinator, host, port, token):
    if len(token) < 24:
        raise ValueError("MACREDUCE_TOKEN must be at least 24 characters")

    class Handler(BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(30)

        def log_message(self, *args):
            pass

        def respond(self, value, status=200, content_type="application/json"):
            data = json.dumps(value).encode() if content_type == "application/json" else value
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def dispatch(self, method):
            if not hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token):
                self.respond({"error": "unauthorized"}, 401)
                return
            parsed = urlsplit(self.path)
            q = parse_qs(parsed.query)
            node = q.get("node", [""])[0]
            try:
                if method == "GET":
                    if parsed.path == "/status":
                        return self.respond(coordinator.status())
                    if parsed.path == "/job":
                        return self.respond(coordinator.job(node))
                    if parsed.path == "/checkpoint":
                        return self.respond(
                            coordinator.checkpoint(int(q["version"][0])),
                            content_type="application/octet-stream",
                        )
                    if parsed.path == "/artifact/manifest":
                        if q.get("kind", [""])[0] != "checkpoint":
                            raise ValueError("unsupported artifact manifest")
                        return self.respond(
                            coordinator.checkpoint_manifest(int(q["version"][0]), q["run_id"][0])
                        )
                    if parsed.path == "/artifact/chunk":
                        if q.get("kind", [""])[0] != "checkpoint":
                            raise ValueError("unsupported artifact chunk")
                        return self.respond(
                            coordinator.checkpoint_chunk(
                                int(q["version"][0]), int(q["index"][0]), q["run_id"][0]
                            ),
                            content_type="application/octet-stream",
                        )
                    if parsed.path == "/metrics":
                        s = coordinator.status()
                        text = (
                            f"# TYPE macreduce_version gauge\nmacreduce_version {s['version']}\n"
                            f"# TYPE macreduce_pending_workers gauge\n"
                            f"macreduce_pending_workers {len(s['pending'])}\n"
                            f"# TYPE macreduce_validation_loss gauge\n"
                            f"macreduce_validation_loss {s['validation_loss']}\n"
                            f"# TYPE macreduce_failed gauge\n"
                            f"macreduce_failed {int(s['state'] == 'failed')}\n"
                        )
                        return self.respond(text.encode(), content_type="text/plain; version=0.0.4")
                elif method == "POST":
                    length = int(self.headers.get("Content-Length", "-1"))
                    if parsed.path == "/artifact/complete" and length in (-1, 0):
                        raw = b""
                    else:
                        if parsed.path == "/artifact/chunk":
                            limit = MAX_CHUNK_SIZE
                        elif parsed.path == "/artifact/update":
                            limit = CONTROL_RESPONSE_BYTES
                        elif parsed.path == "/update":
                            limit = len(coordinator.raw) + 65536
                        else:
                            limit = 65536
                        if not 0 <= length <= limit:
                            return self.respond({"error": "invalid payload size"}, 413)
                        raw = self.rfile.read(length)
                        if len(raw) != length:
                            raise ValueError("truncated request")
                    if parsed.path == "/register":
                        return self.respond(coordinator.register(node, json.loads(raw)))
                    if parsed.path == "/update":
                        return self.respond(
                            coordinator.submit(
                                node, int(q["version"][0]), int(q["tokens"][0]), raw, q["run_id"][0]
                            )
                        )
                    if parsed.path == "/artifact/update":
                        return self.respond(
                            coordinator.begin_update(
                                node,
                                int(q["version"][0]),
                                int(q["tokens"][0]),
                                q["run_id"][0],
                                json.loads(raw),
                            )
                        )
                    if parsed.path == "/artifact/chunk":
                        return self.respond(
                            coordinator.receive_chunk(q["upload_id"][0], int(q["index"][0]), raw)
                        )
                    if parsed.path == "/artifact/complete":
                        return self.respond(coordinator.complete_update(q["upload_id"][0]))
                self.respond({"error": "unknown endpoint"}, 404)
            except Conflict as exc:
                self.respond({"error": str(exc)}, 409)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond({"error": str(exc)}, 400)
            except (BrokenPipeError, ConnectionResetError, socket.timeout):
                pass
            except Exception as exc:
                coordinator.events.emit("server_error", error=type(exc).__name__, detail=str(exc))
                self.respond({"error": "internal error; see coordinator events"}, 500)

        def do_GET(self):
            self.dispatch("GET")

        def do_POST(self):
            self.dispatch("POST")

    return ThreadingHTTPServer((host, port), Handler)
