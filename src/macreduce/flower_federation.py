"""
Pinned Flower transport adapter with MacReduce authentication/recovery rules.

Flower owns RPC and concurrent dispatch. MacReduce owns logical identities,
fixed-cohort commits, finite float32 arrays and durable local trainer state.
"""

from __future__ import annotations

import hashlib
import hmac
import io
import json
import math
import secrets
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from flwr.client import Client
from flwr.common import (
    Code,
    FitIns,
    FitRes,
    GetPropertiesIns,
    GetPropertiesRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.compat.client.app import start_client_internal
from flwr.compat.server.grpc_bidi.grpc_server import start_grpc_server
from flwr.server import Server, SimpleClientManager
from flwr.server.strategy import FedAvg

from .artifacts import CONTROL_RESPONSE_BYTES, MAX_ARTIFACT_BYTES
from .cluster import atomic_write
from .federation_plan import client_spec, fingerprint
from .federation_state import RunState, pack_adapter, tensor_digest, unpack_adapter


def check_flower_version():
    if version("flwr") != "1.43.0":
        raise RuntimeError("transport requires flwr==1.43.0; synchronize the federation extra")


def signed(body, token):
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    mac = hmac.new(token.encode(), raw.encode(), hashlib.sha256).hexdigest()
    return json.dumps({"body": raw, "mac": mac})


def verified(raw, token):
    if not isinstance(raw, str) or len(raw.encode()) > CONTROL_RESPONSE_BYTES:
        raise ValueError("invalid signed envelope size")
    envelope = json.loads(raw)
    body = envelope["body"]
    expected = hmac.new(token.encode(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, envelope["mac"]):
        raise ValueError("authentication failed")
    return json.loads(body)


def decode_parameters(parameters, names, schema):
    """Bound/validate NPY headers before Flower/NumPy allocates array buffers."""
    if len(parameters.tensors) != len(names):
        raise ValueError("tensor count differs from the adapter")
    if sum(map(len, parameters.tensors)) > MAX_ARTIFACT_BYTES:
        raise ValueError("adapter exceeds the 64 MiB tensor cap")
    for key, raw in zip(names, parameters.tensors, strict=True):
        stream = io.BytesIO(raw)
        version = np.lib.format.read_magic(stream)
        reader = { (1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0 }.get(version)
        if reader is None:
            raise ValueError("unsupported NPY header")
        shape, fortran, dtype = reader(stream)
        if (dtype != np.float32 or list(shape) != schema[key] or fortran
                or math.prod(shape) * dtype.itemsize != len(raw) - stream.tell()):
            raise ValueError("invalid tensor dtype/shape/length")
    arrays = parameters_to_ndarrays(parameters)
    if not all(np.isfinite(value).all() for value in arrays):
        raise ValueError("adapter contains nonfinite values")
    return dict(zip(names, arrays, strict=True))


class PersistentClient(Client):
    def __init__(self, backend, plan, name, data_hash, root: Path, token):
        check_flower_version()
        self.backend, self.plan, self.name = backend, plan, name
        self.spec = client_spec(plan, name)
        self.data_hash, self.root, self.token = data_hash, Path(root), token
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / "worker.json"
        self.meta = json.loads(path.read_text()) if path.exists() else None
        if self.meta is not None:
            if (self.meta["plan_hash"] != fingerprint(plan) or self.meta["client"] != name
                    or self.meta["data_hash"] != data_hash):
                raise ValueError("worker checkpoint identity differs; use a new output directory")
            self.backend.restore(self.root / self.meta.get("generation", "initial"))

    def _request(self, config, op):
        body = verified(config["envelope"], self.token)
        if (body["op"] != op or body["plan_hash"] != fingerprint(self.plan)
                or body["data_hash"] != self.data_hash):
            raise ValueError("coordinator operation/plan/data mismatch")
        version = body["version"]
        if self.meta is None:
            if version != 0:
                raise ValueError("client optimizer checkpoint missing for resumed run")
            self.meta = {"run_id": body["run_id"], "client": self.name,
                         "plan_hash": body["plan_hash"], "data_hash": self.data_hash,
                         "prepared_version": None}
            self.backend.save(self.root / "initial")
            atomic_write(self.root / "worker.json", json.dumps(self.meta).encode())
        if self.meta["run_id"] != body["run_id"]:
            raise ValueError("coordinator run changed; use a new worker output directory")
        previous = self.meta["prepared_version"]
        if previous not in (version, version - 1) and not (previous is None and version == 0):
            raise ValueError("worker optimizer/sampler state does not match the global version")
        return body

    def get_properties(self, ins):
        body = self._request(ins.config, "hello")
        response = {**body, "op": "ready", "client": self.name,
                    "device": self.spec["device"], "backend_ready": True}
        return GetPropertiesRes(Status(Code.OK, "ready"), {"envelope": signed(response, self.token)})

    def fit(self, ins):
        request = self._request(ins.config, "fit")
        if request["client"] != self.name:
            raise ValueError("fit request is for a different client")
        names = list(self.backend.snapshot())
        schema = {k: list(v.shape) for k, v in self.backend.snapshot().items()}
        start = decode_parameters(ins.parameters, names, schema)
        digest = tensor_digest(names, [start[k] for k in names])
        if digest != request["start_hash"]:
            raise ValueError("broadcast checksum mismatch")
        version = request["version"]
        if self.meta["prepared_version"] == version:
            if self.meta["start_hash"] != digest:
                raise ValueError("conflicting replay of an already prepared round")
            directory = self.root / self.meta["generation"]
            raw = (directory / "delta.npz").read_bytes()
            if hashlib.sha256(raw).hexdigest() != self.meta["delta_file_hash"]:
                raise ValueError("pending update checksum mismatch")
            delta = unpack_adapter(raw, schema)
            metrics = self.meta["metrics"]
        else:
            baseline = self.root / self.meta.get("generation", "initial")
            try:
                after, metrics = self.backend.train_round(start)
                delta = {key: np.asarray(after[key] - start[key], dtype=np.float32) for key in names}
                raw = pack_adapter(delta)
                directory = self.root / f"round-{version:06d}"
                self.backend.save(directory)
                atomic_write(directory / "delta.npz", raw)
                candidate = {**self.meta, "prepared_version": version, "start_hash": digest,
                             "generation": directory.name, "metrics": metrics,
                             "delta_file_hash": hashlib.sha256(raw).hexdigest()}
                atomic_write(self.root / "worker.json", json.dumps(candidate, allow_nan=False).encode())
                self.meta = candidate
            except Exception:
                self.backend.restore(baseline)
                raise
        arrays = [delta[key] for key in names]
        response = {**request, "op": "update", "client": self.name,
                    "delta_hash": tensor_digest(names, arrays), "metrics": metrics}
        examples = self.spec["microbatch"] * self.spec["accumulation"] * self.plan["local_steps"]
        return FitRes(Status(Code.OK, "prepared"), ndarrays_to_parameters(arrays), examples,
                      {"envelope": signed(response, self.token)})


class FixedCohort(FedAvg):
    def __init__(self, state: RunState, token):
        check_flower_version()
        self.state, self.token = state, token
        self.expected = {row["id"] for row in state.plan["clients"]}
        self.assignments = {}
        super().__init__(fraction_evaluate=0, min_fit_clients=len(self.expected),
                         min_available_clients=len(self.expected), accept_failures=False,
                         initial_parameters=ndarrays_to_parameters(state.arrays))

    def _body(self, op):
        return {"op": op, "run_id": self.state.meta["run_id"], "version": self.state.version,
                "plan_hash": self.state.plan_hash, "data_hash": self.state.meta["data_hash"],
                "nonce": secrets.token_hex(16)}

    def configure_fit(self, server_round, parameters, client_manager):
        timeout = self.state.plan["round_timeout"]
        if not client_manager.wait_for(len(self.expected), timeout=timeout):
            raise RuntimeError("fixed cohort did not connect; no round was committed")
        proxies = list(client_manager.all().values())
        if len(proxies) != len(self.expected):
            raise RuntimeError("unexpected additional clients; refusing ambiguous cohort")
        challenge = self._body("hello")
        by_name = {}
        for proxy in proxies:
            reply = proxy.get_properties(GetPropertiesIns({"envelope": signed(challenge, self.token)}),
                                         timeout=timeout, group_id=server_round)
            body = verified(reply.properties["envelope"], self.token)
            name = body["client"]
            spec = client_spec(self.state.plan, name)
            if (reply.status.code != Code.OK or body != {**challenge, "op": "ready",
                    "client": name, "device": spec["device"], "backend_ready": True}
                    or name in by_name):
                raise ValueError("invalid/duplicate client readiness identity")
            by_name[name] = proxy
        if set(by_name) != self.expected:
            raise RuntimeError("fixed cohort is incomplete")
        jobs = []
        self.assignments = {}
        for name, proxy in by_name.items():
            body = {**self._body("fit"), "client": name, "start_hash": self.state.digest}
            self.assignments[proxy.cid] = body
            jobs.append((proxy, FitIns(parameters, {"envelope": signed(body, self.token)})))
        return jobs

    def aggregate_fit(self, _server_round, results, failures):
        if failures or len(results) != len(self.expected):
            raise RuntimeError("round failed/incomplete; stop and resume the uncommitted version")
        updates = {}
        for proxy, result in results:
            request = self.assignments[proxy.cid]
            response = verified(result.metrics["envelope"], self.token)
            name = request["client"]
            base = {k: response[k] for k in request}
            if result.status.code != Code.OK or base != {**request, "op": "update"}:
                raise ValueError("update identity/authentication mismatch")
            delta = decode_parameters(result.parameters, self.state.meta["names"],
                                      self.state.meta["schema"])
            digest = tensor_digest(self.state.meta["names"], list(delta.values()))
            if response["delta_hash"] != digest or name in updates:
                raise ValueError("conflicting or duplicate client update")
            tokens = response["metrics"]["tokens"]
            if type(tokens) is not int or tokens < 1:
                raise ValueError("invalid supervised-token count")
            updates[name] = (delta, {"delta_hash": digest, "metrics": response["metrics"],
                                    "serialized_tensor_bytes": sum(map(len, result.parameters.tensors))})
        self.state.commit(updates)
        print(f"committed global version {self.state.version}/{self.state.plan['rounds']}", flush=True)
        return ndarrays_to_parameters(self.state.arrays), {"global_version": self.state.version}


def serve(state: RunState, address: str, token: str):
    manager = SimpleClientManager()
    strategy = FixedCohort(state, token)
    server = Server(client_manager=manager, strategy=strategy)
    server.set_max_workers(len(strategy.expected))
    grpc_server = start_grpc_server(manager, address, max_message_length=MAX_ARTIFACT_BYTES,
                                    max_concurrent_workers=max(8, len(strategy.expected) * 2))
    print(f"Flower coordinator listening on {address}; mode={state.plan['mode']}; "
          f"version={state.version}; clients={','.join(sorted(strategy.expected))}", flush=True)
    try:
        remaining = state.plan["rounds"] - state.version
        if remaining:
            server.fit(remaining, timeout=state.plan["round_timeout"])
        server.disconnect_all_clients(timeout=5)
    finally:
        grpc_server.stop(grace=0).wait(timeout=5)


def client_session(client, address):
    # Pinned core avoids the deprecated wrapper's unconditional telemetry
    # threads (including platform subprocesses/forks during active gRPC on macOS).
    start_client_internal(server_address=address, node_config={}, client=client,
                          grpc_max_message_length=MAX_ARTIFACT_BYTES,
                          insecure=True, max_retries=0)


def connect(client: PersistentClient, address: str, *, reconnect_seconds=60):
    import grpc

    deadline = None
    previous_version = None
    while True:
        try:
            client_session(client, address)
        except grpc.RpcError:
            version = (client.meta or {}).get("prepared_version")
            if deadline is None or version != previous_version:
                deadline = time.monotonic() + reconnect_seconds
                previous_version = version
            if time.monotonic() >= deadline:
                raise TimeoutError("coordinator disconnected; checkpoint retained; restart explicitly") from None
            time.sleep(2)
        else:
            return
