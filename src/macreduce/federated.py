"""Heterogeneous federated learning for MacReduce.

Clients differ by device, not by data mixture. A client is a GPU, a
combined GPU and Neural Engine, or a Neural Engine alone. Several Macs
repeat the same choice. The bytes on the network are a LoRA adapter
delta. The base weights stay put.

Star, ring, and tree averages match when the delta is uncompressed.
Int8 quantization uses error feedback. Unequal local step counts use
the FedNova normalization. A missing Neural Engine stack is not
replaced by the GPU. Importing this module does not train.
"""

from __future__ import annotations

import hmac
import io
import json
import threading
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .artifacts import MAX_ARTIFACT_BYTES
from .finetune import (
    CURVE_ACCUMULATION,
    CURVE_BATCH_SIZE,
    CURVE_EPOCHS,
    CURVE_TRAIN_EXAMPLES,
    DATASET_CONFIG,
    DATASET_ID,
    DATASET_REVISION,
    TRAIN_MODEL,
)

MODES = ("gpu", "gpu_ane", "split")
ALGORITHMS = ("fedavg", "allreduce")
TOPOLOGIES = ("star", "ring", "tree")
DEVICES = ("gpu", "ane", "gpu_ane")


@dataclass(frozen=True)
class Client:
    """One federated client. ``mac`` is an index, not a hostname."""

    name: str
    mac: int
    device: str


def cohort(n_macs: int, mode: str) -> tuple[Client, ...]:
    """The clients for ``n_macs`` machines that all use ``mode``.

    ``gpu`` is one GPU client per Mac. ``gpu_ane`` is one combined client
    per Mac. ``split`` is a GPU client and a Neural Engine client on every Mac.
    """

    if n_macs < 1:
        raise ValueError("a cohort needs at least one Mac")
    if mode not in MODES:
        raise ValueError(f"unknown intra-Mac mode {mode}")
    clients = []
    for mac in range(n_macs):
        if mode == "gpu":
            clients.append(Client(f"m{mac}-gpu", mac, "gpu"))
        elif mode == "gpu_ane":
            clients.append(Client(f"m{mac}-gpu-ane", mac, "gpu_ane"))
        else:
            clients.append(Client(f"m{mac}-gpu", mac, "gpu"))
            clients.append(Client(f"m{mac}-ane", mac, "ane"))
    return tuple(clients)


def iid_shards(n_records: int, n_clients: int, seed: int) -> tuple[np.ndarray, ...]:
    """Disjoint shards of one mixture. The shuffle seed is the only randomness."""

    if n_clients < 1:
        raise ValueError("shard count must be positive")
    if n_records < n_clients:
        raise ValueError("each client needs at least one record")
    order = np.random.default_rng(seed).permutation(n_records)
    return tuple(order[index::n_clients] for index in range(n_clients))


def fednova_average(
    deltas: list[np.ndarray],
    steps: list[int],
    weights: list[float] | None = None,
) -> np.ndarray:
    """Average adapter deltas after dividing out each client's step count.

    Equal step counts reduce to a weighted mean of the deltas. This is the
    plain-SGD normalization from FedNova, not a claim about Adam.
    """

    if not deltas or len(deltas) != len(steps):
        raise ValueError("each delta needs a step count")
    if any(count < 1 for count in steps):
        raise ValueError("local steps must be positive")
    if weights is None:
        weights = [1.0 / len(deltas)] * len(deltas)
    if len(weights) != len(deltas) or abs(sum(weights) - 1.0) > 1e-8:
        raise ValueError("client weights must sum to 1")
    total = np.zeros_like(deltas[0], dtype=np.float64)
    tau_eff = 0.0
    for delta, count, weight in zip(deltas, steps, weights, strict=True):
        total = total + weight * (np.asarray(delta, dtype=np.float64) / count)
        tau_eff += weight * count
    return total * tau_eff


def quantize_int8(delta: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Quantize ``delta + residual`` and return the new residual.

    The residual keeps the part int8 could not represent, so later rounds
    can send it. The scale travels with the int8 values.
    """

    pending = np.asarray(delta, dtype=np.float64) + np.asarray(residual, dtype=np.float64)
    peak = float(np.max(np.abs(pending))) if pending.size else 0.0
    scale = peak / 127.0 if peak > 0 else 1.0
    codes = np.clip(np.rint(pending / scale), -127, 127).astype(np.int8)
    reconstructed = codes.astype(np.float64) * scale
    return codes, scale, pending - reconstructed


def dequantize_int8(codes: np.ndarray, scale: float) -> np.ndarray:
    return np.asarray(codes, dtype=np.float64) * scale


def star_average(deltas: list[np.ndarray]) -> np.ndarray:
    """Equal-weight mean. The coordinator is the hub."""

    _same_shape(deltas)
    return np.mean(np.stack([np.asarray(delta, dtype=np.float64) for delta in deltas]), axis=0)


def ring_average(deltas: list[np.ndarray]) -> np.ndarray:
    """Ring all-reduce of equal-sized vectors. The result is their mean."""

    _same_shape(deltas)
    stacked = [np.asarray(delta, dtype=np.float64) for delta in deltas]
    width = stacked[0].size
    count = len(stacked)
    if width % count != 0:
        padded = [np.pad(delta.ravel(), (0, count - width % count)) for delta in stacked]
    else:
        padded = [delta.ravel() for delta in stacked]
    chunk = padded[0].size // count
    reduced = []
    for index in range(count):
        block = np.zeros(chunk, dtype=np.float64)
        for delta in padded:
            block = block + delta[index * chunk : (index + 1) * chunk]
        reduced.append(block / count)
    flat = np.concatenate(reduced)
    return flat[:width].reshape(stacked[0].shape)


def tree_average(deltas: list[np.ndarray]) -> np.ndarray:
    """Binary-tree reduction. A parent average is weighted by subtree size."""

    _same_shape(deltas)
    nodes = [(np.asarray(delta, dtype=np.float64), 1) for delta in deltas]
    while len(nodes) > 1:
        nxt = []
        for index in range(0, len(nodes), 2):
            if index + 1 == len(nodes):
                nxt.append(nodes[index])
                continue
            left, left_n = nodes[index]
            right, right_n = nodes[index + 1]
            total = left_n + right_n
            nxt.append((left * (left_n / total) + right * (right_n / total), total))
        nodes = nxt
    return nodes[0][0]


def average(deltas: list[np.ndarray], topology: str) -> np.ndarray:
    if topology == "star":
        return star_average(deltas)
    if topology == "ring":
        return ring_average(deltas)
    if topology == "tree":
        return tree_average(deltas)
    raise ValueError(f"unknown topology {topology}")


def communication_rounds(optimizer_steps: int, local_steps: int) -> dict[str, int]:
    """How many times a cohort sends an adapter delta.

    All-reduce sends one delta per optimizer step. FedAvg sends one delta
    after each block of local steps, so the same work uses fewer rounds.
    """

    if optimizer_steps < 1 or local_steps < 1:
        raise ValueError("step counts must be positive")
    if optimizer_steps % local_steps != 0:
        raise ValueError("local steps must divide the optimizer steps")
    return {"allreduce": optimizer_steps, "fedavg": optimizer_steps // local_steps}


def payload_bytes(n_params: int, quantized: bool, n_tensors: int = 1) -> int:
    """Bytes of one adapter delta. The base model is not included.

    A quantized tensor also sends an 8-byte scale. ``n_tensors`` is the
    number of those scales.
    """

    if n_params < 1 or n_tensors < 1:
        raise ValueError("an adapter delta needs parameters")
    scales = 8 * n_tensors if quantized else 0
    return n_params * (1 if quantized else 4) + scales


def round_wire_bytes(
    n_clients: int,
    n_params: int,
    topology: str,
    quantized: bool,
    n_tensors: int = 1,
) -> int:
    """Bytes moved to give every client the average, counting both directions."""

    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology}")
    if n_clients < 1:
        raise ValueError("a round needs a client")
    one = payload_bytes(n_params, quantized, n_tensors)
    if n_clients == 1 or topology == "star":
        return 0 if n_clients == 1 else 2 * n_clients * one
    if topology == "ring":
        return 2 * (n_clients - 1) * one
    return 2 * (n_clients - 1) * one


def study() -> dict:
    """The rubric. ``measured`` is false until a run writes its own evidence."""

    arms = []
    for n_macs in (1, 2):
        for mode in MODES:
            clients = cohort(n_macs, mode)
            arms.append(
                {
                    "macs": n_macs,
                    "mode": mode,
                    "clients": [asdict(client) for client in clients],
                    "needs_second_mac": n_macs > 1,
                    "measured": False,
                }
            )
    return {
        "project": "MacReduce",
        "status": "implemented",
        "measured": False,
        "dataset": {
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "revision": DATASET_REVISION,
            "mixture": "iid shards of one corpus",
            "train_examples": CURVE_TRAIN_EXAMPLES,
            "epochs": CURVE_EPOCHS,
            "batch_size": CURVE_BATCH_SIZE,
            "grad_accumulation_steps": CURVE_ACCUMULATION,
        },
        "model": TRAIN_MODEL,
        "payload": "lora_adapter_delta",
        "base_weights_sent": False,
        "algorithms": list(ALGORITHMS),
        "topologies": list(TOPOLOGIES),
        "communication": [
            "adapter_delta_not_base_weights",
            "fedavg_local_steps",
            "int8_with_error_feedback",
            "fednova_when_step_counts_differ",
            "one_delta_per_combined_client",
        ],
        "stacks": ["mlx_gpu", "coreml", "aneforge", "orion", "maderix"],
        "future_scope": ["serving placement", "speculative decoding"],
        "arms": arms,
    }


def _same_shape(deltas: list[np.ndarray]) -> None:
    if len(deltas) < 1:
        raise ValueError("a reduction needs a delta")
    shape = np.asarray(deltas[0]).shape
    if any(np.asarray(delta).shape != shape for delta in deltas):
        raise ValueError("adapter deltas must share a shape")


class PeerRequired(RuntimeError):
    """A multi-Mac round was asked for without a connected peer."""


@dataclass(frozen=True)
class LocalResult:
    """One client's adapter after its local steps."""

    state: dict[str, np.ndarray]
    steps: int
    loss: float | None = None


def copy_state(state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: np.array(value, dtype=np.float64, copy=True) for key, value in state.items()}


def state_delta(after: dict[str, np.ndarray], before: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    _same_keys(after, before)
    return {
        key: np.asarray(after[key], dtype=np.float64) - np.asarray(before[key], dtype=np.float64)
        for key in before
    }


def state_add(state: dict[str, np.ndarray], delta: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    _same_keys(state, delta)
    return {
        key: np.asarray(state[key], dtype=np.float64) + np.asarray(delta[key], dtype=np.float64)
        for key in state
    }


def aggregate_deltas(
    deltas: list[dict[str, np.ndarray]],
    steps: list[int],
    topology: str,
    normalize: bool,
) -> dict[str, np.ndarray]:
    """Average each adapter tensor. FedNova rescales before the topology mean."""

    if len(deltas) != len(steps):
        raise ValueError("each delta needs a step count")
    keys = _same_keys(*deltas)
    merged = {}
    for key in keys:
        columns = [np.asarray(delta[key], dtype=np.float64) for delta in deltas]
        if normalize:
            merged[key] = _fednova_on_topology(columns, steps, topology)
        else:
            merged[key] = average(columns, topology)
    return merged


def encode_tree(
    delta: dict[str, np.ndarray],
    residual: dict[str, np.ndarray],
    quantize: bool,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], int]:
    """Return the tree a peer can reconstruct, the new residual, and its bytes."""

    _same_keys(delta, residual)
    if not quantize:
        width = sum(int(np.asarray(value).size) for value in delta.values())
        return copy_state(delta), copy_state(residual), payload_bytes(width, False, len(delta))
    reconstructed = {}
    updated = {}
    width = 0
    for key, value in delta.items():
        codes, scale, updated[key] = quantize_int8(value, residual[key])
        reconstructed[key] = dequantize_int8(codes, scale)
        width += int(np.asarray(value).size)
    return reconstructed, updated, payload_bytes(width, True, len(delta))


def merge_device_updates(start: dict[str, np.ndarray], updates: list[LocalResult]) -> LocalResult:
    """One delta for a Mac whose GPU and Neural Engine both stepped.

    The device step counts are normalized here. The returned step count is 1
    so a later cohort average does not divide that work out a second time.
    """

    if len(updates) < 2:
        raise ValueError("a combined client needs both devices")
    deltas = [state_delta(update.state, start) for update in updates]
    steps = [update.steps for update in updates]
    merged = aggregate_deltas(deltas, steps, "star", True)
    losses = [update.loss for update in updates if update.loss is not None]
    loss = float(np.mean(losses)) if losses else None
    return LocalResult(state_add(start, merged), 1, loss)


def pack_state(state: dict[str, np.ndarray]) -> bytes:
    """Serialize an adapter delta. The base weights are not in this buffer."""

    buffer = io.BytesIO()
    np.savez(buffer, **{key: np.asarray(value) for key, value in state.items()})
    raw = buffer.getvalue()
    if len(raw) > MAX_ARTIFACT_BYTES or len(raw) < 1:
        raise ValueError("adapter delta is empty or exceeds the size cap")
    return raw


def unpack_state(raw: bytes) -> dict[str, np.ndarray]:
    if len(raw) > MAX_ARTIFACT_BYTES:
        raise ValueError("adapter delta exceeds the size cap")
    with np.load(io.BytesIO(raw)) as bundle:
        return {key: np.array(bundle[key], dtype=np.float64) for key in bundle.files}


class AdapterRelay:
    """Collect one adapter payload per client.

    ``start`` listens on loopback. A loopback relay is a transport test, not a
    second Mac. ``publish`` is the in-process path the round loop uses.
    """

    def __init__(self, names: Sequence[str], token: str, max_bytes: int = MAX_ARTIFACT_BYTES):
        if not names or len(set(names)) != len(names):
            raise ValueError("relay clients must be unique")
        if not token:
            raise ValueError("relay token required")
        self.names = tuple(names)
        self.token = token
        self.max_bytes = max_bytes
        self.payloads: dict[str, bytes] = {}
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def publish(self, name: str, payload: bytes) -> dict[str, bytes] | None:
        if name not in self.names:
            raise ValueError(f"unknown client {name}")
        if type(payload) is not bytes or not 1 <= len(payload) <= self.max_bytes:
            raise ValueError("adapter delta is empty or exceeds the size cap")
        with self._lock:
            if set(self.payloads) == set(self.names):
                self.payloads = {}
            self.payloads[name] = payload
            if set(self.payloads) == set(self.names):
                return dict(self.payloads)
        return None

    def start(self) -> str:
        relay = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if not self._allowed():
                    return
                length = int(self.headers.get("Content-Length", "0"))
                if length < 1 or length > relay.max_bytes:
                    self.send_error(413)
                    return
                name = parse_qs(urlsplit(self.path).query).get("client", [""])[0]
                try:
                    done = relay.publish(name, self.rfile.read(length))
                except ValueError:
                    self.send_error(400)
                    return
                self._json({"complete": done is not None})

            def do_GET(self):
                if not self._allowed():
                    return
                with relay._lock:
                    ready = set(relay.payloads) == set(relay.names)
                if not ready:
                    self.send_error(409)
                    return
                self._json({"complete": True, "clients": list(relay.names)})

            def log_message(self, fmt, *args):
                return

            def _allowed(self) -> bool:
                header = self.headers.get("Authorization", "")
                expected = "Bearer " + relay.token
                if len(header) != len(expected) or not _digest_equal(header, expected):
                    self.send_error(401)
                    return False
                return True

            def _json(self, body: dict) -> None:
                raw = json.dumps(body).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        host, port = self._server.server_address
        return f"http://{host}:{port}"

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None


def run_federated(
    records: Sequence,
    initial_state: dict[str, np.ndarray],
    local_update: Callable[[Client, dict[str, np.ndarray], list, int], LocalResult],
    *,
    n_macs: int,
    mode: str,
    algorithm: str,
    topology: str,
    optimizer_steps: int,
    local_steps: int,
    quantize: bool = False,
    normalize: bool = False,
    seed: int = 0,
    relay: AdapterRelay | None = None,
    evaluate: Callable[[dict[str, np.ndarray]], float] | None = None,
    eval_every: int | None = None,
    base_nbytes: int = 0,
    trainer: str = "injected",
) -> dict:
    """Run FedAvg or per-step all-reduce over adapter deltas.

    One Mac aggregates in process. More than one Mac requires ``relay``.
    This function does not invent the missing machine. ``measured`` stays
    false: a caller that ran QLoRA writes its own evidence file.
    """

    if algorithm not in ALGORITHMS:
        raise ValueError(f"unknown algorithm {algorithm}")
    if optimizer_steps < 1 or local_steps < 1:
        raise ValueError("step counts must be positive")
    if eval_every is not None and eval_every < 1:
        raise ValueError("validation spacing must be positive")
    clients = cohort(n_macs, mode)
    if n_macs > 1 and relay is None:
        raise PeerRequired("a second Mac is not connected")
    if relay is not None and tuple(relay.names) != tuple(client.name for client in clients):
        raise ValueError("relay clients do not match the cohort")
    if algorithm == "allreduce":
        block_steps = 1
        n_rounds = optimizer_steps
    else:
        if optimizer_steps % local_steps != 0:
            raise ValueError("local steps must divide the optimizer steps")
        block_steps = local_steps
        n_rounds = optimizer_steps // local_steps
    shards = iid_shards(len(records), len(clients), seed)
    global_state = copy_state(initial_state)
    residuals = {
        client.name: {key: np.zeros_like(value) for key, value in global_state.items()}
        for client in clients
    }
    history = []
    for round_index in range(n_rounds):
        results = []
        for client, shard in zip(clients, shards, strict=True):
            local_records = [records[int(index)] for index in shard]
            result = local_update(client, copy_state(global_state), local_records, block_steps)
            if result.steps < 1:
                raise ValueError("a client reported no local steps")
            results.append(result)
        encoded = []
        payload = 0
        for client, result in zip(clients, results, strict=True):
            delta = state_delta(result.state, global_state)
            reconstructed, residuals[client.name], nbytes = encode_tree(
                delta, residuals[client.name], quantize
            )
            encoded.append(reconstructed)
            payload += nbytes
        if relay is not None:
            done = None
            for client, reconstructed in zip(clients, encoded, strict=True):
                done = relay.publish(client.name, pack_state(reconstructed))
            if done is None:
                raise PeerRequired("the relay is missing a client")
            encoded = [unpack_state(done[client.name]) for client in clients]
        steps = [result.steps for result in results]
        global_state = state_add(
            global_state, aggregate_deltas(encoded, steps, topology, normalize)
        )
        row = {
            "round": round_index,
            "client_steps": {client.name: result.steps for client, result in zip(clients, results)},
            "client_loss": {client.name: result.loss for client, result in zip(clients, results)},
            "payload_bytes": payload,
            "wire_bytes": round_wire_bytes(
                len(clients),
                _width(global_state),
                topology,
                quantize,
                len(global_state),
            ),
            "base_weights_sent": False,
        }
        done = (round_index + 1) * block_steps
        if evaluate is not None and (eval_every is None or done % eval_every == 0):
            row["validation"] = float(evaluate(copy_state(global_state)))
        history.append(row)
    return {
        "project": "MacReduce",
        "measured": False,
        "trainer": trainer,
        "algorithm": algorithm,
        "topology": topology,
        "quantize": quantize,
        "fednova": normalize,
        "macs": n_macs,
        "mode": mode,
        "clients": [asdict(client) for client in clients],
        "rounds": n_rounds,
        "wire_bytes": sum(row["wire_bytes"] for row in history),
        "base_weights_sent": False,
        "base_nbytes_not_sent": base_nbytes,
        "state": global_state,
        "history": history,
    }


def reference_update(
    client: Client,
    state: dict[str, np.ndarray],
    records: list,
    steps: int,
    targets: dict[str, np.ndarray],
    lr: float = 1.0,
) -> LocalResult:
    """CPU SGD toward a client target. This is the numerical reference, not QLoRA."""

    del records
    if lr <= 0 or steps < 1:
        raise ValueError("learning rate and steps must be positive")
    weights = np.array(state["adapter"], dtype=np.float64, copy=True)
    target = np.asarray(targets[client.name], dtype=np.float64)
    for _ in range(steps):
        weights = weights - lr * (weights - target)
    loss = float(np.mean((weights - target) ** 2))
    return LocalResult({"adapter": weights}, steps, loss)


def _fednova_on_topology(columns: list[np.ndarray], steps: list[int], topology: str) -> np.ndarray:
    if any(count < 1 for count in steps):
        raise ValueError("local steps must be positive")
    weights = [1.0 / len(columns)] * len(columns)
    scaled = [column / count for column, count in zip(columns, steps, strict=True)]
    tau_eff = sum(weight * count for weight, count in zip(weights, steps, strict=True))
    return average(scaled, topology) * tau_eff


def _same_keys(*states: dict[str, np.ndarray]) -> tuple[str, ...]:
    if not states or not states[0]:
        raise ValueError("an adapter state needs a tensor")
    keys = tuple(states[0])
    for state in states[1:]:
        if tuple(state) != keys:
            raise ValueError("adapter states must share keys")
    return keys


def _width(state: dict[str, np.ndarray]) -> int:
    return sum(int(np.asarray(value).size) for value in state.values())


def _digest_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)
