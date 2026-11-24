"""Explicit, hashable remote-federation plans. No hardware imports or secrets."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
import subprocess
from pathlib import Path

import yaml

from .finetune import DATASET_REVISION, DEFAULT_RECIPE, TRAIN_MODEL, validate_recipe
from .serving import MODEL_REVISIONS

MODES = {"gpu": ("gpu",), "gpu_ane": ("gpu_ane",), "split": ("gpu", "ane")}


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def implementation_hash() -> str:
    root = Path(__file__).parent
    digest = hashlib.sha256()
    files = sorted(root.glob("federation_*.py")) + [root / "flower_federation.py"]
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def positive(value, name, maximum=1_000_000):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")
    return value


def load_plan(path: Path, mode: str | None = None) -> dict:
    body = yaml.safe_load(path.read_text())
    if not isinstance(body, dict) or body.get("schema") != 1:
        raise ValueError("federation config requires schema: 1")
    mode = mode or body.get("mode", "gpu")
    if mode not in MODES:
        raise ValueError("mode must be gpu, gpu_ane, or split")
    machines = body.get("machines")
    if not isinstance(machines, dict) or not 1 <= len(machines) <= 8:
        raise ValueError("configure between one and eight named machines")
    coordinator = body.get("coordinator")
    if not isinstance(coordinator, str) or coordinator not in machines:
        raise ValueError("coordinator must name one of the configured machines")
    clients = []
    for machine, settings in machines.items():
        if not isinstance(machine, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", machine):
            raise ValueError("machine IDs must be short lowercase names with hyphens")
        if not isinstance(settings, dict):
            raise ValueError("machine settings must be a mapping")
        for device in MODES[mode]:
            local = {**settings, **settings.get(device, {})}
            checkpoint = local.get("grad_checkpoint", False)
            if type(checkpoint) is not bool:
                raise ValueError("grad_checkpoint must be true or false; calibration is a TODO")
            weight = local.get("weight", 1.0)
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ValueError("objective weight must be numeric")
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("objective weight must be finite and positive")
            clients.append({
                "id": f"{machine}-{device}", "machine": machine, "device": device,
                "microbatch": positive(local.get("microbatch", 1), "microbatch", 64),
                "accumulation": positive(local.get("accumulation", 1), "accumulation", 64),
                "grad_checkpoint": checkpoint, "weight": float(weight),
            })
    total = sum(row["weight"] for row in clients)
    for row in clients:
        row["weight"] /= total
    training = body.get("training", {})
    recipe = validate_recipe({**DEFAULT_RECIPE, **body.get("recipe", {})})
    if recipe["dropout"] != 0:
        raise ValueError("remote deterministic recovery currently requires LoRA dropout 0")
    lr = training.get("learning_rate", 1e-4)
    if isinstance(lr, bool) or not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    deadline = training.get("round_timeout", 1800)
    positive(deadline, "round_timeout", 86400)
    seed = training.get("seed", 1)
    if type(seed) is not int or not 0 <= seed <= 1_000_000:
        raise ValueError("seed must be a nonnegative integer")
    sequence = positive(training.get("max_seq_length", 512), "sequence", 4096)
    if sequence < 2:
        raise ValueError("sequence must be at least two tokens")
    return {
        "schema": 1, "mode": mode, "coordinator": coordinator, "clients": clients,
        "implementation_sha256": implementation_hash(),
        "model": TRAIN_MODEL, "model_revision": MODEL_REVISIONS[TRAIN_MODEL],
        "dataset_revision": DATASET_REVISION,
        "train_examples": positive(body.get("data", {}).get("train_examples", 4096),
                                   "train_examples", 100000),
        "seed": seed,
        "max_seq_length": sequence,
        "rounds": positive(training.get("rounds", 3), "rounds", 100000),
        "local_steps": positive(training.get("local_steps", 2), "local_steps", 10000),
        "learning_rate": float(lr), "round_timeout": deadline, "recipe": recipe,
        "scope": "explicit fixed-round pilot; hardware batch/sync calibration is TODO",
    }


def client_spec(plan: dict, name: str) -> dict:
    for row in plan["clients"]:
        if row["id"] == name:
            return row
    raise ValueError(f"client {name!r} is not in this {plan['mode']} cohort")


def data_identity(records: list[dict]) -> str:
    return fingerprint([row["text"] for row in records])


def shard_indices(plan: dict, name: str, size: int) -> list[int]:
    """Disjoint seeded IID shards; weights are explicit, not inferred from speed."""
    import numpy as np

    names = [row["id"] for row in plan["clients"]]
    if size < len(names):
        raise ValueError("the training prefix needs at least one record per client")
    rank = names.index(name)
    order = np.random.default_rng(plan["seed"]).permutation(size)
    return [int(index) for index in order[rank::len(names)]]


def bind_address(value: str) -> str:
    if value == "tailscale":
        result = subprocess.run(["tailscale", "ip", "-4"], capture_output=True,
                                text=True, timeout=10, check=True)
        value = result.stdout.strip().splitlines()[0]
    address = ipaddress.ip_address(value)
    if address.version != 4 or not (address.is_loopback or address in ipaddress.ip_network(
        "100.64.0.0/10"
    )):
        raise ValueError("bind to loopback or the Mac's Tailscale IPv4; wildcard/public binds refused")
    return str(address)
