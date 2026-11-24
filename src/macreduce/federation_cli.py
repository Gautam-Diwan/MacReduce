"""Remote federation commands; explicit pilots, never implicit training."""

from __future__ import annotations

import gc
import json
import os
import socket
from pathlib import Path

from .federation_plan import MODES, bind_address, client_spec, data_identity, load_plan


def add_parser(sub):
    parser = sub.add_parser("federation", help="Flower-based federation over Tailscale")
    actions = parser.add_subparsers(dest="federation_action", required=True)
    for action in ("plan", "prepare", "check", "coordinator", "worker"):
        command = actions.add_parser(action)
        command.add_argument("--config", default="configs/federation.yaml")
        command.add_argument("--mode", choices=list(MODES))
        command.add_argument("--data", default="data/wikitext-2-raw-v1")
        if action in {"plan", "check"}:
            command.add_argument("--output")
        if action == "check":
            command.add_argument("--server", help="optional TCP reachability check, HOST:PORT")
            command.add_argument("--client")
            command.add_argument("--execute", action="store_true", help="real model forward/backward preflight")
        if action == "coordinator":
            command.add_argument("--host", default="tailscale", help="tailscale, Tailscale IPv4, or loopback")
            command.add_argument("--port", type=int, default=8765)
            command.add_argument("--output", required=True)
            command.add_argument("--resume", action="store_true")
        if action == "worker":
            command.add_argument("--server", required=True, help="Tailscale HOST:PORT (gRPC, no URL scheme)")
            command.add_argument("--client", required=True)
            command.add_argument("--output", required=True)
            command.add_argument("--reconnect-seconds", type=int, default=60)


def _records(args, plan):
    from .lora_client import load_curve_records

    directory = Path(args.data)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("revision") != plan["dataset_revision"]:
        raise ValueError("local dataset revision differs; run federation prepare")
    return load_curve_records(directory, plan["train_examples"])


def _backend(plan, name, records):
    from .federation_backend import ResidentLoraClient

    backend = ResidentLoraClient(plan, client_spec(plan, name), records)
    backend.setup()
    return backend


def _address(value):
    if "://" in value or value.count(":") != 1:
        raise ValueError("server must be a Tailscale hostname/IPv4 and port, e.g. m4-air:8765")
    host, port = value.rsplit(":", 1)
    if not host or not 1 <= int(port) <= 65535:
        raise ValueError("invalid server address")
    # Plain gRPC is restricted to the encrypted tailnet or a loopback SSH tunnel.
    resolved = socket.getaddrinfo(host, int(port), family=socket.AF_INET, type=socket.SOCK_STREAM)
    for result in resolved:
        bind_address(result[4][0])
    return host, int(port)


def _require_combination(plan):
    if plan["mode"] == "split":
        from .lora_client import BackendUnavailable

        raise BackendUnavailable(
            "split cohort has six logical clients but standalone ANE Qwen3 training is not "
            "implemented; refusing to start a partial GPU-only cohort"
        )


def execute(args, get_token):
    plan = load_plan(Path(args.config), args.mode)
    action = args.federation_action
    if action == "plan":
        result = {**plan, "backend_support": {
            "gpu": "implemented; each device requires execution preflight",
            "gpu_ane": "existing ANE suffix; each device requires execution preflight",
            "ane": "unavailable: standalone Qwen3 trainer is not implemented",
        }}
    elif action == "prepare":
        from .finetune import prepare_dataset

        prepare_dataset(Path(args.data))
        result = {"data_sha256": data_identity(_records(args, plan)),
                  "train_examples": plan["train_examples"]}
    elif action == "check":
        result = {"plan": plan, "network": "not checked", "backend": "not executed"}
        if args.server:
            host, port = _address(args.server)
            with socket.create_connection((host, port), timeout=5):
                result["network"] = "TCP reachable; authenticated identity checked when worker joins"
        if args.execute:
            if not args.client:
                raise ValueError("--execute preflight requires --client")
            _require_combination(plan)
            backend = _backend(plan, args.client, _records(args, plan))
            result["backend"] = {"forward_backward_preflight": "passed",
                                 "placement": backend.placement()}
    else:
        _require_combination(plan)
        secret = get_token()
        records = _records(args, plan)
        data_hash = data_identity(records)
        # Keep optional upstream analytics off the critical training/network path.
        os.environ.setdefault("FLWR_TELEMETRY_ENABLED", "0")
        try:
            from .flower_federation import PersistentClient, connect, serve
        except ModuleNotFoundError as exc:
            if exc.name == "flwr":
                raise RuntimeError("install the federation extra: uv sync --extra federation") from exc
            raise
        if action == "coordinator":
            from .federation_backend import ResidentLoraClient
            from .federation_state import RunState

            host = bind_address(args.host)
            if not 1 <= args.port <= 65535:
                raise ValueError("port must be between 1 and 65535")
            initial = None
            if not args.resume:
                # Bootstrap a common adapter once, then free the model. Reduction is CPU NumPy.
                spec = {**plan["clients"][0], "device": "gpu", "grad_checkpoint": False}
                bootstrap = ResidentLoraClient(plan, spec, records)
                initial = bootstrap.setup()
                del bootstrap
                gc.collect()
                import mlx.core as mx

                mx.clear_cache()
            state = RunState(Path(args.output), plan, data_hash, initial, resume=args.resume)
            serve(state, f"{host}:{args.port}", secret)
            result = {"run_id": state.meta["run_id"], "committed_version": state.version,
                      "scope": plan["scope"]}
        else:
            _address(args.server)
            if not 1 <= args.reconnect_seconds <= 86400:
                raise ValueError("reconnect-seconds must be between 1 and 86400")
            backend = _backend(plan, args.client, records)
            client = PersistentClient(backend, plan, args.client, data_hash,
                                      Path(args.output), secret)
            connect(client, args.server, reconnect_seconds=args.reconnect_seconds)
            result = {"client": args.client, "checkpoint": str(Path(args.output)),
                      "scope": plan["scope"]}
    text = json.dumps(result, indent=2, allow_nan=False)
    if action in {"plan", "check"} and args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n")
    print(text)
