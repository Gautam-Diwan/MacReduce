import argparse
import json
import os
import secrets
import ssl
import subprocess
import sys
import threading
import time
from pathlib import Path

from .benchmark import accelerate_matmul
from .cluster import Coordinator, make_server
from .observe import capabilities, collect_mactop, report
from .worker import run_worker


def token():
    value = os.environ.get("MACREDUCE_TOKEN", "")
    if len(value) < 24:
        raise ValueError("set MACREDUCE_TOKEN to a shared random secret of at least 24 characters")
    return value


def demo(args):
    directory = Path(args.output).resolve()
    coordinator = Coordinator(
        directory,
        ["small", "large"],
        args.rounds,
        args.steps,
        args.lr,
        args.backend,
        round_timeout=180,
    )
    secret = secrets.token_urlsafe(32)
    server = make_server(coordinator, "127.0.0.1", 0, secret)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    children, handles = [], []
    initial_loss = coordinator.validation_loss
    started = time.monotonic()
    try:
        for node, size in (("small", 2), ("large", 4)):
            log = (directory / f"{node}.log").open("w")
            handles.append(log)
            cmd = [
                sys.executable,
                "-m",
                "macreduce",
                "worker",
                "--url",
                f"http://127.0.0.1:{server.server_port}",
                "--node",
                node,
                "--batch-size",
                str(size),
                "--backend",
                args.backend,
                "--output",
                str(directory / "events"),
            ]
            if args.torch_trace and node == "small":
                cmd.extend(["--torch-trace", str(directory / "small.operators.trace.json")])
            children.append(
                subprocess.Popen(
                    cmd,
                    env=dict(os.environ, MACREDUCE_TOKEN=secret),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        deadline = time.monotonic() + 300
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children):
                raise RuntimeError(f"worker failed; inspect logs in {directory}")
            if time.monotonic() > deadline:
                raise TimeoutError("demo did not finish within 300 seconds")
            time.sleep(0.2)
        if any(p.returncode for p in children):
            raise RuntimeError("worker failed")
        result = {
            "status": coordinator.status(),
            "initial_validation_loss": initial_loss,
            "final_validation_loss": coordinator.validation_loss,
            "elapsed_seconds": time.monotonic() - started,
            "backend": args.backend,
            "transport": "HTTP over loopback",
            "physical_machines": 1,
            "worker_processes": 2,
            "batch_sizes": [2, 4],
            "wifi_tested": False,
            "ane_execution_tested": False,
        }
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        report(directory)
        print(json.dumps(result, indent=2))
        if result["status"]["state"] != "done":
            raise RuntimeError("run did not complete")
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        for handle in handles:
            handle.close()


def main():
    p = argparse.ArgumentParser(description="MacReduce: observable local SGD research prototype")
    sub = p.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="inventory hardware and runtime availability")
    doctor.add_argument("--smoke", action="store_true", help="attempt an MLX GPU matmul")
    doctor.add_argument("--output")
    for name in ("coordinator", "demo"):
        a = sub.add_parser(name)
        a.add_argument("--output", required=True)
        a.add_argument("--rounds", type=int, default=8)
        a.add_argument("--steps", type=int, default=10)
        a.add_argument("--lr", type=float, default=0.05)
        a.add_argument(
            "--backend",
            choices=["torch-cpu", "mlx-cpu", "mlx-gpu"],
            default="torch-cpu",
            help="coordinator evaluation backend (and demo workers)",
        )
        if name == "coordinator":
            a.add_argument("--nodes", required=True, help="comma-separated fixed worker IDs")
            a.add_argument("--host", default="127.0.0.1")
            a.add_argument("--port", type=int, default=8765)
            a.add_argument("--round-timeout", type=float, default=120)
            a.add_argument("--resume", action="store_true")
            a.add_argument("--validation-text")
            a.add_argument("--tls-cert")
            a.add_argument("--tls-key")
        else:
            a.add_argument("--torch-trace", action="store_true", help="profile one CPU step")
    w = sub.add_parser("worker")
    w.add_argument("--url", required=True)
    w.add_argument("--node", required=True)
    w.add_argument("--output", required=True)
    w.add_argument("--backend", choices=["mlx-gpu", "mlx-cpu", "torch-cpu"], default="mlx-gpu")
    w.add_argument("--batch-size", type=int, default=4)
    w.add_argument("--train-text")
    w.add_argument("--ca-file")
    w.add_argument("--capture", help="new .gputrace path; requires MTL_CAPTURE_ENABLED=1")
    w.add_argument("--torch-trace", help="export a CPU operator trace for one step after warmup")
    r = sub.add_parser("report")
    r.add_argument("directory")
    t = sub.add_parser("telemetry")
    t.add_argument("--output", required=True)
    t.add_argument("--seconds", type=float, default=10)
    t.add_argument("--interval-ms", type=int, default=1000)
    b = sub.add_parser("matmul", help="measure public Accelerate CPU BLAS")
    b.add_argument("--size", type=int, default=512)
    b.add_argument("--repeats", type=int, default=10)
    b.add_argument("--output")
    args = p.parse_args()
    try:
        if args.command == "doctor":
            text = json.dumps(capabilities(args.smoke), indent=2)
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(text)
            print(text)
        elif args.command == "demo":
            demo(args)
        elif args.command == "coordinator":
            c = Coordinator(
                args.output,
                args.nodes.split(","),
                args.rounds,
                args.steps,
                args.lr,
                args.backend,
                args.round_timeout,
                args.resume,
                args.validation_text,
            )
            server = make_server(c, args.host, args.port, token())
            if bool(args.tls_cert) != bool(args.tls_key):
                raise ValueError("provide both --tls-cert and --tls-key")
            if args.tls_cert:
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(args.tls_cert, args.tls_key)
                server.socket = context.wrap_socket(server.socket, server_side=True)
            print(
                f"Coordinator ready on {args.host}:{server.server_port}; run {c.run_id}", flush=True
            )
            try:
                server.serve_forever()
            finally:
                server.server_close()
        elif args.command == "worker":
            run_worker(
                args.url,
                token(),
                args.node,
                args.output,
                args.backend,
                args.batch_size,
                args.train_text,
                args.ca_file,
                args.capture,
                args.torch_trace,
            )
        elif args.command == "report":
            print(report(args.directory))
        elif args.command == "telemetry":
            if args.seconds <= 0 or args.interval_ms < 100:
                raise ValueError("seconds must be positive and interval at least 100 ms")
            print(json.dumps(collect_mactop(args.output, args.seconds, args.interval_ms), indent=2))
        elif args.command == "matmul":
            result = json.dumps(accelerate_matmul(args.size, args.repeats), indent=2)
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(result)
            print(result)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        p.exit(1, f"{type(exc).__name__}: {exc}\n")
