import contextlib
import hashlib
import json
import math
import ssl
import time
import urllib.error
import urllib.request
from urllib.parse import urlencode

import numpy as np

from .artifacts import (
    CONTROL_RESPONSE_BYTES,
    MAX_CHUNK_SIZE,
    assemble_chunks,
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
)
from .observe import Events, Sampler, capabilities


class Client:
    def __init__(self, url, token, events, ca_file=None, retry_seconds=60):
        if not url.startswith(("http://", "https://")):
            raise ValueError("coordinator URL must use http:// or https://")
        self.url, self.token, self.events = url.rstrip("/"), token, events
        self.context = ssl.create_default_context(cafile=ca_file)
        self.retry_seconds = retry_seconds

    def request(self, endpoint, payload=None, max_bytes=CONTROL_RESPONSE_BYTES, **query):
        url = self.url + endpoint + ("?" + urlencode(query) if query else "")
        data = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        deadline = time.monotonic() + self.retry_seconds
        attempts = 0
        while True:
            attempts += 1
            try:
                req = urllib.request.Request(
                    url,
                    data=data,
                    headers={
                        "Authorization": "Bearer " + self.token,
                        "Content-Type": "application/octet-stream",
                    },
                )

                # Do not follow redirects with a bearer token to another endpoint.
                class NoRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, *args, **kwargs):
                        return None

                opener = urllib.request.build_opener(
                    NoRedirect(),
                    urllib.request.HTTPSHandler(context=self.context),
                    urllib.request.ProxyHandler({}),
                )
                with self.events.span("http" + endpoint, attempt=attempts):
                    with opener.open(req, timeout=min(30, self.retry_seconds)) as response:
                        raw = response.read(max_bytes + 1)
                        if len(raw) > max_bytes:
                            raise ValueError("server response exceeds transfer response cap")
                        kind = response.headers.get("Content-Type", "")
                self.events.emit(
                    "transfer",
                    endpoint=endpoint,
                    attempt=attempts,
                    sent_payload_bytes=len(data or b""),
                    received_payload_bytes=len(raw),
                )
                return json.loads(raw) if kind.startswith("application/json") else raw
            except urllib.error.HTTPError as exc:
                if exc.code < 500:
                    raise
                error = f"HTTP {exc.code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                error = str(exc)
            self.events.emit(
                "request_retry",
                endpoint=endpoint,
                attempt=attempts,
                error=error,
                attempted_payload_bytes=len(data or b""),
            )
            if time.monotonic() >= deadline:
                raise TimeoutError(f"coordinator unavailable for {self.retry_seconds}s: {error}")
            time.sleep(min(0.2 * 2 ** min(attempts, 4), 2))

    def fetch_artifact(self, version, run_id, expected_sha256):
        manifest = validate_manifest(
            self.request("/artifact/manifest", kind="checkpoint", version=version, run_id=run_id)
        )
        if manifest["sha256"] != expected_sha256:
            raise ValueError("checkpoint transfer checksum mismatch")

        def read_chunk(index):
            return self.request(
                "/artifact/chunk",
                max_bytes=manifest["chunk_size"],
                kind="checkpoint",
                version=version,
                index=index,
                run_id=run_id,
            )

        return assemble_chunks(manifest, read_chunk)

    def push_artifact(
        self, raw, *, node, version, tokens, run_id, parameter_names, plan, chunk_size=MAX_CHUNK_SIZE
    ):
        manifest = build_manifest(
            raw,
            run_id=run_id,
            version=version,
            role="update",
            parameter_names=parameter_names,
            plan=plan,
            node=node,
            chunk_size=chunk_size,
        )
        started = self.request(
            "/artifact/update", manifest, node=node, version=version, tokens=tokens, run_id=run_id
        )
        if started.get("duplicate"):
            return started
        upload_id = started["upload_id"]
        for index, expected in enumerate(manifest["chunk_hashes"]):
            start = index * manifest["chunk_size"]
            chunk = raw[start : start + manifest["chunk_size"]]
            if hashlib.sha256(chunk).hexdigest() != expected:
                raise ValueError("local chunk hash mismatch")
            self.request("/artifact/chunk", chunk, upload_id=upload_id, index=index)
        return self.request("/artifact/complete", b"", upload_id=upload_id)


def run_worker(
    url,
    token,
    node,
    directory,
    backend_name="mlx-gpu",
    batch_size=4,
    train_text=None,
    ca_file=None,
    capture=None,
    torch_trace=None,
):
    if not 1 <= batch_size <= 64:
        raise ValueError("batch size must be between 1 and 64")
    events = Events(directory, node)
    client = Client(url, token, events, ca_file)
    data, digest = corpus(seed=1, path=train_text)
    with events.span("backend_probe"):
        # Fail before joining the cohort if the requested runtime cannot start.
        backend(backend_name, ModelConfig(), initial_weights())
    info = {
        "backend": backend_name,
        "batch_size": batch_size,
        "data_sha256": digest,
        "capabilities": capabilities(),
        "backend_execution": "initialized",
    }
    registration = client.request("/register", info, node=node)
    events.run_id = registration["run_id"]
    rank = registration["rank"]
    model = ModelConfig.from_mapping(registration["config"]["model"])
    with events.span("backend_init"):
        trainer = backend(backend_name, model, initial_weights(model))
    events.emit("worker_started", info=info)
    captured = False
    profiled = False
    if torch_trace and backend_name != "torch-cpu":
        raise ValueError("--torch-trace requires --backend torch-cpu")
    if (torch_trace or capture) and registration["config"]["steps"] < 2:
        raise ValueError("operator capture needs at least two local steps for a warmup")
    with Sampler(events):
        while True:
            with events.span("barrier_wait"):
                job = client.request("/job", node=node)
                while job["state"] == "wait":
                    time.sleep(0.15)
                    job = client.request("/job", node=node)
            if job["run_id"] != events.run_id:
                raise RuntimeError("coordinator run changed; restart this worker explicitly")
            if job["state"] == "done":
                events.emit("worker_finished", version=job["version"])
                return job
            if job["state"] == "failed":
                raise RuntimeError(job["failure"])
            version, config = job["version"], job["config"]
            if ModelConfig.from_mapping(config["model"]) != model:
                raise RuntimeError("coordinator changed the model configuration")
            raw = client.fetch_artifact(version, events.run_id, job["checkpoint_sha256"])
            with events.span("checkpoint_load", version=version):
                trainer.load(decode_weights(raw, config=model))
            # A replay of an uncommitted round samples exactly the same local mini-batches.
            rng = np.random.default_rng(np.random.SeedSequence([config["seed"], rank, version]))
            tokens, loss = 0, None
            for step in range(config["steps"]):
                with events.span("data_batch", version=version, step=step):
                    x, y = batch(data, rng, batch_size, model.context)
                capture_now = capture is not None and not captured and step == 1
                profile_now = torch_trace is not None and not profiled and step == 1
                operator_profile = (
                    trainer.t.profiler.profile(
                        activities=[trainer.t.profiler.ProfilerActivity.CPU],
                        record_shapes=True,
                        profile_memory=True,
                    )
                    if profile_now
                    else contextlib.nullcontext()
                )
                if capture_now:
                    if backend_name != "mlx-gpu":
                        raise ValueError("Metal capture requires --backend mlx-gpu")
                    trainer.mx.metal.start_capture(capture)
                try:
                    with (
                        operator_profile,
                        events.span(
                            "train_step",
                            version=version,
                            step=step,
                            tokens=int(y.size),
                            backend=backend_name,
                        ),
                    ):
                        loss = trainer.step(x, y, config["lr"])
                finally:
                    if capture_now:
                        trainer.mx.metal.stop_capture()
                        captured = True
                if profile_now:
                    operator_profile.export_chrome_trace(torch_trace)
                    profiled = True
                    events.emit(
                        "operator_profile_saved",
                        path=torch_trace,
                        scope="PyTorch CPU operations",
                        version=version,
                        step=step,
                    )
                if not math.isfinite(loss):
                    raise ValueError("non-finite local training loss")
                tokens += int(y.size)
                events.emit("training_loss", version=version, step=step, loss=loss)
            with events.span("checkpoint_export", version=version):
                exported = trainer.export()
                outgoing = encode_weights(exported, model)
            events.emit(
                "round_local_complete",
                version=version,
                tokens=tokens,
                loss=loss,
                **trainer.memory(),
            )
            client.push_artifact(
                outgoing,
                node=node,
                version=version,
                tokens=tokens,
                run_id=events.run_id,
                parameter_names=list(exported),
                plan=plan_sha256(model),
            )
