import json
import threading
import time
import urllib.error
import urllib.request

import numpy as np
import pytest

from macreduce.cluster import Conflict, Coordinator, make_server
from macreduce.model import encode_weights
from macreduce.observe import Events
from macreduce.worker import Client


@pytest.fixture
def coordinator(tmp_path):
    c = Coordinator(tmp_path / "run", ["a", "b"], rounds=2, steps=1)
    for n, batch in (("a", 2), ("b", 4)):
        c.register(n, {"batch_size": batch, "backend": "torch-cpu", "data_sha256": "test"})
    return c


def test_barrier_weighting_duplicate_and_recovery(coordinator):
    c = coordinator
    initial = {k: v.copy() for k, v in c.weights.items()}
    a = encode_weights({k: v + np.float32(0.01) for k, v in initial.items()})
    b = encode_weights({k: v + np.float32(0.02) for k, v in initial.items()})
    c.submit("a", 0, 64, a, c.run_id)
    assert c.version == 0 and c.job("a")["state"] == "wait"
    assert c.submit("a", 0, 64, a, c.run_id)["duplicate"]
    c.submit("b", 0, 128, b, c.run_id)
    assert c.version == 1
    for k, value in initial.items():
        np.testing.assert_allclose(c.weights[k], value + 0.05 / 3, atol=2e-7)
    restored = Coordinator(c.root, ["a", "b"], rounds=2, steps=1, resume=True)
    assert restored.version == 1 and restored.run_id == c.run_id
    assert restored.submit("b", 0, 128, b, c.run_id)["duplicate"]
    with pytest.raises(Conflict):
        restored.submit("a", 0, 64, b, c.run_id)
    with pytest.raises(Conflict):
        restored.submit("a", 1, 64, a, "different-run")


def test_reject_wrong_tokens_future_and_worker(coordinator):
    c = coordinator
    with pytest.raises(ValueError, match="token"):
        c.submit("a", 0, 1, c.raw, c.run_id)
    with pytest.raises(Conflict):
        c.submit("a", 3, 64, c.raw, c.run_id)
    with pytest.raises(Conflict):
        c.submit("unknown", 0, 64, c.raw, c.run_id)
    assert c.pending == {}


def test_deadline_does_not_drop_worker(coordinator):
    c = coordinator
    c.started = time.monotonic() - 200
    assert c.status()["state"] == "failed"
    assert c.version == 0
    with pytest.raises(Conflict):
        c.submit("a", 0, 64, c.raw, c.run_id)


def test_uncommitted_round_is_replayed(coordinator):
    c = coordinator
    c.submit("a", 0, 64, c.raw, c.run_id)
    restored = Coordinator(c.root, ["a", "b"], rounds=2, steps=1, resume=True)
    assert restored.version == 0 and not restored.pending
    assert restored.job("a")["state"] == "train"


def test_resume_verifies_checkpoint(coordinator):
    c = coordinator
    state = json.loads((c.root / "state.json").read_text())
    (c.root / state["checkpoint"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="checksum"):
        Coordinator(c.root, ["a", "b"], rounds=2, steps=1, resume=True)


def test_real_http_auth_payload_bound_and_idempotent_retry(coordinator, tmp_path):
    secret = "t" * 32
    server = make_server(coordinator, "127.0.0.1", 0, secret)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}"
    client = Client(url, secret, Events(tmp_path / "events", "test"))
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(url + "/status")
        assert error.value.code == 401
        assert client.request("/status")["state"] == "running"
        raw = client.request("/checkpoint", version=0)
        kwargs = dict(node="a", version=0, tokens=64, run_id=coordinator.run_id)
        assert client.request("/update", raw, **kwargs)["accepted"]
        # Simulates a lost response: retry the same bytes rather than running more SGD steps.
        assert client.request("/update", raw, **kwargs)["duplicate"]
        assert b"macreduce_version 0" in client.request("/metrics")
        with pytest.raises(urllib.error.HTTPError) as error:
            client.request("/register", b"x" * 70000, node="a")
        assert error.value.code == 413
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_client_retries_after_server_accepts_but_response_is_lost(coordinator, tmp_path):
    original = coordinator.submit
    attempts = []

    def lose_first_response(*args, **kwargs):
        result = original(*args, **kwargs)
        attempts.append(result)
        if len(attempts) == 1:
            raise ConnectionResetError("simulated lost acknowledgement")
        return result

    coordinator.submit = lose_first_response
    secret = "t" * 32
    server = make_server(coordinator, "127.0.0.1", 0, secret)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = Client(
        f"http://127.0.0.1:{server.server_port}",
        secret,
        Events(tmp_path / "retry-events", "test"),
        retry_seconds=3,
    )
    try:
        result = client.request(
            "/update", coordinator.raw, node="a", version=0, tokens=64, run_id=coordinator.run_id
        )
        assert result["duplicate"]
        assert len(attempts) == 2 and len(coordinator.pending) == 1
        assert coordinator.version == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
