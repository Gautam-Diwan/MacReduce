import hashlib
import json
import threading
import urllib.error

import pytest

from macreduce.artifacts import MAX_ARTIFACT_BYTES, build_manifest, plan_sha256
from macreduce.cluster import Coordinator, make_server
from macreduce.model import ModelConfig, decode_weights
from macreduce.observe import Events
from macreduce.worker import Client


@pytest.fixture
def coordinator(tmp_path):
    coordinator = Coordinator(tmp_path / "run", ["a", "b"], rounds=1, steps=1)
    coordinator.chunk_size = 1024
    for name, batch in (("a", 2), ("b", 4)):
        coordinator.register(
            name, {"batch_size": batch, "backend": "torch-cpu", "data_sha256": "test"}
        )
    return coordinator


def _serve(coordinator, tmp_path):
    secret = "t" * 32
    server = make_server(coordinator, "127.0.0.1", 0, secret)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = Client(
        f"http://127.0.0.1:{server.server_port}", secret, Events(tmp_path / "events", "test")
    )
    return server, thread, client


def _push(client, coordinator, node, raw, tokens):
    return client.push_artifact(
        raw,
        node=node,
        version=0,
        tokens=tokens,
        run_id=coordinator.run_id,
        parameter_names=list(coordinator.weights),
        plan=plan_sha256(coordinator.model),
        chunk_size=1024,
    )


def test_chunked_transfer_happy_path(coordinator, tmp_path):
    server, thread, client = _serve(coordinator, tmp_path)
    try:
        manifest = client.request(
            "/artifact/manifest", kind="checkpoint", version=0, run_id=coordinator.run_id
        )
        assert manifest["length"] == len(coordinator.raw)
        assert len(manifest["chunk_hashes"]) > 1
        fetched = client.fetch_artifact(
            0, coordinator.run_id, hashlib.sha256(coordinator.raw).hexdigest()
        )
        original = coordinator.raw
        assert fetched == original
        decoded = decode_weights(fetched, config=coordinator.model)
        assert set(decoded) == set(coordinator.weights)
        assert _push(client, coordinator, "a", original, 64)["accepted"]
        assert coordinator.version == 0 and set(coordinator.pending) == {"a"}
        assert _push(client, coordinator, "b", original, 128)["accepted"]
        assert coordinator.version == 1 and coordinator.pending == {}
        state = json.loads((coordinator.root / "state.json").read_text())
        assert state["schema"] == 2
        assert state["config"]["model"]["layers"] == 1
        assert state["config"]["model"]["tokenizer"] == "utf-8-bytes"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_corrupted_chunk_is_rejected(coordinator, tmp_path):
    server, thread, client = _serve(coordinator, tmp_path)
    try:
        manifest = build_manifest(
            coordinator.raw,
            run_id=coordinator.run_id,
            version=0,
            role="update",
            parameter_names=list(coordinator.weights),
            plan=plan_sha256(coordinator.model),
            node="a",
            chunk_size=1024,
        )
        started = client.request(
            "/artifact/update",
            manifest,
            node="a",
            version=0,
            tokens=64,
            run_id=coordinator.run_id,
        )
        chunk = bytearray(coordinator.raw[:1024])
        chunk[0] ^= 0xFF
        with pytest.raises(urllib.error.HTTPError) as error:
            client.request("/artifact/chunk", bytes(chunk), upload_id=started["upload_id"], index=0)
        assert error.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as error:
            client.request("/artifact/complete", b"", upload_id=started["upload_id"])
        assert error.value.code == 400
        assert coordinator.pending == {} and coordinator.version == 0
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_duplicate_chunk_retry_does_not_double_count(coordinator, tmp_path):
    server, thread, client = _serve(coordinator, tmp_path)
    try:
        original = bytes(coordinator.raw)
        first = _push(client, coordinator, "a", original, 64)
        second = _push(client, coordinator, "a", original, 64)
        assert first["accepted"] and not first["duplicate"]
        assert second["duplicate"]
        assert coordinator.version == 0 and list(coordinator.pending) == ["a"]
        _push(client, coordinator, "b", original, 128)
        assert coordinator.version == 1
        retry = _push(client, coordinator, "a", original, 64)
        assert retry["duplicate"] and coordinator.version == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_oversize_and_partial_updates_are_rejected(coordinator):
    manifest = build_manifest(
        coordinator.raw,
        run_id=coordinator.run_id,
        version=0,
        role="update",
        parameter_names=list(coordinator.weights),
        plan=plan_sha256(coordinator.model),
        node="a",
        chunk_size=1024,
    )
    manifest["length"] = MAX_ARTIFACT_BYTES + 1
    with pytest.raises(ValueError, match="size cap"):
        coordinator.begin_update("a", 0, 64, coordinator.run_id, manifest)
    assert not (coordinator.root / "incoming").exists()
    started = coordinator.begin_update(
        "a",
        0,
        64,
        coordinator.run_id,
        build_manifest(
            coordinator.raw,
            run_id=coordinator.run_id,
            version=0,
            role="update",
            parameter_names=list(coordinator.weights),
            plan=plan_sha256(coordinator.model),
            node="a",
            chunk_size=1024,
        ),
    )
    with pytest.raises(ValueError, match="incomplete"):
        coordinator.complete_update(started["upload_id"])
    assert coordinator.pending == {} and coordinator.version == 0


def test_custom_context_and_two_layer_config_travel_with_checkpoint(tmp_path):
    config = ModelConfig(layers=2, context=16)
    coordinator = Coordinator(tmp_path / "run", ["a"], rounds=1, steps=1, model_config=config)
    coordinator.register("a", {"batch_size": 2, "backend": "torch-cpu", "data_sha256": "test"})
    with pytest.raises(ValueError, match="token"):
        coordinator.submit("a", 0, 64, coordinator.raw, coordinator.run_id)
    assert coordinator.pending == {}
    loaded = decode_weights(coordinator.raw, config=config)
    assert set(loaded) == set(coordinator.weights)
    state = json.loads((coordinator.root / "state.json").read_text())
    assert state["config"]["model"] == config.to_dict()
