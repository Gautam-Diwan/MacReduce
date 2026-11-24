"""Real Flower loopback transport with an injected CPU numerical client.

These tests never impersonate GPU/ANE execution. Physical networks and MLX
are validated separately; the mode parametrization tests cohort orchestration.
"""

import io
import json
import socket
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from macreduce.federation_backend import ResidentLoraClient, collate, sample_indices
from macreduce.federation_plan import bind_address, fingerprint, load_plan, shard_indices
from macreduce.federation_state import RunState, pack_adapter, unpack_adapter
from macreduce.lora_client import BackendUnavailable

flwr = pytest.importorskip("flwr", reason="install the federation extra to test Flower transport")
from flwr.common import FitIns, GetPropertiesIns, Parameters, ndarrays_to_parameters  # noqa: E402

from macreduce.flower_federation import (  # noqa: E402
    FixedCohort,
    PersistentClient,
    client_session,
    decode_parameters,
    serve,
    signed,
    verified,
)

SECRET = "a-test-secret-with-at-least-24-characters"
DATA_HASH = fingerprint(["fixed test corpus"])


@pytest.fixture(autouse=True)
def no_upstream_analytics(monkeypatch):
    import flwr.supercore.telemetry as telemetry

    monkeypatch.setattr(telemetry, "FLWR_TELEMETRY_ENABLED", "0")


def plan(mode="gpu", rounds=2):
    result = load_plan(Path("configs/federation.yaml"), mode)
    result.update(rounds=rounds, local_steps=2, round_timeout=10)
    return result


class NumericalClient:
    """Durable deterministic SGD oracle, injected only by tests."""

    def __init__(self, increment):
        self.value = np.zeros(2, np.float32)
        self.increment = increment
        self.cursor = 0
        self.calls = 0

    def snapshot(self):
        return {"adapter": self.value.copy()}

    def train_round(self, start):
        self.calls += 1
        self.cursor += 2
        self.value = start["adapter"] + 2 * self.increment
        return self.snapshot(), {"tokens": 4, "cursor": self.cursor, "loss": 1.0}

    def save(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "oracle.json").write_text(json.dumps({"cursor": self.cursor,
                                                      "value": self.value.tolist()}))

    def restore(self, path):
        data = json.loads((path / "oracle.json").read_text())
        self.cursor = data["cursor"]
        self.value = np.array(data["value"], np.float32)


def clients_for(tmp_path, configuration):
    return [PersistentClient(NumericalClient(i + 1), configuration, spec["id"], DATA_HASH,
                             tmp_path / spec["id"], SECRET)
            for i, spec in enumerate(configuration["clients"])]


def jobs_for(state, clients):
    strategy = FixedCohort(state, SECRET)
    proxies = {str(i): SimpleNamespace(
        cid=str(i), get_properties=lambda ins, timeout, group_id, c=client: c.get_properties(ins)
    ) for i, client in enumerate(clients)}
    manager = SimpleNamespace(wait_for=lambda *args, **kwargs: True, all=lambda: proxies)
    return strategy, strategy.configure_fit(1, ndarrays_to_parameters(state.arrays), manager)


def test_modes_configurable_batches_and_disjoint_shards():
    for mode, count in [("gpu", 3), ("gpu_ane", 3), ("split", 6)]:
        config = plan(mode)
        assert len(config["clients"]) == count
        assert sum(row["weight"] for row in config["clients"]) == pytest.approx(1)
        shards = [shard_indices(config, row["id"], 31) for row in config["clients"]]
        assert sorted(sum(shards, [])) == list(range(31))
    assert plan()["clients"][-1]["grad_checkpoint"] is True
    assert plan()["coordinator"] == "m4-air"


def test_coordinator_must_name_a_configured_machine(tmp_path):
    path = tmp_path / "federation.yaml"
    path.write_text(Path("configs/federation.yaml").read_text().replace(
        "coordinator: m4-air", "coordinator: missing-host"
    ))
    with pytest.raises(ValueError, match="coordinator must name"):
        load_plan(path)


def test_sampler_continues_without_resetting_at_round_boundaries():
    full = sample_indices(5, 0, 12, 3)
    assert full == sample_indices(5, 0, 4, 3) + sample_indices(5, 4, 8, 3)
    assert len(set(full[:5])) == len(set(full[5:10])) == 5
    tok = SimpleNamespace(encode=lambda text: [2] * len(text), eos_token_id=1)
    batch, lengths = collate([{"text": "abc"}, {"text": "abcdef"}], tok, 5)
    assert batch.shape == (2, 5)
    np.testing.assert_array_equal(lengths, [[0, 4], [0, 5]])
    assert batch[0, 3] == 1


def test_standalone_ane_fails_before_any_hardware_import():
    config = plan("split")
    spec = config["clients"][1]
    backend = ResidentLoraClient(config, spec, [{"text": "x"}] * 12)
    with pytest.raises(BackendUnavailable, match="standalone ANE"):
        backend.setup()


def test_signed_envelopes_and_safe_bind():
    raw = signed({"version": 1}, SECRET)
    assert verified(raw, SECRET) == {"version": 1}
    with pytest.raises(ValueError, match="authentication"):
        verified(raw, "wrong-token")
    assert bind_address("127.0.0.1") == "127.0.0.1"
    assert bind_address("100.64.0.1") == "100.64.0.1"
    for address in ["0.0.0.0", "8.8.8.8", "192.168.1.2"]:
        with pytest.raises(ValueError):
            bind_address(address)


def test_float32_and_header_validation_before_array_allocation():
    start = {"adapter": np.zeros(2, np.float32)}
    raw = pack_adapter(start)
    assert unpack_adapter(raw)["adapter"].dtype == np.float32
    with pytest.raises(ValueError, match="nonfinite"):
        pack_adapter({"adapter": np.array([np.nan])})
    parameters = ndarrays_to_parameters([np.zeros(2, np.float64)])
    with pytest.raises(ValueError, match="dtype/shape"):
        decode_parameters(parameters, ["adapter"], {"adapter": [2]})
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, {
        "descr": "<f4", "fortran_order": False, "shape": (10**12,),
    })
    with pytest.raises(ValueError, match="dtype/shape"):
        decode_parameters(Parameters([stream.getvalue()], "numpy.ndarray"),
                          ["adapter"], {"adapter": [2]})


def test_cached_result_lost_ack_and_optimizer_resume(tmp_path):
    config = plan()
    state = RunState(tmp_path / "server", config, DATA_HASH, {"adapter": np.zeros(2, np.float32)})
    clients = clients_for(tmp_path, config)
    strategy, jobs = jobs_for(state, clients)
    results = [(proxy, clients[int(proxy.cid)].fit(ins)) for proxy, ins in jobs]
    # Same logical round, new RPC nonce: use exact prepared delta and saved trainer state.
    proxy, ins = jobs[0]
    request = verified(ins.config["envelope"], SECRET)
    retry = FitIns(ins.parameters, {"envelope": signed({**request, "nonce": "retry"}, SECRET)})
    again = clients[0].fit(retry)
    assert again.parameters.tensors == results[0][1].parameters.tensors
    assert clients[0].backend.calls == 1
    strategy.aggregate_fit(1, results, [])
    assert state.version == 1
    np.testing.assert_array_equal(state.adapter["adapter"], [4, 4])
    # Simulate all processes restarting after commit but before clients learn the result.
    resumed = RunState(tmp_path / "server", config, DATA_HASH, resume=True)
    restored = clients_for(tmp_path, config)
    assert restored[0].backend.cursor == 2
    strategy, jobs = jobs_for(resumed, restored)
    results = [(proxy, restored[int(proxy.cid)].fit(ins)) for proxy, ins in jobs]
    strategy.aggregate_fit(1, results, [])
    assert resumed.version == 2
    np.testing.assert_array_equal(resumed.adapter["adapter"], [8, 8])
    assert restored[0].backend.cursor == 4
    assert SECRET not in (tmp_path / "server" / "state.json").read_text()


def test_incomplete_cohort_and_changed_identity_never_commit(tmp_path):
    config = plan()
    state = RunState(tmp_path / "server", config, DATA_HASH, {"adapter": np.zeros(2, np.float32)})
    clients = clients_for(tmp_path, config)
    strategy, jobs = jobs_for(state, clients)
    results = [(proxy, clients[int(proxy.cid)].fit(ins)) for proxy, ins in jobs[:2]]
    with pytest.raises(RuntimeError, match="incomplete"):
        strategy.aggregate_fit(1, results, [])
    assert state.version == 0
    with pytest.raises(ValueError, match="differs"):
        RunState(tmp_path / "server", config, "different-data", resume=True)
    body = strategy._body("hello")
    with pytest.raises(ValueError, match="run changed"):
        clients[0].get_properties(GetPropertiesIns({"envelope": signed(
            {**body, "run_id": "different"}, SECRET)}))


def test_commit_failure_replays_exact_prepared_results(tmp_path, monkeypatch):
    import macreduce.federation_state as persistence

    config = plan(rounds=1)
    state = RunState(tmp_path / "server", config, DATA_HASH, {"adapter": np.zeros(2, np.float32)})
    clients = clients_for(tmp_path, config)
    strategy, jobs = jobs_for(state, clients)
    results = [(proxy, clients[int(proxy.cid)].fit(ins)) for proxy, ins in jobs]
    real_write = persistence.atomic_write

    def fail_pointer(path, raw):
        if path.name == "state.json":
            raise OSError("injected disk failure")
        real_write(path, raw)

    with monkeypatch.context() as patch:
        patch.setattr(persistence, "atomic_write", fail_pointer)
        with pytest.raises(OSError):
            strategy.aggregate_fit(1, results, [])
    assert state.version == 0
    restored = RunState(tmp_path / "server", config, DATA_HASH, resume=True)
    strategy, jobs = jobs_for(restored, clients)
    results = [(proxy, clients[int(proxy.cid)].fit(ins)) for proxy, ins in jobs]
    strategy.aggregate_fit(1, results, [])
    assert restored.version == 1
    assert [client.backend.calls for client in clients] == [1, 1, 1]
    np.testing.assert_array_equal(restored.adapter["adapter"], [4, 4])


@pytest.mark.parametrize("mode", ["gpu", "gpu_ane", "split"])
def test_actual_flower_transport_concurrent_clients(tmp_path, mode):
    config = plan(mode, rounds=2)
    state = RunState(tmp_path / "server", config, DATA_HASH, {"adapter": np.zeros(2, np.float32)})
    clients = clients_for(tmp_path, config)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    address = f"127.0.0.1:{port}"
    errors = []

    def server_run():
        try:
            serve(state, address, SECRET)
        except BaseException as exc:
            errors.append(exc)

    def worker_run(client):
        try:
            client_session(client, address)
        except BaseException as exc:
            errors.append(exc)

    server_thread = threading.Thread(target=server_run, daemon=True)
    server_thread.start()
    for _ in range(100):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=.1):
                break
        except OSError:
            time.sleep(.02)
    threads = [threading.Thread(target=worker_run, args=(client,), daemon=True) for client in clients]
    for thread in threads:
        thread.start()
    server_thread.join(timeout=20)
    for thread in threads:
        thread.join(timeout=5)
    assert not server_thread.is_alive()
    assert not any(thread.is_alive() for thread in threads)
    assert not errors, errors
    assert state.version == 2
    expected = 4 * np.mean(np.arange(1, len(clients) + 1))
    np.testing.assert_allclose(state.adapter["adapter"], [expected, expected])
    assert all(client.backend.calls == 2 for client in clients)


def test_failed_local_round_restores_previous_trainer_state(tmp_path, monkeypatch):
    config = plan()
    state = RunState(tmp_path / "server", config, DATA_HASH, {"adapter": np.zeros(2, np.float32)})
    clients = clients_for(tmp_path, config)
    _strategy, jobs = jobs_for(state, clients)
    proxy, ins = jobs[0]
    client = clients[int(proxy.cid)]
    good_train = client.backend.train_round

    def fail_after_some_work(start):
        good_train(start)
        raise RuntimeError("injected local failure")

    with monkeypatch.context() as patch:
        patch.setattr(client.backend, "train_round", fail_after_some_work)
        with pytest.raises(RuntimeError, match="injected"):
            client.fit(ins)
    assert client.backend.cursor == 0
    assert client.meta["prepared_version"] is None
    client.fit(ins)
    assert client.backend.cursor == 2


@pytest.mark.parametrize("device,microbatch,accumulation", [("gpu", 1, 1), ("gpu", 2, 2), ("gpu_ane", 1, 1)])
def test_resident_mlx_optimizer_and_broadcast_resume(
    tmp_path, monkeypatch, device, microbatch, accumulation
):
    import os

    if os.environ.get("MACREDUCE_TEST_MLX") != "1":
        pytest.skip("opt-in MLX/ANE hardware test; unavailable devices are not a pass")
    import mlx.core as mx
    import mlx_lm.utils
    from mlx_lm.models.qwen3 import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=1, intermediate_size=128,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=128, num_key_value_heads=2,
        max_position_embeddings=32, head_dim=16, tie_word_embeddings=True, rope_theta=10000.0,
    )
    tokenizer = SimpleNamespace(encode=lambda text: [2 + ord(c) % 100 for c in text], eos_token_id=1)
    monkeypatch.setattr(mlx_lm.utils, "load", lambda *a, **k: (Model(args), tokenizer))
    config = plan("gpu" if device == "gpu" else "gpu_ane")
    config.update(max_seq_length=16, local_steps=1)
    spec = config["clients"][0]
    spec.update(microbatch=microbatch, accumulation=accumulation)
    records = [{"text": "tiny deterministic text"}] * 12
    first = ResidentLoraClient(config, spec, records)
    initial = first.setup()
    local, metrics = first.train_round(initial)
    assert metrics["tokens"] > 0
    first.save(tmp_path / "checkpoint")
    broadcast = {key: value + np.float32(1e-5) for key, value in local.items()}
    after, _ = first.train_round(broadcast)
    second = ResidentLoraClient(config, spec, records)
    second.setup()
    second.restore(tmp_path / "checkpoint")
    restored, _ = second.train_round(broadcast)
    for key in after:
        np.testing.assert_allclose(after[key], restored[key], atol=2e-5, rtol=2e-4)
    assert first.cursor == second.cursor == 2 * microbatch * accumulation
    assert int(np.array(first.optimizer.step)) == int(np.array(second.optimizer.step)) == 2
    if device == "gpu_ane":
        assert second.placement()["backward_calls"] >= 2
        assert second.placement()["ane_mask"] == "0x4"
    mx.eval(first.model.parameters(), second.model.parameters())


def test_explicit_objective_weights_are_used_not_hardware_or_token_counts(tmp_path):
    config = plan(rounds=1)
    for spec, weight in zip(config["clients"], [.5, .3, .2], strict=True):
        spec["weight"] = weight
    state = RunState(tmp_path / "server", config, DATA_HASH, {"adapter": np.zeros(2, np.float32)})
    clients = clients_for(tmp_path, config)
    strategy, jobs = jobs_for(state, clients)
    results = [(proxy, clients[int(proxy.cid)].fit(ins)) for proxy, ins in jobs]
    strategy.aggregate_fit(1, results, [])
    np.testing.assert_allclose(state.adapter["adapter"], [3.4, 3.4])
