import json
import urllib.error
import urllib.request

import numpy as np

from macreduce.federated import (
    AdapterRelay,
    LocalResult,
    PeerRequired,
    average,
    cohort,
    communication_rounds,
    dequantize_int8,
    fednova_average,
    iid_shards,
    merge_device_updates,
    pack_state,
    quantize_int8,
    reference_update,
    round_wire_bytes,
    run_federated,
    study,
    unpack_state,
)


def test_intra_mac_modes_repeat_on_every_mac():
    assert [client.device for client in cohort(1, "gpu")] == ["gpu"]
    assert [client.device for client in cohort(1, "gpu_ane")] == ["gpu_ane"]
    assert [client.device for client in cohort(1, "split")] == ["gpu", "ane"]
    split = cohort(3, "split")
    assert len(split) == 6
    assert {client.mac for client in split} == {0, 1, 2}
    assert all(client.device in {"gpu", "ane"} for client in split)


def test_iid_shards_are_disjoint_and_cover_the_corpus():
    shards = iid_shards(100, 4, seed=0)
    flat = np.concatenate(shards)
    assert len(flat) == 100
    assert len(set(flat.tolist())) == 100


def test_topologies_match_on_uncompressed_deltas():
    rng = np.random.default_rng(1)
    deltas = [rng.normal(size=8) for _ in range(3)]
    star = average(deltas, "star")
    assert np.allclose(average(deltas, "ring"), star)
    assert np.allclose(average(deltas, "tree"), star)


def test_fednova_matches_the_mean_when_every_client_takes_the_same_steps():
    deltas = [np.array([2.0, 4.0]), np.array([6.0, 8.0])]
    assert np.allclose(fednova_average(deltas, [3, 3]), np.array([4.0, 6.0]))
    unequal = fednova_average([np.array([4.0]), np.array([1.0])], [2, 1])
    assert not np.allclose(unequal, np.array([2.5]))


def test_error_feedback_keeps_the_mass_quantization_drops():
    delta = np.array([0.3, -0.7, 1.1])
    residual = np.zeros_like(delta)
    sent = np.zeros_like(delta)
    for _ in range(5):
        codes, scale, residual = quantize_int8(delta, residual)
        sent = sent + dequantize_int8(codes, scale)
    assert np.allclose(sent + residual, 5 * delta)


def test_one_mac_sends_nothing_and_fedavg_sends_fewer_rounds():
    assert round_wire_bytes(1, 100, "star", quantized=False) == 0
    full_step = round_wire_bytes(2, 100, "ring", quantized=False)
    assert round_wire_bytes(2, 100, "ring", quantized=True) < full_step
    rounds = communication_rounds(1536, 8)
    assert rounds["fedavg"] < rounds["allreduce"]
    body = study()
    assert body["project"] == "MacReduce"
    assert body["status"] == "implemented"
    assert body["payload"] == "lora_adapter_delta"
    assert body["base_weights_sent"] is False
    assert body["measured"] is False
    assert "serving placement" in body["future_scope"]


def test_fedavg_reaches_the_mean_and_allreduce_sends_more_rounds():
    records = [{"text": str(index)} for index in range(8)]
    initial = {"adapter": np.zeros(2)}
    targets = {
        "m0-gpu": np.array([2.0, 0.0]),
        "m0-ane": np.array([0.0, 4.0]),
    }

    def update(client, state, shard, steps):
        assert shard
        return reference_update(client, state, shard, steps, targets, lr=1.0)

    shared = dict(
        records=records,
        initial_state=initial,
        local_update=update,
        n_macs=1,
        mode="split",
        optimizer_steps=4,
        local_steps=4,
        trainer="reference",
    )
    fedavg = run_federated(**shared, algorithm="fedavg", topology="star")
    control = run_federated(**shared, algorithm="allreduce", topology="star")
    assert np.allclose(fedavg["state"]["adapter"], np.array([1.0, 2.0]))
    assert np.allclose(control["state"]["adapter"], fedavg["state"]["adapter"])
    assert fedavg["wire_bytes"] * 4 == control["wire_bytes"]
    assert fedavg["base_weights_sent"] is False
    assert fedavg["measured"] is False
    ring = run_federated(**shared, algorithm="fedavg", topology="ring")
    tree = run_federated(**shared, algorithm="fedavg", topology="tree")
    assert np.allclose(ring["state"]["adapter"], fedavg["state"]["adapter"])
    assert np.allclose(tree["state"]["adapter"], fedavg["state"]["adapter"])


def test_validation_is_recorded_on_the_requested_stride():
    records = [{"text": "only"}]

    def update(client, state, shard, steps):
        return reference_update(client, state, shard, steps, {client.name: np.ones(1)})

    result = run_federated(
        records,
        {"adapter": np.zeros(1)},
        update,
        n_macs=1,
        mode="gpu",
        algorithm="allreduce",
        topology="star",
        optimizer_steps=4,
        local_steps=1,
        eval_every=2,
        evaluate=lambda state: 1.5,
    )
    assert [row.get("validation") for row in result["history"]] == [None, 1.5, None, 1.5]


def test_quantized_rounds_move_fewer_bytes_and_fednova_rescales():
    records = [{"text": "a"}, {"text": "b"}]
    initial = {"adapter": np.zeros(32)}

    def update(client, state, shard, steps):
        del shard, steps
        jump = 4.0 if client.device == "gpu" else 1.0
        taken = 2 if client.device == "gpu" else 1
        return LocalResult({"adapter": state["adapter"] + jump}, taken, None)

    plain = run_federated(
        records,
        initial,
        update,
        n_macs=1,
        mode="split",
        algorithm="fedavg",
        topology="star",
        optimizer_steps=1,
        local_steps=1,
        normalize=True,
    )
    quantized = run_federated(
        records,
        initial,
        update,
        n_macs=1,
        mode="split",
        algorithm="fedavg",
        topology="star",
        optimizer_steps=1,
        local_steps=1,
        quantize=True,
        normalize=True,
    )
    assert np.allclose(plain["state"]["adapter"], np.full(32, 2.25))
    assert quantized["wire_bytes"] < plain["wire_bytes"]


def test_one_mac_sends_no_adapter_and_a_second_mac_needs_a_peer():
    records = [{"text": "only"}]
    initial = {"adapter": np.zeros(1)}

    def update(client, state, shard, steps):
        return reference_update(client, state, shard, steps, {client.name: np.ones(1)})

    alone = run_federated(
        records,
        initial,
        update,
        n_macs=1,
        mode="gpu",
        algorithm="fedavg",
        topology="star",
        optimizer_steps=1,
        local_steps=1,
    )
    assert alone["wire_bytes"] == 0
    try:
        run_federated(
            records,
            initial,
            update,
            n_macs=2,
            mode="gpu",
            algorithm="fedavg",
            topology="ring",
            optimizer_steps=1,
            local_steps=1,
        )
    except PeerRequired:
        return
    raise AssertionError("two Macs ran without a relay")


def test_combined_client_emits_one_normalized_delta():
    start = {"adapter": np.zeros(1)}
    gpu = LocalResult({"adapter": np.array([4.0])}, 2, 1.0)
    ane = LocalResult({"adapter": np.array([1.0])}, 1, 3.0)
    merged = merge_device_updates(start, [gpu, ane])
    assert merged.steps == 1
    assert np.allclose(merged.state["adapter"], np.array([2.25]))


def test_shards_are_disjoint_inside_the_round():
    records = [{"text": str(index)} for index in range(6)]
    seen = {}

    def update(client, state, shard, steps):
        seen[client.name] = [row["text"] for row in shard]
        return reference_update(
            client, state, shard, steps, {client.name: np.zeros(1) for client in cohort(1, "split")}
        )

    run_federated(
        records,
        {"adapter": np.zeros(1)},
        update,
        n_macs=1,
        mode="split",
        algorithm="fedavg",
        topology="star",
        optimizer_steps=1,
        local_steps=1,
    )
    left, right = seen.values()
    assert set(left).isdisjoint(right)
    assert set(left) | set(right) == {row["text"] for row in records}


def test_loopback_relay_accepts_the_cohort_and_rejects_a_bad_token():
    relay = AdapterRelay(("m0-gpu", "m0-ane"), token="secret")
    url = relay.start()
    try:
        denied = urllib.request.Request(
            url + "?client=m0-gpu",
            data=pack_state({"adapter": np.array([1.0])}),
            headers={"Authorization": "Bearer wrong"},
            method="POST",
        )
        try:
            urllib.request.urlopen(denied, timeout=2)
        except urllib.error.HTTPError as exc:
            assert exc.code == 401
        else:
            raise AssertionError("a bad token was accepted")
        _post(url, "m0-gpu", pack_state({"adapter": np.array([1.0])}))
        second = _post(url, "m0-ane", pack_state({"adapter": np.array([3.0])}))
        assert json.loads(second)["complete"] is True
        ready = urllib.request.Request(url, headers={"Authorization": "Bearer secret"})
        with urllib.request.urlopen(ready, timeout=2) as response:
            assert json.loads(response.read())["complete"] is True
        assert np.allclose(
            unpack_state(relay.payloads["m0-gpu"])["adapter"],
            np.array([1.0]),
        )
    finally:
        relay.close()


def _post(url: str, name: str, payload: bytes) -> bytes:
    request = urllib.request.Request(
        f"{url}?client={name}",
        data=payload,
        headers={"Authorization": "Bearer secret"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=2) as response:
        return response.read()
