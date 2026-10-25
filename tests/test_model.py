import contextlib
import hashlib
import io
import json

import numpy as np
import pytest

from macreduce.model import (
    CONFIG_KEY,
    MODEL,
    ModelConfig,
    TorchBackend,
    batch,
    corpus,
    decode_weights,
    encode_weights,
    initial_weights,
    weighted_average,
)


def test_checkpoint_round_trip_and_no_pickle():
    p = initial_weights()
    for key, value in decode_weights(encode_weights(p), p).items():
        np.testing.assert_array_equal(value, p[key])
    bad = dict(p, q=np.array([object()], dtype=object))
    with pytest.raises(ValueError):
        decode_weights(encode_weights(bad), p)


@pytest.mark.parametrize("change", ["nan", "shape", "dtype", "missing", "oversize"])
def test_reject_corrupt_updates(change):
    p = initial_weights()
    bad = {k: v.copy() for k, v in p.items()}
    if change == "nan":
        bad["q"][0, 0] = np.nan
    elif change == "shape":
        bad["q"] = bad["q"][:1]
    elif change == "dtype":
        bad["q"] = bad["q"].astype(np.float64)
    elif change == "missing":
        del bad["q"]
    else:
        bad["q"] = np.zeros(10_000_000, dtype=np.float32)
    with pytest.raises(ValueError):
        decode_weights(encode_weights(bad), p)


def test_unequal_weights():
    p = {"x": np.array([1, 5], dtype=np.float32)}
    q = {"x": np.array([5, 9], dtype=np.float32)}
    np.testing.assert_allclose(weighted_average([(p, 1), (q, 3)])["x"], [4, 8])


def test_one_step_matches_pooled_sgd_with_unequal_batches():
    """Independent numerical oracle: K=1 local SGD equals pooled-gradient SGD."""
    data, _ = corpus()
    x, y = batch(data, np.random.default_rng(31), 6, 32)
    p = initial_weights()
    a, b, pooled = [TorchBackend(MODEL, p) for _ in range(3)]
    a.step(x[:2], y[:2], 0.01)
    b.step(x[2:], y[2:], 0.01)
    pooled.step(x, y, 0.01)
    averaged = weighted_average([(a.export(), 64), (b.export(), 128)])
    for k, target in pooled.export().items():
        np.testing.assert_allclose(averaged[k], target, atol=2e-6, rtol=2e-5)


def test_causal_attention():
    trainer = TorchBackend(MODEL, initial_weights())
    x = trainer.t.ones((1, 32), dtype=trainer.t.long)
    before = trainer.forward(x).detach().numpy()
    x[:, 16:] = 200
    after = trainer.forward(x).detach().numpy()
    np.testing.assert_allclose(before[:, :16], after[:, :16], atol=1e-6)


def test_model_config_validation():
    with pytest.raises(ValueError, match="divisible"):
        ModelConfig(width=30, heads=4)
    with pytest.raises(ValueError, match="float32"):
        ModelConfig(dtype="float16")
    with pytest.raises(ValueError, match="tokenizer"):
        ModelConfig(tokenizer="other")
    with pytest.raises(ValueError, match="disagree"):
        ModelConfig.from_mapping({"sequence": 16, "context": 32})
    restored = ModelConfig.from_mapping(MODEL)
    assert restored == ModelConfig()
    assert restored.to_dict()["context"] == 32 and restored.to_dict()["layers"] == 1


def test_default_config_preserves_original_parameters():
    weights = initial_weights()
    assert sum(value.size for value in weights.values()) == 29888
    digest = hashlib.sha256()
    for name, value in weights.items():
        digest.update(name.encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    assert digest.hexdigest() == "4ffc935be927e3ac8405301d1a28e5a9a1fabeec14bccef64d3cded1a5a38676"
    encoded = encode_weights(weights, ModelConfig())
    with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
        saved = ModelConfig.from_mapping(json.loads(archive[CONFIG_KEY].tobytes()))
    assert saved == ModelConfig()


def test_two_layer_checkpoint_round_trip_and_train():
    config = ModelConfig(layers=2)
    weights = initial_weights(config)
    assert sum(value.size for value in weights.values()) == 42304
    assert "b0_q" in weights and "b1_ff2" in weights and "q" not in weights
    restored = decode_weights(encode_weights(weights, config), config=config)
    for name, value in weights.items():
        np.testing.assert_array_equal(restored[name], value)
    with pytest.raises(ValueError, match="config mismatch"):
        decode_weights(encode_weights(weights, config), config=ModelConfig(layers=1))
    data, _digest = corpus()
    tokens, target = batch(data, np.random.default_rng(0), 2, config.context)
    trainer = TorchBackend(config, weights)
    before = trainer.export()
    loss = trainer.step(tokens, target, 0.05)
    assert np.isfinite(loss)
    after = trainer.export()
    assert any(not np.array_equal(before[name], after[name]) for name in before)
    inputs = trainer.t.ones((1, config.context), dtype=trainer.t.long)
    first = trainer.forward(inputs).detach().numpy()
    inputs[:, config.context // 2 :] = 200
    second = trainer.forward(inputs).detach().numpy()
    np.testing.assert_allclose(
        first[:, : config.context // 2], second[:, : config.context // 2], atol=1e-6
    )


def test_zip_duplicate_rejected():
    import zipfile

    p = initial_weights()
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for _ in p:
            with pytest.warns(UserWarning) if len(z.namelist()) else contextlib.nullcontext():
                z.writestr("q.npy", b"bad")
    with pytest.raises(ValueError):
        decode_weights(buf.getvalue(), p)
