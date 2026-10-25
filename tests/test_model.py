import contextlib
import io

import numpy as np
import pytest

from macreduce.model import (
    MODEL,
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
