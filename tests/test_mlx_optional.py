"""Run explicitly outside a restricted sandbox: MACREDUCE_TEST_MLX=1 pytest."""

import os

import numpy as np
import pytest

from macreduce.model import MODEL, MLXBackend, TorchBackend, batch, corpus, initial_weights


@pytest.mark.skipif(os.environ.get("MACREDUCE_TEST_MLX") != "1", reason="opt-in Metal hardware test")
def test_mlx_matches_torch_forward_and_sgd():
    p = initial_weights()
    mlx, torch = MLXBackend(MODEL, p), TorchBackend(MODEL, p)
    x, y = batch(corpus()[0], np.random.default_rng(3), 2, 32)
    np.testing.assert_allclose(mlx.evaluate(x, y), torch.evaluate(x, y), atol=2e-5)
    mlx.step(x, y, 0.01)
    torch.step(x, y, 0.01)
    for k, expected in torch.export().items():
        np.testing.assert_allclose(mlx.export()[k], expected, atol=3e-5, rtol=3e-4)
