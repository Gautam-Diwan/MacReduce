import os

import numpy as np
import pytest

from macreduce.ane_numerics import restore_lora_gradients, scale_lora_factors


def test_factor_rescaling_preserves_update_and_master_gradients():
    rng = np.random.default_rng(4)
    x, a, b, cot = [rng.normal(size=shape) for shape in [(3, 8), (8, 2), (2, 6), (3, 6)]]
    b *= 1e-6
    scaled_a, scaled_b = scale_lora_factors([a, b])
    np.testing.assert_allclose(x @ scaled_a @ scaled_b, x @ a @ b)
    ga = x.T @ cot @ scaled_b.T
    gb = scaled_a.T @ x.T @ cot
    original_ga, original_gb = restore_lora_gradients([ga, gb])
    np.testing.assert_allclose(original_ga, x.T @ cot @ b.T)
    np.testing.assert_allclose(original_gb, a.T @ x.T @ cot)


@pytest.mark.skipif(os.environ.get("MACREDUCE_TEST_MLX") != "1", reason="opt-in ANE hardware test")
def test_scaled_norm_forward_and_backward_across_activation_ranges():
    import aneforge as af
    from aneforge.autograd import backward_from

    from macreduce.ane_numerics import scaled_rms_norm
    from macreduce.ane_stack import _release, _require_ane

    rng = np.random.default_rng(5)
    data = rng.normal(size=(4, 1024)).astype(np.float32)
    data[0] = 0
    data[1] *= 0.01
    data[3, 0] = 6624
    gamma = rng.uniform(0.5, 2, 1024).astype(np.float32)
    cot = rng.normal(size=data.shape).astype(np.float32) * 0.1
    eps = 1e-6
    x = af.input(data.shape)
    forward = af.compile(scaled_rms_norm(x, gamma, eps))
    x2, dy = af.input(data.shape), af.input(data.shape)
    y2 = scaled_rms_norm(x2, gamma, eps)
    backward = af.compile(backward_from(dy, y2, [x2], stop=[x2])[x2])
    try:
        _require_ane(forward, "normalization forward")
        _require_ane(backward, "normalization backward")
        actual = np.asarray(forward(data))
        feed = [data if t is x2 else cot if t is dy
                else np.asarray(t.attrs["value"]).reshape(t.shape) for t in backward._input_tensors]
        grad = np.asarray(backward(*feed))
        val = data.astype(np.float64)
        inv = 1 / np.sqrt(np.mean(val**2, axis=-1, keepdims=True) + eps)
        normed = val * inv
        gn = cot * gamma
        expected_grad = (gn - normed * np.mean(gn * normed, axis=-1, keepdims=True)) * inv
        assert np.isfinite(actual).all() and np.isfinite(grad).all()
        np.testing.assert_allclose(actual, normed * gamma, rtol=0.03, atol=0.002)
        np.testing.assert_allclose(grad, expected_grad, rtol=0.03, atol=0.002)
    finally:
        _release(forward)
        _release(backward)
