"""Tiny ANE linear forward, backward, SGD and stability check against NumPy."""

import argparse
import json
from pathlib import Path

import aneforge as af
import numpy as np


def run():
    rng = np.random.default_rng(0)
    x = af.input((8, 16))
    target = af.input((8, 1))
    initial = (rng.standard_normal((16, 1)) * 0.1).astype(np.float32)
    weight = af.parameter(initial.copy())
    residual = x @ weight - target
    loss = (residual * residual).mean((0, 1))
    trainer = af.Trainer(loss, [weight], lr=0.01, optimizer="sgd", device_optimizer=True)
    masks = [int(trainer._fwd._prog._device_mask),
             *[int(net._prog._device_mask) for net in trainer._bwd],
             *[int(net._prog._device_mask) for net in trainer._upd]]
    if any(mask != 4 for mask in masks):
        raise RuntimeError(f"programs off the ANE: {masks}")
    cases = []
    try:
        for scale in (0.0, 0.01, 1.0, 4.0):
            features = (rng.standard_normal((8, 16)) * scale).astype(np.float32)
            labels = (rng.standard_normal((8, 1)) * scale * 0.1).astype(np.float32)
            weight.attrs["value"] = initial.copy()
            trainer.data[x], trainer.data[target] = features, labels
            prediction = features.astype(np.float64) @ initial.astype(np.float64)
            error = prediction - labels
            expected_loss = float(np.mean(error**2))
            expected_grad = 2 * features.astype(np.float64).T @ error / len(features)
            actual_loss = trainer.loss()
            net = trainer._bwd[0]
            actual_grad = np.asarray(net(*trainer._feed(net))).reshape(initial.shape)
            np.testing.assert_allclose(actual_loss, expected_loss, rtol=0.02, atol=1e-3)
            np.testing.assert_allclose(actual_grad, expected_grad, rtol=0.02, atol=1e-3)
            trainer.step()
            actual_weight = np.asarray(weight.attrs["value"])
            np.testing.assert_allclose(actual_weight, initial - 0.01 * expected_grad,
                                       rtol=0.02, atol=1e-3)
            epsilon = 1e-5
            plus, minus = initial.astype(np.float64), initial.astype(np.float64)
            plus[0, 0] += epsilon
            minus[0, 0] -= epsilon
            fd = (np.mean((features @ plus - labels)**2)
                  - np.mean((features @ minus - labels)**2)) / (2 * epsilon)
            np.testing.assert_allclose(fd, expected_grad[0, 0], rtol=1e-6, atol=1e-8)
            cases.append({
                "input_scale": scale, "loss_ane": actual_loss, "loss_reference": expected_loss,
                "gradient_max_abs_error": float(np.max(np.abs(actual_grad - expected_grad))),
                "sgd_max_abs_error": float(np.max(np.abs(actual_weight - initial
                                                         + 0.01 * expected_grad))),
                "reference_finite_difference_error": float(abs(fd - expected_grad[0, 0])),
            })
        weight.attrs["value"] = initial.copy()
        features = (rng.standard_normal((8, 16)) * 0.25).astype(np.float32)
        labels = features @ np.linspace(-0.2, 0.2, 16).reshape(16, 1).astype(np.float32)
        trainer.data[x], trainer.data[target] = features, labels
        losses = [trainer.loss()]
        for _ in range(100):
            trainer.step()
            value = trainer.loss()
            if not np.isfinite(value) or not np.isfinite(weight.attrs["value"]).all():
                raise RuntimeError("ANE stability check became nonfinite")
            losses.append(value)
        if losses[-1] >= losses[0]:
            raise RuntimeError("fixed-batch SGD loss did not decrease")
        return {"status": "ok", "program_masks": masks, "seed": 0,
                "shape": {"batch": 8, "width": 16}, "cases": cases,
                "comparison_rtol": 0.02, "comparison_atol": 0.001,
                "stability_losses": losses,
                "scope": "linear float16 ANE vs float64 reference; not Qwen3 or attention"}
    finally:
        trainer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run()
    with args.output.open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(report["status"], report["stability_losses"][0], report["stability_losses"][-1])
