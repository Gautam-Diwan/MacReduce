import os
import sys
from datetime import datetime

import pytest

from macreduce.batch_fit import select_trial, steady_tokens_per_second
from macreduce.curve import allocate_run, bind_run_log
from macreduce.train_loop import resolve_save_every


def test_each_launch_gets_its_own_directory(tmp_path):
    when = datetime(2026, 10, 28, 20, 3, 0)
    first = allocate_run(tmp_path, now=when)
    second = allocate_run(tmp_path, now=when)
    assert first.name == "2026-10-28T20-03-00"
    assert second.name == "2026-10-28T20-03-00-2"
    assert (first / "figures").exists() is False


def test_run_log_lives_in_the_run_directory(tmp_path):
    stdout, stderr = sys.stdout, sys.stderr
    run = allocate_run(tmp_path, now=datetime(2026, 10, 28, 20, 40, 0))
    try:
        path = bind_run_log(run)
        print("hello-run", flush=True)
        text = path.read_text()
    finally:
        sys.stdout, sys.stderr = stdout, stderr
    assert path.parent == run
    assert "hello-run" in text
    assert (tmp_path / "latest").read_text().strip() == str(run)


def test_save_every_epoch_splits_the_run():
    assert resolve_save_every("epoch", 300, 3) == 100
    assert resolve_save_every(None, 10, 3) == 3
    assert resolve_save_every(25, 300, 3) == 25


def test_steady_rate_ignores_the_compile_step():
    assert steady_tokens_per_second([100.0, 400.0, 500.0]) == 450.0
    assert steady_tokens_per_second([80.0]) == 80.0


def test_select_trial_prefers_tokens_per_second():
    best = select_trial(
        [
            {"status": "error", "batch_size": 128, "tokens_per_second": 1},
            {"status": "ok", "batch_size": 32, "tokens_per_second": 900.0},
            {"status": "ok", "batch_size": 64, "tokens_per_second": 700.0, "grad_checkpoint": True},
        ]
    )
    assert best["batch_size"] == 32


@pytest.mark.skipif(os.environ.get("MACREDUCE_TEST_MLX") != "1", reason="opt-in Metal hardware test")
def test_checkpoint_restores_adapter_and_adamw(tmp_path):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np
    from mlx.utils import tree_flatten

    from macreduce.train_loop import load_checkpoint, save_checkpoint

    mx.random.seed(0)
    model = nn.Linear(4, 3)
    opt = optim.AdamW(learning_rate=1e-2)
    data = mx.random.normal((8, 4))

    def loss_fn(module, sample):
        return (module(sample) ** 2).mean()

    value_and_grad = nn.value_and_grad(model, loss_fn)
    loss, grads = value_and_grad(model, data)
    opt.update(model, grads)
    mx.eval(model.parameters(), opt.state, loss)
    saved = {name: np.array(value) for name, value in tree_flatten(model.parameters())}
    step = int(np.array(opt.step))
    destination = save_checkpoint(tmp_path, 4, model, opt, {"train_loss": float(loss), "seed": 0})

    blank = nn.Linear(4, 3)
    fresh = optim.AdamW(learning_rate=1e-2)
    meta = load_checkpoint(destination, blank, fresh)
    mx.eval(blank.parameters(), fresh.state)
    restored = {name: np.array(value) for name, value in tree_flatten(blank.parameters())}
    for name, value in saved.items():
        np.testing.assert_allclose(restored[name], value)
    assert meta["step"] == 4
    assert int(np.array(fresh.step)) == step


def test_resume_identity_checks_the_recipe():
    import pytest

    from macreduce.train_loop import _check_resume

    identity = {
        "seed": 0,
        "batch_size": 2,
        "max_seq_length": 512,
        "grad_checkpoint": False,
        "learning_rate": 1e-4,
        "lora_rank": 8,
        "lora_scale": 2.0,
        "lora_dropout": 0.0,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "lr_schedule": {"name": "cosine_decay", "arguments": [1e-4, 5844, 1e-6], "warmup": 300, "warmup_init": 1e-7},
    }
    _check_resume(dict(identity), identity)
    _check_resume({"seed": 0, "learning_rate": 1e-4}, identity)
    with pytest.raises(ValueError):
        _check_resume({**identity, "lora_scale": 20.0}, identity)
    with pytest.raises(ValueError):
        _check_resume({**identity, "max_grad_norm": None}, identity)
    with pytest.raises(ValueError):
        _check_resume({**identity, "lr_schedule": None}, identity)
