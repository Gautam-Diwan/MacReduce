import runpy
from pathlib import Path

import yaml

from macreduce.preflight import GIB, check_config


def test_full_baseline_does_not_inherit_smoke_step_or_checkpoint_restrictions():
    config = yaml.safe_load(Path("configs/M1_8GB/gpu_full_baseline.yaml").read_text())
    host = {
        "system": "Darwin", "architecture": "arm64", "macos": "27.2.1",
        "python": [3, 12, 0], "memory_total_bytes": 8 * GIB,
        "disk_free_bytes": 20 * GIB,
        "packages": {"mlx": "0.32.4", "mlx-lm": "0.32.2"}, "mactop": True,
    }
    assert check_config(config, host)["status"] == "ready"
    config["placement"]["mode"] = "gpu_ane"
    assert check_config(config, host)["status"] == "blocked"


def test_full_dataset_none_reaches_runner_with_checkpointing_off(monkeypatch, tmp_path):
    import macreduce.curve as curve

    config = yaml.safe_load(Path("configs/M1_8GB/gpu_full_baseline.yaml").read_text())
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(curve, "run_curve", capture)
    entry = runpy.run_path("main.py")
    entry["train"](config, tmp_path)
    assert seen["train_examples"] is None
    assert seen["grad_checkpoint"] is False
    assert seen["iters"] is None
    assert seen["epochs"] == 3
    assert seen["telemetry_interval_ms"] == 30000
    assert seen["test_batches"] == seen["val_batches"] == -1
