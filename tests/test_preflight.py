import copy
import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from macreduce.preflight import GIB, check_config


@pytest.fixture
def config():
    return yaml.safe_load(Path("configs/M1_8GB/gpu_smoke.yaml").read_text())


@pytest.fixture
def host():
    return {
        "system": "Darwin", "architecture": "arm64", "macos": "15.0",
        "python": [3, 12, 0], "memory_total_bytes": 8 * GIB,
        "disk_free_bytes": 20 * GIB,
        "packages": {"mlx": "0.32.4", "mlx-lm": "0.32.2", "aneforge": "0.4.0"},
        "mactop": True,
    }


def test_smoke_configs_change_only_placement_and_output(config, host):
    hybrid = yaml.safe_load(Path("configs/M1_8GB/gpu_ane_smoke.yaml").read_text())
    assert check_config(config, host)["status"] == "ready"
    assert check_config(hybrid, host)["status"] == "blocked"
    for key in ("placement", "logging"):
        hybrid.pop(key)
        config.pop(key)
    assert config == hybrid


def test_hybrid_rejects_a_failed_gradient_gate(config, host, tmp_path):
    config["placement"]["mode"] = "gpu_ane"
    report = tmp_path / "validation.json"
    config["placement"]["validation_report"] = str(report)
    for status in ("failed_gradient_gate", "error"):
        report.write_text(json.dumps({"status": status}))
        assert check_config(config, host)["status"] == "blocked"
    report.write_text(json.dumps({"status": "pass"}))
    assert check_config(config, host)["status"] == "blocked"
    report.write_text(json.dumps({"status": "pass", "sequence": config["model"]["max_seq_length"]}))
    assert check_config(config, host)["status"] == "ready"
    config["model"]["max_seq_length"] *= 2
    assert check_config(config, host)["status"] == "blocked"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("macos", "13.4.1", "macOS 14"),
        ("architecture", "x86_64", "arm64"),
        ("memory_total_bytes", None, "memory is unknown"),
        ("memory_total_bytes", 5 * GIB, "memory reserve"),
        ("disk_free_bytes", 9 * GIB, "free disk"),
        ("mactop", False, "mactop"),
        ("python", [3, 11, 9], "Python 3.12"),
    ],
)
def test_host_blockers_are_not_passes(config, host, field, value, message):
    host[field] = value
    result = check_config(config, host)
    assert result["status"] == "blocked"
    assert any(message in reason for reason in result["blockers"])


def test_missing_ane_does_not_fall_back_to_gpu(config, host):
    host["packages"]["aneforge"] = None
    assert check_config(config, host)["status"] == "ready"
    config["placement"]["mode"] = "gpu_ane"
    assert "aneforge==0.4.0" in " ".join(check_config(config, host)["blockers"])
    config["placement"]["mode"] = "ane"
    assert "not implemented" in " ".join(check_config(config, host)["blockers"])


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("training", "batch_size", "auto"),
        ("training", "grad_checkpoint", "auto"),
        ("training", "iters", None),
        ("training", "iters", 21),
        ("infrastructure", "memory_limit_bytes", "auto"),
    ],
)
def test_smoke_cannot_turn_into_unbounded_search(config, host, section, key, value):
    config[section][key] = value
    assert check_config(config, host)["status"] == "blocked"


def test_blocked_launch_does_not_create_output_or_import_accelerators(config, tmp_path):
    config = copy.deepcopy(config)
    config["placement"]["mode"] = "ane"
    output = tmp_path / "uncreated"
    config["logging"]["dir"] = str(output)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    env = {**os.environ, "PYTHONPATH": str(Path("src").resolve())}
    for extra in ([], ["--preflight"]):
        result = subprocess.run(
            [sys.executable, "main.py", "--config", str(path), *extra],
            capture_output=True, text=True, env=env, timeout=30,
        )
        assert result.returncode == 1, result.stderr
        assert json.loads(result.stdout)["status"] == "blocked"
        assert not output.exists()
        assert "mlx.core" not in result.stderr


def test_explicit_memory_budget_does_not_wire_the_recommended_working_set(monkeypatch):
    from macreduce.memory import apply_memory_limit

    calls = {}
    core = ModuleType("mlx.core")
    core.device_info = lambda: {"max_recommended_working_set_size": 6 * GIB}
    for name in ("set_wired_limit", "set_memory_limit", "set_cache_limit"):
        setattr(core, name, lambda value, name=name: calls.update({name: value}))
    core.clear_cache = lambda: None
    core.reset_peak_memory = lambda: None
    mlx = ModuleType("mlx")
    mlx.core = core
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)
    apply_memory_limit(2 * GIB)
    assert calls == {
        "set_wired_limit": 2 * GIB,
        "set_memory_limit": 2 * GIB,
        "set_cache_limit": GIB // 4,
    }
