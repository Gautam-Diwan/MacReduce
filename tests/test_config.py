from pathlib import Path

import yaml


def test_qlora_config_points_at_the_curve():
    config = yaml.safe_load(Path("configs/qlora_m4.yaml").read_text())
    assert config["training"]["batch_size"] == "auto"
    assert config["training"]["grad_checkpoint"] == "auto"
    assert config["training"]["epochs"] == 3
    assert config["training"]["grad_accumulation_steps"] == 1
    assert config["data"]["train_examples"] == 4096
    assert config["model"]["max_seq_length"] == 512
    assert config["model"]["lora_rank"] == 8
    assert config["infrastructure"]["memory_limit_bytes"] == "auto"
    assert config["checkpoint"]["save_every"] == "epoch"
    assert config["placement"]["mode"] == "gpu_ane"
    assert config["model"]["lora_scale"] == 2.0
    assert config["model"]["lora_dropout"] == 0.0
    assert config["training"]["learning_rate"] == 1e-4
    assert config["training"]["weight_decay"] == 0.01
    assert config["training"]["max_grad_norm"] == 1.0
    assert config["training"]["lr_schedule"]["name"] == "cosine_decay"
    assert config["training"]["lr_schedule"]["warmup"] == 300


def test_yaml_recipe_reaches_the_training_namespace():
    from macreduce.finetune import _namespace, recipe_from_config, resolve_schedule

    config = yaml.safe_load(Path("configs/qlora_m4.yaml").read_text())
    recipe = recipe_from_config(config)
    assert recipe == {
        "rank": 8,
        "scale": 2.0,
        "dropout": 0.0,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "lr_schedule": {"name": "cosine_decay", "warmup": 300, "warmup_init": 1e-7, "end": 1e-6},
    }
    args = _namespace(Path("data"), Path("adapters"), 1, 3, -1, 2, 512, 1e-4, 0, 1, -1, recipe=recipe)
    assert args.lora_parameters == {"rank": 8, "dropout": 0.0, "scale": 2.0}
    assert args.weight_decay == 0.01
    assert args.max_grad_norm == 1.0
    assert args.lr_schedule == recipe["lr_schedule"]
    schedule = resolve_schedule(args.lr_schedule, args.learning_rate, 6144)
    assert schedule == {
        "name": "cosine_decay",
        "arguments": [1e-4, 5844, 1e-6],
        "warmup": 300,
        "warmup_init": 1e-7,
    }
    assert resolve_schedule(None, 1e-4, 10) is None


def test_recipe_rejects_bad_values():
    import pytest

    from macreduce.finetune import resolve_schedule, validate_recipe

    with pytest.raises(ValueError):
        validate_recipe({"scale": 0})
    with pytest.raises(ValueError):
        validate_recipe({"max_grad_norm": -1})
    with pytest.raises(ValueError):
        resolve_schedule({"name": "cosine_decay", "warmup": 300}, 1e-4, 300)
    assert validate_recipe({"max_grad_norm": None})["max_grad_norm"] is None


def test_schedule_hits_warmup_peak_and_end():
    import os

    import pytest

    if os.environ.get("MACREDUCE_TEST_MLX") != "1":
        pytest.skip("MLX test is opt-in")
    from mlx_lm.tuner.utils import build_schedule

    from macreduce.finetune import DEFAULT_RECIPE, resolve_schedule

    total = 6144
    schedule = build_schedule(resolve_schedule(DEFAULT_RECIPE["lr_schedule"], 1e-4, total))
    assert float(schedule(0)) == pytest.approx(1e-7, rel=1e-3)
    assert float(schedule(300)) == pytest.approx(1e-4, rel=1e-3)
    assert float(schedule(301)) == pytest.approx(1e-4, rel=1e-3)
    assert float(schedule(total - 1)) == pytest.approx(1e-6, rel=5e-2)
    assert float(schedule(total + 10)) == pytest.approx(1e-6, rel=1e-3)
