import copy
import json
import runpy

import pytest
import yaml

SUITE = runpy.run_path("tools/run_baselines.py")


def test_proposal_recipe_is_not_silently_replaced_by_the_full_m1_recipe():
    proposal = SUITE["build_plan"]("proposal")
    full = SUITE["build_plan"]("full")
    p = proposal["arms"][0]["config"]
    f = full["arms"][0]["config"]
    assert (p["data"]["train_examples"], p["training"]["batch_size"],
            p["training"]["grad_accumulation_steps"]) == (4096, 8, 2)
    assert (f["data"]["train_examples"], f["training"]["batch_size"],
            f["training"]["grad_accumulation_steps"]) == (None, 1, 1)
    assert p["training"]["epochs"] == f["training"]["epochs"] == 3
    assert all(row["status"] == "blocked" for row in full["proposal_coverage"])


def test_comparison_scope_matches_the_requested_4096_record_recipe():
    plan = SUITE["build_plan"]("comparison")
    config = plan["arms"][0]["config"]
    assert config["data"]["train_examples"] == 4096
    assert config["training"]["batch_size"] == 1
    assert config["training"]["grad_accumulation_steps"] == 1
    assert config["training"]["grad_checkpoint"] is False
    assert config["training"]["epochs"] == 3
    assert config["checkpoint"]["save_every"] == 500
    assert config["checkpoint"]["resume"] is None
    assert config["placement"]["mode"] == "gpu"


def test_existing_run_must_be_complete_and_match_the_requested_recipe(tmp_path):
    arm = SUITE["build_plan"]("full")["arms"][0]
    config = copy.deepcopy(arm["config"])
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    result = {"status": "ok", "iters": 71301, "train_examples": 23767,
              "baseline_test": {"perplexity": 10}, "adapted_test": {"perplexity": 9}}
    (tmp_path / "result.json").write_text(json.dumps(result))
    assert SUITE["import_existing"](arm, tmp_path)["imported"] is True
    result["status"] = "error"
    (tmp_path / "result.json").write_text(json.dumps(result))
    with pytest.raises(ValueError, match="completed"):
        SUITE["import_existing"](arm, tmp_path)
    result["status"] = "ok"
    (tmp_path / "result.json").write_text(json.dumps(result))
    config["training"]["grad_checkpoint"] = True
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match="recipe"):
        SUITE["import_existing"](arm, tmp_path)


def test_blocked_results_stay_explicit_in_json_and_csv(tmp_path):
    summary = {"scope": "full", "results": [
        {"name": "local_gpu_ane", "status": "blocked", "reason": "failed gradient gate"}
    ]}
    SUITE["write_summary"](tmp_path, summary)
    assert json.loads((tmp_path / "summary.json").read_text()) == summary
    assert "blocked" in (tmp_path / "summary.csv").read_text()
    assert not (tmp_path / "summary.tmp").exists()


def test_runner_refuses_to_overlap_an_active_training_job(monkeypatch, tmp_path):
    execute = SUITE["execute"]
    monkeypatch.setitem(execute.__globals__, "active_supervisors", lambda: [123])
    output = tmp_path / "must-not-start"
    with pytest.raises(RuntimeError, match="active"):
        execute(SUITE["build_plan"]("full"), output, 30)
    assert not output.exists()


def test_cli_swap_override_reaches_all_arm_configs_without_changing_recipe():
    plan = SUITE["build_plan"]("comparison")
    recipes = [copy.deepcopy(row["config"]["training"]) for row in plan["arms"]]
    SUITE["set_swap_override"](plan, 1024)
    assert plan["max_swap_growth_override_mib"] == 1024
    for arm, recipe in zip(plan["arms"], recipes):
        assert arm["config"]["safety"]["max_swap_growth_bytes"] == 1073741824
        assert arm["config"]["safety"]["min_free_disk_bytes"] == 10737418240
        assert arm["config"]["training"] == recipe
    with pytest.raises(ValueError):
        SUITE["set_swap_override"](plan, 0)


def test_disabling_swap_guard_retains_other_policies_and_recipe():
    plan = SUITE["build_plan"]("comparison")
    before = copy.deepcopy(plan)
    SUITE["set_swap_override"](plan, None, disable=True)
    for arm, original in zip(plan["arms"], before["arms"]):
        expected = original["config"]
        expected["safety"]["max_swap_growth_bytes"] = None
        assert arm["config"] == expected
    assert plan["swap_growth_guard_disabled"] is True
    with pytest.raises(ValueError):
        SUITE["set_swap_override"](plan, 1024, disable=True)


def test_resource_stop_preserves_partial_metrics_without_claiming_completion(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs/latest").write_text(str(run))
    (tmp_path / "job").mkdir()
    (tmp_path / "job/exit.json").write_text(json.dumps({"stop_reason": "swap guard"}))
    (run / "history.jsonl").write_text(json.dumps({
        "kind": "train", "iteration": 200, "train_loss": 3.4,
        "trained_tokens": 1000, "peak_memory": 2.1,
    }) + "\n")
    result = SUITE["result_row"]("local_gpu", tmp_path, 1)
    assert result["status"] == "stopped"
    assert result["last_reported_step"] == 200
    assert result["last_reported_train_loss"] == 3.4
    assert result["peak_mlx_bytes"] == 2100000000
    assert result["final_test_perplexity"] is None


def test_output_collision_keeps_old_files_and_allocates_distinct_directories(tmp_path):
    output = tmp_path / "experiment"
    output.mkdir()
    marker = output / "summary.json"
    marker.write_text("previous results")
    first = SUITE["allocate_suite"](output)
    second = SUITE["allocate_suite"](output)
    assert first.name == "experiment-2"
    assert second.name == "experiment-3"
    assert marker.read_text() == "previous results"
    assert first.is_dir() and second.is_dir()


def test_resume_checks_checkpoint_completeness_and_matching_recipe(tmp_path):
    config = SUITE["build_plan"]("comparison")["arms"][0]["config"]
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/0002000"
    checkpoint.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    for name in ("state.json", "adapters.safetensors", "optimizer.safetensors"):
        (checkpoint / name).write_text("test placeholder")
    assert SUITE["validate_resume"](config, checkpoint) == checkpoint.resolve()
    hybrid = copy.deepcopy(config)
    hybrid["placement"]["mode"] = "gpu_ane"
    with pytest.raises(ValueError, match="placement mode"):
        SUITE["validate_resume"](hybrid, checkpoint)
    changed = copy.deepcopy(config)
    changed["training"]["batch_size"] = 2
    with pytest.raises(ValueError, match="training recipe"):
        SUITE["validate_resume"](changed, checkpoint)
    (checkpoint / "optimizer.safetensors").unlink()
    with pytest.raises(ValueError, match="missing optimizer"):
        SUITE["validate_resume"](config, checkpoint)


def test_hybrid_resume_keeps_imported_gpu_and_applies_checkpoint(monkeypatch, tmp_path):
    execute = SUITE["execute"]
    plan = SUITE["build_plan"]("comparison")
    config = plan["arms"][1]["config"]
    source = tmp_path / "source"
    checkpoint = source / "checkpoints/0000500"
    checkpoint.mkdir(parents=True)
    (source / "config.yaml").write_text(yaml.safe_dump(config))
    for name in ("state.json", "adapters.safetensors", "optimizer.safetensors"):
        (checkpoint / name).write_text("test placeholder")
    config["safety"]["min_free_disk_bytes"] = 5 * 1024**3
    seen = []

    def inspect(config):
        seen.append(config)
        return {"status": "blocked", "blockers": ["test stops before hardware launch"]}

    monkeypatch.setitem(execute.__globals__, "active_supervisors", lambda: [])
    monkeypatch.setitem(execute.__globals__, "check_config", inspect)
    monkeypatch.setitem(execute.__globals__, "import_existing",
                        lambda arm, path: {"name": arm["name"], "status": "completed", "imported": True})
    result = execute(plan, tmp_path / "suite", 30, existing_gpu=tmp_path / "gpu",
                     resume_gpu_ane=checkpoint)
    assert result["results"][0]["imported"] is True
    assert len(seen) == 1
    assert seen[0]["placement"]["mode"] == "gpu_ane"
    assert seen[0]["checkpoint"]["resume"] == str(checkpoint.resolve())
    assert seen[0]["safety"]["min_free_disk_bytes"] == 5 * 1024**3
