"""Plan, run, and summarize sequential baselines without an assistant process.

Default invocation is read-only. --run explicitly starts the available arms.
The proposal recipe and the user's full-data M1 baseline are distinct scopes.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from macreduce.preflight import check_config  # noqa: E402


def build_plan(scope: str) -> dict:
    filename = "gpu_m4_comparison.yaml" if scope == "comparison" else "gpu_full_baseline.yaml"
    base = yaml.safe_load((ROOT / "configs/M1_8GB" / filename).read_text())
    if scope == "proposal":
        base["data"]["train_examples"] = 4096
        base["training"]["batch_size"] = 8
        base["training"]["grad_accumulation_steps"] = 2
    rows = []
    for mode in ("gpu", "gpu_ane"):
        config = copy.deepcopy(base)
        config["placement"] = {"mode": mode}
        if mode == "gpu_ane":
            config["placement"]["validation_report"] = (
                "evidence/m1-hybrid-validation-2026-11-10/boundary-parity-512.json"
            )
        rows.append({"name": f"local_{mode}", "kind": "local_training", "config": config})
    # These are proposal requirements, not aliases for the local curve runner.
    coverage = [
        {"name": "one_gpu_fedavg_star_uncompressed",
         "status": "blocked", "reason": "federated executor lacks M1 config/budget plumbing; local curve is not FedAvg"},
        {"name": "combined_gpu_ane_federated_client",
         "status": "blocked", "reason": "local hybrid validated on M1; federated hybrid executor still missing"},
        {"name": "separate_gpu_ane_clients",
         "status": "blocked", "reason": "independent full-Qwen3 ANE trainer is not implemented"},
        {"name": "per_step_allreduce_control",
         "status": "blocked", "reason": "requires configured federated clients; local curve does not measure all-reduce"},
        {"name": "ring_tree_int8_error_feedback_fednova_variants",
         "status": "blocked", "reason": "requires validated clients and an explicit unequal-step schedule for FedNova"},
        {"name": "multiple_physical_macs",
         "status": "blocked", "reason": "no second connected Mac or physical peer executor"},
    ]
    return {"scope": scope, "proposal": "docs/proposal.md", "arms": rows,
            "proposal_coverage": coverage,
            "recipe_note": (
                "Proposal data/batch/accumulation recipe; M1 batch admission may block execution. "
                "Local curves remain component tests, not federated measurements."
                if scope == "proposal" else
                "Provisional M4 comparison: first 4096 records, batch 1, accumulation 1, "
                "checkpointing off, three epochs. Verify the M4 saved recipe; not FedAvg."
                if scope == "comparison" else
                "User-requested M1 adaptation: full training split, batch 1, accumulation 1, "
                "checkpointing off, three epochs; differs from proposal's 4096/batch8/accum2."
            )}


def active_supervisors() -> list[int]:
    active = []
    for pointer in (ROOT / "runs").rglob("supervisor.pid"):
        if (pointer.parent / "exit.json").exists():
            continue
        try:
            pid = int(pointer.read_text().strip())
            command = psutil.Process(pid).cmdline()
            if any("bounded_smoke.py" in word for word in command):
                active.append(pid)
        except (OSError, ValueError, psutil.NoSuchProcess):
            continue
        except psutil.AccessDenied:
            active.append(pid)  # Unknown ownership is not permission to overlap runs.
    return active


def set_swap_override(plan: dict, mib: int | None, disable: bool = False):
    if disable:
        if mib is not None:
            raise ValueError("cannot disable and set a swap-growth limit together")
        for arm in plan["arms"]:
            arm["config"].setdefault("safety", {})["max_swap_growth_bytes"] = None
        plan["swap_growth_guard_disabled"] = True
        return
    if mib is None:
        return
    if isinstance(mib, bool) or not isinstance(mib, int) or mib <= 0:
        raise ValueError("--max-swap-growth-mib must be a positive integer")
    for arm in plan["arms"]:
        arm["config"].setdefault("safety", {})["max_swap_growth_bytes"] = mib * 1024**2
    plan["max_swap_growth_override_mib"] = mib


def result_row(name: str, directory: Path, exit_code: int) -> dict:
    pointer = directory / "logs" / "latest"
    result = {}
    run = Path(pointer.read_text().strip()) if pointer.exists() else None
    if run is not None and (run / "result.json").exists():
        result = json.loads((run / "result.json").read_text())
    supervisor = directory / "job" / "exit.json"
    control = json.loads(supervisor.read_text()) if supervisor.exists() else {}
    history = result.get("history") or {}
    if not history and run is not None and (run / "history.jsonl").exists():
        history = {"train": [], "validation": []}
        for line in (run / "history.jsonl").read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if row.get("kind") in history:
                history[row["kind"]].append(row)
    reports = history.get("train") or []
    last = reports[-1] if reports else {}
    return {"name": name,
            "status": ("completed" if exit_code == 0 and result.get("status") == "ok"
                       else "stopped" if control.get("stop_reason") else "failed"),
            "reason": control.get("stop_reason") or result.get("error"),
            "exit_code": exit_code, "run_directory": None if run is None else str(run),
            "steps": result.get("iters"), "train_examples": result.get("train_examples"),
            "resumed_step": result.get("resumed_step"),
            "timing_scope": "this invocation only; resumed-run time excludes earlier segments",
            "initial_test_perplexity": (result.get("baseline_test") or {}).get("perplexity"),
            "final_test_perplexity": (result.get("adapted_test") or {}).get("perplexity"),
            "last_reported_step": last.get("iteration"),
            "last_reported_train_loss": last.get("train_loss"),
            "last_reported_tokens": last.get("trained_tokens"),
            "peak_mlx_bytes": result.get("mlx_peak_bytes") or (
                None if last.get("peak_memory") is None else int(last["peak_memory"] * 1e9)),
            "seconds": result.get("seconds"),
            "validation": history.get("validation", []),
            "energy_joules": None,
            "energy_note": "power channel validity requires review; no automatic energy ranking"}


def import_existing(arm: dict, source: Path) -> dict:
    """Reuse a completed matching GPU run rather than repeat hours of training."""

    config = yaml.safe_load((source / "config.yaml").read_text())
    result = json.loads((source / "result.json").read_text())
    expected = arm["config"]
    if result.get("status") != "ok" or config.get("placement", {}).get("mode") != "gpu":
        raise ValueError("existing run must be completed GPU training")
    for section in ("data", "model", "training"):
        if config[section] != expected[section]:
            raise ValueError(f"existing run differs from requested {section} recipe")
    return {"name": arm["name"], "status": "completed", "imported": True,
            "run_directory": str(source), "steps": result["iters"],
            "train_examples": result["train_examples"],
            "initial_test_perplexity": result["baseline_test"]["perplexity"],
            "final_test_perplexity": result["adapted_test"]["perplexity"],
            "peak_mlx_bytes": result.get("mlx_peak_bytes"), "seconds": result.get("seconds"),
            "validation": result.get("history", {}).get("validation", []),
            "energy_joules": None, "reason": None}


def write_summary(output: Path, summary: dict):
    temporary = output / "summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2) + "\n")
    temporary.replace(output / "summary.json")
    fields = ["name", "status", "initial_test_perplexity", "final_test_perplexity",
              "last_reported_step", "last_reported_train_loss",
              "peak_mlx_bytes", "seconds", "reason"]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["results"])


def print_summary(summary: dict):
    print(f"Scope: {summary['scope']}")
    print(summary["recipe_note"])
    for row in summary["results"]:
        print(f"{row['name']}: {row['status']} | "
              f"test PPL={row.get('final_test_perplexity')} | "
              f"last reported step={row.get('last_reported_step')} | "
              f"train loss={row.get('last_reported_train_loss')} | {row.get('reason') or ''}")
    print("Proposal coverage:")
    for row in summary["proposal_coverage"]:
        print(f"  {row['name']}: {row['status']} — {row['reason']}")


def progress(directory: Path) -> str:
    log = directory / "job" / "stdout.log"
    if not log.exists():
        return "starting"
    with log.open("rb") as handle:
        handle.seek(max(0, log.stat().st_size - 2000))
        text = handle.read().decode("utf-8", errors="replace")
    lines = text.replace("\r", "\n").splitlines()
    return next((line.strip() for line in reversed(lines) if line.strip()), "running")[:300]


def allocate_suite(output: Path) -> Path:
    candidate = output
    suffix = 2
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = output.with_name(f"{output.name}-{suffix}")
            suffix += 1


def validate_resume(config: dict, checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    for name in ("state.json", "adapters.safetensors", "optimizer.safetensors"):
        if not (checkpoint / name).is_file():
            raise ValueError(f"resume checkpoint missing {name}: {checkpoint}")
    source_config = yaml.safe_load((checkpoint.parent.parent / "config.yaml").read_text())
    if source_config.get("placement", {}).get("mode") != config.get("placement", {}).get("mode"):
        raise ValueError("resume checkpoint differs from requested placement mode")
    for section in ("data", "model", "training"):
        if source_config[section] != config[section]:
            raise ValueError(f"resume checkpoint differs from requested {section} recipe")
    return checkpoint


def execute(plan: dict, output: Path, seconds: int, existing_gpu: Path | None = None,
            resume_gpu: Path | None = None, resume_gpu_ane: Path | None = None) -> dict:
    active = active_supervisors()
    if active:
        raise RuntimeError(f"another supervised run is active: {active}; wait for it to finish")
    requested = output
    output = allocate_suite(output)
    if output != requested:
        print(f"Output already exists; preserving it and using {output}", flush=True)
    (output / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    summary = {k: v for k, v in plan.items() if k != "arms"}
    summary["output_directory"] = str(output)
    summary["results"] = []
    write_summary(output, summary)
    for arm in plan["arms"]:
        directory = output / arm["name"]
        directory.mkdir()
        if arm["name"] == "local_gpu" and existing_gpu is not None:
            try:
                row = import_existing(arm, existing_gpu)
            except (OSError, ValueError, KeyError) as exc:
                row = {"name": arm["name"], "status": "blocked", "reason": str(exc)}
            summary["results"].append(row)
            write_summary(output, summary)
            continue
        config = copy.deepcopy(arm["config"])
        config["logging"]["dir"] = str(directory / "logs")
        resume_source = resume_gpu if arm["name"] == "local_gpu" else resume_gpu_ane
        if resume_source is not None:
            try:
                checkpoint = validate_resume(config, resume_source)
            except (OSError, ValueError, KeyError) as exc:
                summary["results"].append({"name": arm["name"], "status": "blocked",
                                           "reason": str(exc)})
                write_summary(output, summary)
                continue
            config.setdefault("checkpoint", {})["resume"] = str(checkpoint)
            print(f"Resuming {arm['name']} from {checkpoint}", flush=True)
        readiness = check_config(config)
        (directory / "preflight.json").write_text(json.dumps(readiness, indent=2) + "\n")
        if readiness["status"] != "ready":
            summary["results"].append({"name": arm["name"], "status": "blocked",
                                       "reason": "; ".join(readiness["blockers"])})
            write_summary(output, summary)
            continue
        config_path = directory / "config.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        command = [sys.executable, str(ROOT / "tools/bounded_smoke.py"), "--config",
                   str(config_path), "--output", str(directory / "job"), "--seconds", str(seconds)]
        if Path("/usr/bin/caffeinate").exists():
            command = ["/usr/bin/caffeinate", "-i", *command]
        swap_bytes = config.get("safety", {}).get("max_swap_growth_bytes", 512 * 1024**2)
        swap_policy = ("swap-growth stop disabled" if swap_bytes is None
                       else f"swap-growth allowance {swap_bytes / 1024**2:g} MiB")
        disk_gib = config["safety"]["min_free_disk_bytes"] / 1024**3
        print(f"Starting {arm['name']} | {swap_policy} | disk reserve {disk_gib:g} GiB | {directory}",
              flush=True)
        with (directory / "supervisor.log").open("w") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            try:
                while child.poll() is None:
                    print(f"[{arm['name']}] {progress(directory)}", flush=True)
                    time.sleep(20)
            except KeyboardInterrupt:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=20)
                summary["results"].append(result_row(arm["name"], directory, child.returncode))
                write_summary(output, summary)
                raise
        summary["results"].append(result_row(arm["name"], directory, child.returncode))
        write_summary(output, summary)
        print_summary(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("comparison", "full", "proposal"), default="comparison")
    parser.add_argument("--run", action="store_true", help="execute available arms sequentially")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seconds-per-arm", type=int, default=172800)
    parser.add_argument("--report", type=Path, help="print a saved suite summary without training")
    gpu_source = parser.add_mutually_exclusive_group()
    gpu_source.add_argument("--existing-gpu", type=Path, help="reuse a finished matching GPU run")
    gpu_source.add_argument("--resume-gpu", type=Path, help="resume an incomplete matching GPU checkpoint")
    parser.add_argument("--resume-gpu-ane", type=Path, help="resume an incomplete matching hybrid checkpoint")
    parser.add_argument("--min-free-disk-gib", type=int, help="override disk reserve in GiB (positive integer)")
    swap_group = parser.add_mutually_exclusive_group()
    swap_group.add_argument("--max-swap-growth-mib", type=int,
                            help="override system swap-growth allowance per run; e.g. 1024")
    swap_group.add_argument("--no-swap-growth-guard", action="store_true",
                            help="disable the swap-growth stop; retain other resource guards")
    args = parser.parse_args()
    os.chdir(ROOT)
    os.environ["PATH"] = str(ROOT / "runs/M1_8GB/setup/bin") + os.pathsep + os.environ["PATH"]
    os.environ.setdefault("HF_HOME", str(ROOT / "runs/M1_8GB/huggingface"))
    os.environ.setdefault("ANEFORGE_CACHE_DIR", str(ROOT / "runs/M1_8GB/ane-cache"))
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "runs/M1_8GB/matplotlib"))
    os.environ["PYTHONPATH"] = str(ROOT / "src")
    if args.report:
        print_summary(json.loads((args.report / "summary.json").read_text()))
        return
    plan = build_plan(args.scope)
    if args.min_free_disk_gib is not None:
        if args.min_free_disk_gib < 1:
            parser.error("--min-free-disk-gib must be a positive integer")
        for arm in plan["arms"]:
            arm["config"]["safety"]["min_free_disk_bytes"] = args.min_free_disk_gib * 1024**3
    try:
        set_swap_override(plan, args.max_swap_growth_mib, args.no_swap_growth_guard)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.run:
        for arm in plan["arms"]:
            arm["readiness"] = check_config(arm["config"])
        print(json.dumps(plan, indent=2))
        return
    output = args.output or ROOT / "runs/baseline-suites" / datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    try:
        summary = execute(plan, output.resolve(), args.seconds_per_arm,
                          args.existing_gpu, args.resume_gpu, args.resume_gpu_ane)
    except (RuntimeError, OSError) as exc:
        parser.exit(2, f"{exc}\n")
    output = Path(summary["output_directory"])
    print(f"Saved: {output / 'summary.json'} and summary.csv", flush=True)
    # Coverage gaps are never a successful full-proposal result.
    raise SystemExit(2 if any(r["status"] != "completed" for r in summary["results"])
                     or any(r["status"] == "blocked" for r in summary["proposal_coverage"]) else 0)


if __name__ == "__main__":
    main()
