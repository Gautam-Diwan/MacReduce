"""Train Qwen3 QLoRA from a YAML config.

    python main.py --config configs/qlora_m4.yaml

``training.batch_size: auto`` measures steady tokens per second under the
GPU's recommended working set and keeps the fastest microbatch.
``grad_checkpoint: auto`` turns recomputation on only when the plain step
does not fit. Checkpoints are written every ``checkpoint.save_every`` steps.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml

from macreduce.batch_fit import choose_batch
from macreduce.curve import allocate_run, bind_run_log, latest_checkpoint, run_curve
from macreduce.finetune import recipe_from_config
from macreduce.memory import resolve_memory_limit


def load_config(path: Path) -> dict:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen3 QLoRA from a YAML config")
    parser.add_argument("--config", required=True, help="hyperparameter YAML")
    parser.add_argument("--batch-size", type=int, default=None, help="override training.batch_size")
    parser.add_argument(
        "--fit-only",
        action="store_true",
        help="choose the batch for tokens per second and exit",
    )
    parser.add_argument("--resume", default=None, help="checkpoint directory, or latest")
    return parser.parse_args()


def resolve_batch(config: dict, override: int | None) -> dict:
    training = config["training"]
    infra = config["infrastructure"]
    infra["memory_limit_bytes"] = resolve_memory_limit(infra.get("memory_limit_bytes"))
    if override is not None:
        training["batch_size"] = override
    checkpoint = training.get("grad_checkpoint", True)
    need_batch = training.get("batch_size") in (None, "auto")
    need_checkpoint = checkpoint in (None, "auto")
    if not need_batch and not need_checkpoint:
        training["grad_checkpoint"] = bool(checkpoint)
        return config
    data = config["data"]
    fitted = choose_batch(
        Path(data["path"]),
        memory_limit_bytes=int(infra["memory_limit_bytes"]),
        max_seq_length=int(config["model"]["max_seq_length"]),
        train_examples=int(data["train_examples"]),
        grad_accumulation_steps=int(training["grad_accumulation_steps"]),
        learning_rate=float(training["learning_rate"]),
        seed=int(infra["seed"]),
        batch_sizes=None if need_batch else [int(training["batch_size"])],
        grad_checkpoint=None if need_checkpoint else bool(checkpoint),
        log_dir=Path("evidence/batch-fit"),
        recipe=recipe_from_config(config),
    )
    training["batch_size"] = fitted["batch_size"]
    training["grad_checkpoint"] = fitted["grad_checkpoint"]
    config["batch_fit"] = fitted
    print(
        f"batch {fitted['batch_size']} at {fitted['tokens_per_second']:.0f} tok/s "
        f"(peak {fitted['peak_bytes']} bytes, checkpoint={fitted['grad_checkpoint']})",
        flush=True,
    )
    return config


def train(config: dict, output: Path) -> dict:
    data = config["data"]
    training = config["training"]
    infra = config["infrastructure"]
    checkpoint = config.get("checkpoint") or {}
    (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    resume = checkpoint.get("resume")
    report = run_curve(
        output,
        Path(data["path"]),
        port=int(infra["prometheus_port"]),
        memory_limit_bytes=int(infra["memory_limit_bytes"]),
        placement=str(config.get("placement", {}).get("mode", "gpu")),
        iters=training.get("iters"),
        epochs=int(training["epochs"]),
        test_batches=int(training["test_batches"]),
        batch_size=int(training["batch_size"]),
        max_seq_length=int(config["model"]["max_seq_length"]),
        learning_rate=float(training["learning_rate"]),
        seed=int(infra["seed"]),
        train_examples=int(data["train_examples"]),
        grad_accumulation_steps=int(training["grad_accumulation_steps"]),
        val_batches=int(training["val_batches"]),
        steps_per_report=int(training["steps_per_report"]),
        grad_checkpoint=bool(training["grad_checkpoint"]),
        save_every=checkpoint.get("save_every", "epoch"),
        resume=None if resume in (None, "null") else str(resume),
        recipe=recipe_from_config(config),
    )
    report["run_dir"] = str(output)
    (output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    if args.resume is not None:
        config.setdefault("checkpoint", {})["resume"] = args.resume
    resume = (config.get("checkpoint") or {}).get("resume")
    if resume == "latest":
        config["checkpoint"]["resume"] = str(latest_checkpoint(Path(config["logging"]["dir"])))
    output = allocate_run(Path(config["logging"]["dir"]))
    (output / "run.pid").write_text(str(os.getpid()) + "\n")
    bind_run_log(output)
    print(f"run directory: {output}", flush=True)
    print(f"run log: {output / 'run.log'}", flush=True)
    config = resolve_batch(config, args.batch_size)
    if args.fit_only:
        print(json.dumps(config.get("batch_fit", {"batch_size": config["training"]["batch_size"]}), indent=2))
        (output / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
        return
    report = train(config, output)
    print(
        json.dumps(
            {
                "status": report.get("status"),
                "batch_size": config["training"]["batch_size"],
                "grad_checkpoint": config["training"]["grad_checkpoint"],
                "figures": report.get("figures"),
                "checkpoint": report.get("checkpoint"),
                "placement": report.get("placement"),
                "error": report.get("error"),
                "output": report.get("run_dir"),
            },
            indent=2,
        )
    )
    if report.get("status") != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
