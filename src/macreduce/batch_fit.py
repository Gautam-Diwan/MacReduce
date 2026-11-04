"""Pick a QLoRA microbatch by steady tokens per second.

Each candidate runs in its own process, so an out-of-memory kill does not
end the search. The uncheckpointed step is tried first. Gradient
checkpointing is the fallback when that step does not fit. The winner is
the fastest fit, not the largest one.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

from .finetune import (
    MODEL_REVISIONS,
    TRAIN_MODEL,
    MetricLog,
    _namespace,
    prepare_dataset,
    write_curve_split,
)
from .memory import apply_memory_limit

PROBE_STEPS = 3
ASCENDING = (1, 2, 4, 8, 16, 32, 48)
DESCENDING = ()


def steady_tokens_per_second(rates: list[float]) -> float:
    """Median tokens/s after the first report. The first report includes compile."""

    if not rates:
        raise ValueError("no token rates")
    body = rates[1:] or rates
    return float(statistics.median(body))


def select_trial(trials: list[dict]) -> dict:
    """Fastest successful trial. Ties go to the larger batch."""

    ok = [trial for trial in trials if trial.get("status") == "ok"]
    if not ok:
        raise RuntimeError("no batch size completed a step under the memory limit")
    return max(ok, key=lambda trial: (trial["tokens_per_second"], trial["batch_size"]))


def choose_batch(
    data_dir: Path,
    *,
    memory_limit_bytes: int,
    max_seq_length: int = 512,
    train_examples: int = 4096,
    grad_accumulation_steps: int = 1,
    learning_rate: float = 1e-4,
    seed: int = 0,
    batch_sizes: list[int] | None = None,
    grad_checkpoint: bool | None = None,
    log_dir: Path | None = None,
    recipe: dict | None = None,
) -> dict:
    """Search batch size. ``grad_checkpoint=None`` tries the fast step, then checkpointing."""

    if memory_limit_bytes < 1:
        raise ValueError("memory limit must be positive")
    sizes = list(batch_sizes) if batch_sizes else list(ASCENDING)
    if log_dir is not None:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        (Path(log_dir) / "trials.jsonl").write_text("")
    print(
        f"batch search: {memory_limit_bytes / 1e9:.2f} GB working set, "
        f"objective steady tokens/s, sequence {max_seq_length}",
        flush=True,
    )
    trials: list[dict] = []
    found = _sweep(
        trials,
        sizes,
        data_dir=data_dir,
        memory_limit_bytes=memory_limit_bytes,
        max_seq_length=max_seq_length,
        train_examples=train_examples,
        grad_accumulation_steps=grad_accumulation_steps,
        learning_rate=learning_rate,
        seed=seed,
        grad_checkpoint=grad_checkpoint,
        log_dir=log_dir,
        recipe=recipe,
        stop_on_miss=batch_sizes is None,
    )
    if not found and batch_sizes is None:
        _sweep(
            trials,
            DESCENDING,
            data_dir=data_dir,
            memory_limit_bytes=memory_limit_bytes,
            max_seq_length=max_seq_length,
            train_examples=train_examples,
            grad_accumulation_steps=grad_accumulation_steps,
            learning_rate=learning_rate,
            seed=seed,
            grad_checkpoint=grad_checkpoint,
            log_dir=log_dir,
            recipe=recipe,
            stop_on_miss=False,
            stop_after_hit=True,
        )
    best = select_trial(trials)
    return {
        "memory_limit_bytes": memory_limit_bytes,
        "objective": "tokens_per_second",
        "batch_size": best["batch_size"],
        "grad_checkpoint": best["grad_checkpoint"],
        "peak_bytes": best["peak_bytes"],
        "tokens_per_second": best["tokens_per_second"],
        "trials": trials,
    }


def run_probe(spec: dict) -> dict:
    """One microbatch measurement. Called in a child process."""

    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import TrainingArgs, train
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load

    batch_size = int(spec["batch_size"])
    limit = int(spec["memory_limit_bytes"])
    apply_memory_limit(limit)
    data_dir = Path(spec["data_dir"])
    prepare_dataset(data_dir)
    examples = max(int(spec["train_examples"]), batch_size)
    train_dir = data_dir.parent / f"{data_dir.name}-curve-{examples}"
    write_curve_split(data_dir, train_dir, examples)
    model, tokenizer = load(TRAIN_MODEL, revision=MODEL_REVISIONS[TRAIN_MODEL])
    try:
        with tempfile.TemporaryDirectory(prefix="macreduce-batch-") as tmp:
            args = _namespace(
                train_dir,
                Path(tmp) / "adapters",
                iters=PROBE_STEPS,
                epochs=1,
                test_batches=1,
                batch_size=batch_size,
                max_seq_length=int(spec["max_seq_length"]),
                learning_rate=float(spec["learning_rate"]),
                seed=int(spec["seed"]),
                grad_accumulation_steps=int(spec["grad_accumulation_steps"]),
                val_batches=1,
                recipe=spec.get("recipe"),
            )
            model.freeze()
            linear_to_lora_layers(model, args.num_layers, args.lora_parameters, use_dora=False)
            history = MetricLog()
            train(
                model,
                optim.AdamW(learning_rate=args.learning_rate, weight_decay=args.weight_decay),
                CacheDataset(load_dataset(args, tokenizer)[0]),
                val_dataset=None,
                args=TrainingArgs(
                    batch_size=batch_size,
                    iters=PROBE_STEPS,
                    val_batches=0,
                    steps_per_report=1,
                    steps_per_eval=10**9,
                    steps_per_save=10**9,
                    max_seq_length=args.max_seq_length,
                    adapter_file=str(Path(tmp) / "adapters.safetensors"),
                    grad_checkpoint=bool(spec["grad_checkpoint"]),
                    grad_accumulation_steps=args.grad_accumulation_steps,
                ),
                training_callback=history,
            )
        peak = int(mx.get_peak_memory())
        rates = [float(row["tokens_per_second"]) for row in history.train]
        rate = steady_tokens_per_second(rates)
    finally:
        del model
        mx.clear_cache()
    # A completed step a hair over the working set still ran. A large overshoot did not fit.
    status = "ok" if peak <= int(limit * 1.02) else "over"
    return {
        "batch_size": batch_size,
        "grad_checkpoint": bool(spec["grad_checkpoint"]),
        "status": status,
        "peak_bytes": peak,
        "tokens_per_second": rate,
    }


def _sweep(trials: list[dict], sizes: list[int] | tuple[int, ...], **kwargs) -> bool:
    stop_on_miss = kwargs.pop("stop_on_miss")
    stop_after_hit = kwargs.pop("stop_after_hit", False)
    found = False
    best_rate = 0.0
    for batch_size in tqdm(list(sizes), desc="batch search", unit="size"):
        attempts = _measure_size(batch_size, **kwargs)
        trials.extend(attempts)
        ok_rates = [
            attempt["tokens_per_second"]
            for attempt in attempts
            if attempt.get("status") == "ok" and attempt.get("tokens_per_second") is not None
        ]
        if ok_rates:
            found = True
            rate = max(ok_rates)
            if best_rate and rate < 0.9 * best_rate:
                tqdm.write(
                    f"batch {batch_size} dropped to {rate:.0f} tok/s from {best_rate:.0f}, stopping"
                )
                break
            best_rate = max(best_rate, rate)
            if stop_after_hit:
                break
            continue
        if stop_on_miss:
            break
    return found


def _measure_size(batch_size: int, *, grad_checkpoint: bool | None, **kwargs) -> list[dict]:
    modes = (grad_checkpoint,) if grad_checkpoint is not None else (False, True)
    attempts = []
    for mode in modes:
        attempt = _child(batch_size, grad_checkpoint=bool(mode), **kwargs)
        attempts.append(attempt)
        _record(kwargs.get("log_dir"), attempt)
        rate = attempt.get("tokens_per_second")
        rate_text = "n/a" if rate is None else f"{rate:.0f} tok/s"
        peak = attempt.get("peak_bytes")
        peak_text = "n/a" if peak is None else f"{peak / 1e9:.2f} GB"
        tqdm.write(
            f"batch {batch_size} checkpoint={bool(mode)} {attempt['status']} "
            f"peak {peak_text} {rate_text}"
        )
        if attempt["status"] == "ok":
            break
    return attempts


def _child(batch_size: int, **kwargs) -> dict:
    log_dir = kwargs.get("log_dir")
    if log_dir is not None:
        result_path = Path(log_dir) / "trials" / f"{batch_size}-ckpt{int(kwargs['grad_checkpoint'])}.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        fd, name = tempfile.mkstemp(prefix="macreduce-trial-", suffix=".json")
        os.close(fd)
        result_path = Path(name)
    spec = {
        "data_dir": str(kwargs["data_dir"]),
        "batch_size": batch_size,
        "memory_limit_bytes": kwargs["memory_limit_bytes"],
        "max_seq_length": kwargs["max_seq_length"],
        "train_examples": kwargs["train_examples"],
        "grad_accumulation_steps": kwargs["grad_accumulation_steps"],
        "learning_rate": kwargs["learning_rate"],
        "seed": kwargs["seed"],
        "grad_checkpoint": kwargs["grad_checkpoint"],
        "recipe": kwargs.get("recipe"),
        "result_path": str(result_path),
    }
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "macreduce.batch_fit", json.dumps(spec)],
            cwd=str(Path(__file__).resolve().parents[2]),
            text=True,
            capture_output=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return _failed(batch_size, kwargs["grad_checkpoint"], "timeout")
    if result_path.is_file():
        return json.loads(result_path.read_text())
    detail = (proc.stderr or proc.stdout or "").strip().splitlines()
    tail = detail[-1] if detail else f"exit {proc.returncode}"
    return _failed(batch_size, kwargs["grad_checkpoint"], tail)


def _failed(batch_size: int, grad_checkpoint: bool, error: str) -> dict:
    return {
        "batch_size": batch_size,
        "grad_checkpoint": grad_checkpoint,
        "status": "error",
        "error": error,
    }


def _record(log_dir: Path | None, attempt: dict) -> None:
    if log_dir is None:
        return
    path = Path(log_dir)
    path.mkdir(parents=True, exist_ok=True)
    with (path / "trials.jsonl").open("a") as stream:
        stream.write(json.dumps(attempt) + "\n")


def main() -> None:
    spec = json.loads(sys.argv[1])
    destination = Path(spec["result_path"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = run_probe(spec)
    except Exception as exc:
        result = _failed(int(spec["batch_size"]), bool(spec["grad_checkpoint"]), f"{type(exc).__name__}: {exc}")
    destination.write_text(json.dumps(result))


if __name__ == "__main__":
    main()
