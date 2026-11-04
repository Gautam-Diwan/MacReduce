"""QLoRA step loop with a progress bar and resumable checkpoints.

One checkpoint is the adapter, the AdamW moments, and the step number.
Resume loads those, then skips the batches already consumed. The batch order
matches a fresh run with the same seed because the iterator is reseeded.
LoRA dropout is 0, so the update itself does not draw random numbers.
"""

from __future__ import annotations

import json
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm


def resolve_save_every(value: int | str | None, steps: int, epochs: int) -> int:
    """``epoch`` saves once per epoch. An integer saves every that many steps."""

    if steps < 1 or epochs < 1:
        raise ValueError("steps and epochs must be positive")
    if value is None or value == "epoch":
        return max(steps // epochs, 1)
    every = int(value)
    if every < 1:
        raise ValueError("save_every must be positive")
    return every


def save_checkpoint(root: Path, step: int, model, optimizer, info: dict) -> Path:
    """Write ``root/0000123`` and point ``root/latest.json`` at it."""

    import mlx.core as mx

    if step < 1:
        raise ValueError("checkpoint step must be positive")
    destination = root / f"{step:07d}"
    destination.mkdir(parents=True, exist_ok=True)
    weights = _arrays(model.trainable_parameters())
    mx.eval(*weights.values())
    mx.save_safetensors(str(destination / "adapters.safetensors"), weights)
    moments, dtypes = _arrays(optimizer.state, dtypes=True)
    mx.eval(*moments.values())
    mx.save_safetensors(str(destination / "optimizer.safetensors"), moments)
    meta = {"step": step, "dtypes": dtypes, **info}
    (destination / "state.json").write_text(json.dumps(meta, indent=2) + "\n")
    (root / "latest.json").write_text(json.dumps({"step": step, "path": str(destination)}) + "\n")
    return destination


def load_checkpoint(path: Path, model, optimizer) -> dict:
    """Restore an adapter and AdamW state written by :func:`save_checkpoint`."""

    import mlx.core as mx
    from mlx.utils import tree_unflatten

    meta = json.loads((path / "state.json").read_text())
    model.load_weights(str(path / "adapters.safetensors"), strict=False)
    mx.eval(model.parameters())
    loaded = mx.load(str(path / "optimizer.safetensors"))
    dtypes = meta.get("dtypes") or {}
    pairs = []
    for key, value in loaded.items():
        if dtypes.get(key) == "uint64":
            value = value.astype(mx.uint64)
        pairs.append((key, value))
    optimizer.state = tree_unflatten(pairs)
    optimizer._initialized = True
    mx.eval(optimizer.state)
    return meta


def train_checked(
    model,
    train_set,
    valid_set,
    args,
    history,
    checkpoint_root: Path,
    save_every: int,
    resume: Path | None = None,
    ane_lora: bool = False,
) -> dict:
    """Train ``args.iters`` steps. ``history`` receives the mlx-lm report dicts."""

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.nn.utils import average_gradients
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import default_loss, evaluate, grad_checkpoint, iterate_batches
    from mlx_lm.tuner.utils import linear_to_lora_layers

    if save_every < 1:
        raise ValueError("save_every must be positive")
    mx.random.seed(args.seed)
    np.random.seed(args.seed)
    model.freeze()
    linear_to_lora_layers(model, args.num_layers, args.lora_parameters, use_dora=False)
    if ane_lora:
        from .ane_stack import enable_ane_suffix

        placed = enable_ane_suffix(model, int(args.max_seq_length))
        tqdm.write(placed["summary"])
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])
    trainable = sum(array.size for _, array in tree_flatten(model.trainable_parameters()))
    tqdm.write(f"trainable parameters: {trainable}")

    from .finetune import resolve_schedule

    schedule = resolve_schedule(getattr(args, "lr_schedule", None), args.learning_rate, args.iters)
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    max_grad_norm = getattr(args, "max_grad_norm", None)
    if schedule is not None:
        from mlx_lm.tuner.utils import build_schedule

        learning_rate = build_schedule(schedule)
    else:
        learning_rate = args.learning_rate
    optimizer = optim.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    lora = dict(args.lora_parameters)
    identity = {
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "max_seq_length": int(args.max_seq_length),
        "grad_checkpoint": bool(args.grad_checkpoint),
        "learning_rate": float(args.learning_rate),
        "lora_rank": int(lora["rank"]),
        "lora_scale": float(lora["scale"]),
        "lora_dropout": float(lora["dropout"]),
        "weight_decay": weight_decay,
        "max_grad_norm": None if max_grad_norm is None else float(max_grad_norm),
        "lr_schedule": schedule,
    }
    completed = 0
    if resume is not None:
        meta = load_checkpoint(resume, model, optimizer)
        completed = int(meta["step"])
        _check_resume(meta, identity)
        tqdm.write(f"resumed at step {completed} from {resume}")
    if completed >= args.iters:
        raise ValueError(f"checkpoint step {completed} is already at the end of {args.iters}")

    if args.grad_accumulation_steps < 1:
        raise ValueError("grad_accumulation_steps must be at least 1")
    world = mx.distributed.init()
    world_size = world.size()
    loss_value_and_grad = nn.value_and_grad(model, default_loss)
    compiled_state = [model.state, optimizer.state, mx.random.state]

    def step(batch, prev_grad, do_update):
        (lvalue, toks), grad = loss_value_and_grad(model, *batch)
        if prev_grad is not None:
            grad = tree_map(lambda left, right: left + right, grad, prev_grad)
        if do_update:
            grad = average_gradients(grad)
            if args.grad_accumulation_steps > 1:
                grad = tree_map(lambda value: value / args.grad_accumulation_steps, grad)
            if max_grad_norm is not None:
                grad, _norm = optim.clip_grad_norm(grad, float(max_grad_norm))
            optimizer.update(model, grad)
            grad = None
        return lvalue, toks, grad

    if not ane_lora:
        step = partial(mx.compile, inputs=compiled_state, outputs=compiled_state)(step)

    train_dataset = CacheDataset(train_set)
    val_dataset = CacheDataset(valid_set) if valid_set is not None else None
    batches = iterate_batches(
        dataset=train_dataset,
        batch_size=args.batch_size,
        max_seq_length=args.max_seq_length,
        loop=True,
        comm_group=world,
    )
    if completed:
        for _ in tqdm(range(completed), desc="resume", unit="step", leave=False):
            next(batches)

    losses = 0
    n_tokens = 0
    window = 0
    trained_tokens = 0
    train_time = 0.0
    grad_accum = None
    last = None
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    bar = tqdm(total=args.iters, initial=completed, desc="train", unit="step")
    try:
        for it, batch in zip(range(completed + 1, args.iters + 1), batches):
            tic = time.perf_counter()
            if val_dataset is not None and (
                it == 1 or it % args.steps_per_eval == 0 or it == args.iters
            ):
                val_loss, val_time = _validate(
                    model, val_dataset, args, evaluate, iterate_batches, default_loss
                )
                model.train()
                if history is not None:
                    history.on_val_loss_report(
                        {"iteration": it - 1, "val_loss": val_loss, "val_time": val_time}
                    )
                tic = time.perf_counter()

            lvalue, toks, grad_accum = step(batch, grad_accum, it % args.grad_accumulation_steps == 0)
            losses += lvalue
            n_tokens += toks
            window += 1
            mx.eval(compiled_state, losses, n_tokens, grad_accum)
            step_loss = float(lvalue.item())
            if not math.isfinite(step_loss):
                raise RuntimeError(f"training loss is not finite at step {it}")
            train_time += time.perf_counter() - tic
            bar.update(1)

            if it % args.steps_per_report == 0 or it == args.iters:
                train_loss = mx.distributed.all_sum(losses, stream=mx.cpu).item()
                train_loss /= window * world_size
                seen = mx.distributed.all_sum(n_tokens, stream=mx.cpu).item()
                tokens_sec = float(seen) / train_time
                trained_tokens += seen
                peak_gb = mx.get_peak_memory() / 1e9
                bar.set_postfix(loss=f"{train_loss:.3f}", tok_s=f"{tokens_sec:.0f}", refresh=False)
                if history is not None:
                    history.on_train_loss_report(
                        {
                            "iteration": it,
                            "train_loss": train_loss,
                            "learning_rate": optimizer.learning_rate.item(),
                            "iterations_per_second": window / train_time,
                            "tokens_per_second": tokens_sec,
                            "trained_tokens": trained_tokens,
                            "peak_memory": peak_gb,
                        }
                    )
                losses = 0
                n_tokens = 0
                window = 0
                train_time = 0.0

            if it % save_every == 0 or it == args.iters:
                info = {**identity, "peak_memory_gb": mx.get_peak_memory() / 1e9}
                last = save_checkpoint(checkpoint_root, it, model, optimizer, info)
                _publish_adapter(model, args.adapter_path)
                tqdm.write(f"checkpoint {it} -> {last}")
    finally:
        bar.close()
    report = {
        "resumed_step": completed,
        "checkpoint": None if last is None else str(last),
        "trainable_parameters": int(trainable),
        "ane_lora": ane_lora,
    }
    if ane_lora:
        from .ane_stack import ane_stats

        report["ane"] = ane_stats()
    return report


def _validate(model, dataset, args, evaluate, iterate_batches, default_loss) -> tuple[float, float]:
    if args.val_batches == -1:
        total = len(dataset) // args.batch_size
    else:
        total = min(len(dataset) // args.batch_size, args.val_batches)
    tic = time.perf_counter()
    bar = tqdm(total=total, desc="valid", unit="batch", leave=False)
    try:
        loss = evaluate(
            model=model,
            dataset=dataset,
            loss=default_loss,
            batch_size=args.batch_size,
            num_batches=args.val_batches,
            max_seq_length=args.max_seq_length,
            iterate_batches=iterate_batches,
            progress_callback=bar.update,
        )
    finally:
        bar.close()
    return float(loss), time.perf_counter() - tic


def _publish_adapter(model, adapter_path: str) -> None:
    import mlx.core as mx

    destination = Path(adapter_path)
    destination.mkdir(parents=True, exist_ok=True)
    weights = _arrays(model.trainable_parameters())
    mx.eval(*weights.values())
    mx.save_safetensors(str(destination / "adapters.safetensors"), weights)


def _arrays(tree, dtypes: bool = False):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    flat = {}
    kinds = {}
    for key, value in tree_flatten(tree):
        kinds[key] = str(value.dtype).split(".")[-1]
        if value.dtype == mx.uint64:
            value = value.astype(mx.int64)
        flat[key] = value
    if dtypes:
        return flat, kinds
    return flat


def _check_resume(meta: dict, identity: dict) -> None:
    exact = ("seed", "batch_size", "max_seq_length", "grad_checkpoint", "lora_rank", "lr_schedule")
    for key in exact:
        if key in meta and meta[key] != identity[key]:
            raise ValueError(f"checkpoint {key} is {meta[key]}, this run is {identity[key]}")
    close = ("learning_rate", "lora_scale", "lora_dropout", "weight_decay", "max_grad_norm")
    for key in close:
        if key not in meta:
            continue
        left, right = meta[key], identity[key]
        if (left is None) != (right is None) or (left is not None and abs(float(left) - float(right)) > 1e-12):
            raise ValueError(f"checkpoint {key} is {left}, this run is {right}")
