"""
Time one federated client's local QLoRA step under alternative step designs.

Each configuration runs in a fresh process so peak memory and the global
grad_checkpoint class patch never leak between measurements. Throughput counts
unpadded target tokens only. This is a GPU-only microbenchmark of step cost; it
does not measure convergence, transport, or ANE placement.

    uv run python tools/bench_local_step.py --out evidence/<dir>/steps.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "line-eager": {"pack": 0, "batch": 1, "compile": False, "checkpoint": False},
    "line-compiled": {"pack": 0, "batch": 1, "compile": True, "checkpoint": False},
    "pack512-eager": {"pack": 512, "batch": 1, "compile": False, "checkpoint": False},
    "pack512-compiled": {"pack": 512, "batch": 1, "compile": True, "checkpoint": False},
    "pack512x4-compiled": {"pack": 512, "batch": 4, "compile": True, "checkpoint": False},
    "line-eager-checkpoint": {"pack": 0, "batch": 1, "compile": False, "checkpoint": True},
    "pack512-eager-checkpoint": {"pack": 512, "batch": 1, "compile": False, "checkpoint": True},
}


def batches(tokenizer, records, pack, batch, count):
    """Yield (tokens, lengths, target_tokens) exactly like the federation collate."""
    encoded = []
    for record in records:
        ids = list(tokenizer.encode(record["text"]))
        if not ids or ids[-1] != tokenizer.eos_token_id:
            ids.append(tokenizer.eos_token_id)
        encoded.append(ids[:512])
    if not pack:
        for ids in encoded[:count]:
            width = min(512, 1 + 32 * ((len(ids) + 31) // 32))
            row = np.zeros((1, width), dtype=np.int32)
            row[0, :len(ids)] = ids
            yield row, np.array([(0, len(ids))], dtype=np.int32), len(ids) - 1
        return
    stream = [token for ids in encoded for token in ids]
    rows = [stream[i:i + pack + 1] for i in range(0, len(stream) - pack, pack)]
    for start in range(0, min(len(rows), count * batch) - batch + 1, batch):
        chunk = np.array(rows[start:start + batch], dtype=np.int32)
        lengths = np.array([(0, pack + 1)] * batch, dtype=np.int32)
        yield chunk, lengths, batch * pack


def measure(name: str, steps: int, warmup: int) -> dict:
    import mlx.core as mx
    import mlx.optimizers as optim
    from mlx import nn
    from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.utils import load

    from macreduce.serve.serving import DRAFT_MODEL, MODEL_REVISIONS
    from macreduce.train.memory import apply_memory_limit, resolve_memory_limit

    config = CONFIGS[name]
    mx.set_default_device(mx.gpu)
    apply_memory_limit(resolve_memory_limit("auto"))
    mx.random.seed(0)
    model, tokenizer = load(DRAFT_MODEL, revision=MODEL_REVISIONS[DRAFT_MODEL])
    model.freeze()
    linear_to_lora_layers(model, -1, {"rank": 8, "scale": 2.0, "dropout": 0.0}, use_dora=False)
    if config["checkpoint"]:
        grad_checkpoint(model.layers[0])
    optimizer = optim.AdamW(learning_rate=1e-4, weight_decay=0.01)
    optimizer.init(model.trainable_parameters())
    loss_and_grad = nn.value_and_grad(model, default_loss)
    model.train()

    def eager(tokens, lengths):
        (loss, toks), grad = loss_and_grad(model, tokens, lengths)
        mx.eval(loss, toks, grad)
        grad, _ = optim.clip_grad_norm(grad, 1.0)
        optimizer.update(model, grad)
        mx.eval(model.parameters(), optimizer.state)
        return loss

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def fused(tokens, lengths):
        (loss, _), grad = loss_and_grad(model, tokens, lengths)
        grad, _ = optim.clip_grad_norm(grad, 1.0)
        optimizer.update(model, grad)
        return loss

    def compiled(tokens, lengths):
        loss = fused(tokens, lengths)
        mx.eval(loss, state)
        return loss

    step = compiled if config["compile"] else eager
    records = [json.loads(line) for line in
               (ROOT / "data/wikitext-2-raw-v1/train.jsonl").read_text().splitlines()[:4096]]
    plan = list(batches(tokenizer, records, config["pack"], config["batch"], warmup + steps))
    for tokens, lengths, _ in plan[:warmup]:
        step(mx.array(tokens), mx.array(lengths))
    mx.reset_peak_memory()
    times, counted = [], 0
    for tokens, lengths, targets in plan[warmup:warmup + steps]:
        begin = time.perf_counter()
        loss = step(mx.array(tokens), mx.array(lengths))
        times.append(time.perf_counter() - begin)
        counted += targets
        if not np.isfinite(float(loss)):
            raise RuntimeError("nonfinite loss in benchmark")
    seconds = float(np.sum(times))
    return {
        "config": name, **config, "steps": len(times), "target_tokens": counted,
        "seconds": seconds, "median_step_s": float(np.median(times)),
        "tokens_per_second": counted / seconds, "tokens_per_step": counted / len(times),
        "peak_memory_bytes": int(mx.get_peak_memory()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--config", choices=sorted(CONFIGS))
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.config:
        print(json.dumps(measure(args.config, args.steps, args.warmup)))
        return
    import mlx.core as mx

    rows = []
    for name in CONFIGS:
        result = subprocess.run(  # noqa: S603
            [sys.executable, __file__, "--config", name, "--steps", str(args.steps),
             "--warmup", str(args.warmup)],
            check=True, capture_output=True, text=True, cwd=ROOT,
        )
        rows.append(json.loads(result.stdout.strip().splitlines()[-1]))
        print(json.dumps(rows[-1]))
    report = {
        "host": {"machine": platform.machine(), "mac_ver": platform.mac_ver()[0],
                 "device": mx.device_info().get("device_name"),
                 "memory_bytes": mx.device_info().get("memory_size")},
        "mlx": mx.__version__, "steps": args.steps, "warmup": args.warmup, "results": rows,
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
