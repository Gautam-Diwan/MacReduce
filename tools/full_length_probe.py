"""Exercise the configured longest sequence before a sustained GPU run."""

import argparse
import json
from pathlib import Path

import mlx.core as mx
import yaml
from mlx_lm.utils import load

from macreduce.finetune import TRAIN_MODEL, MetricLog, _namespace, recipe_from_config
from macreduce.memory import apply_memory_limit
from macreduce.serving import MODEL_REVISIONS
from macreduce.train_loop import train_checked


class TokenDataset(list):
    def process(self, item):
        return item

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    apply_memory_limit(config["infrastructure"]["memory_limit_bytes"])
    model, tokenizer = load(TRAIN_MODEL, revision=MODEL_REVISIONS[TRAIN_MODEL])
    data_dir = Path(config["data"]["path"])
    length = config["model"]["max_seq_length"]
    tokens = None
    for line in (data_dir / "train.jsonl").read_text().splitlines():
        text = json.loads(line)["text"]
        if len(text) >= length:
            encoded = tokenizer.encode(text)
            if len(encoded) >= length:
                tokens = encoded[:length]
                break
    if tokens is None:
        raise RuntimeError("no maximum-length record found")
    recipe = recipe_from_config(config)
    recipe["lr_schedule"] = None  # Three-step memory probe; not the baseline recipe.
    namespace = _namespace(data_dir, args.output / "adapters", 3, 1, 1, 1, length,
                           1e-4, 0, 1, 1, recipe=recipe)
    namespace.grad_checkpoint = config["training"]["grad_checkpoint"]
    history = MetricLog(args.output / "history.jsonl")
    report = train_checked(model, TokenDataset([(tokens, 0)] * 3), None, namespace, history,
                           args.output / "checkpoints", 3)
    report.update(status="ok", max_record_tokens=length, grad_checkpoint=namespace.grad_checkpoint,
                  mlx_peak_bytes=mx.get_peak_memory(), history=history.snapshot())
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "mlx_peak_bytes", "grad_checkpoint")}))
