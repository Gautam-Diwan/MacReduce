"""Bounded real-checkpoint recovery probe; stops after the next update report."""

import hashlib
import json
from pathlib import Path

import mlx.core as mx
import yaml
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.utils import load

from macreduce.finetune import TRAIN_MODEL, _namespace, recipe_from_config
from macreduce.memory import apply_memory_limit
from macreduce.serving import MODEL_REVISIONS
from macreduce.train_loop import train_checked

HERE = Path(__file__).resolve().parent
SOURCE = Path("runs/baseline-suites/comparison-no-swap-1/local_gpu/logs/2026-11-10T12-18-28")
CHECKPOINT = SOURCE / "checkpoints/0002000"


class ProbeComplete(Exception):
    pass


class OneReport:
    def on_train_loss_report(self, report):
        assert report["iteration"] == 2001
        assert mx.isfinite(mx.array(report["train_loss"])).item()
        self.report = report
        raise ProbeComplete()


def hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in CHECKPOINT.iterdir() if p.is_file()}


def main():
    config = yaml.safe_load((SOURCE / "config.yaml").read_text())
    before = hashes()
    apply_memory_limit(config["infrastructure"]["memory_limit_bytes"])
    model, tokenizer = load(TRAIN_MODEL, revision=MODEL_REVISIONS[TRAIN_MODEL])
    args = _namespace(
        Path("runs/M1_8GB/data/wikitext-2-raw-v1-curve-4096"),
        HERE / "unused-adapters", 12288, 3, -1, 1, 512, 1e-4, 0, 1, -1,
        recipe=recipe_from_config(config),
    )
    args.grad_checkpoint = False
    args.steps_per_report = 1
    train, _, _ = load_dataset(args, tokenizer)
    history = OneReport()
    try:
        train_checked(model, train, None, args, history, HERE / "unused-checkpoints", 500, resume=CHECKPOINT)
    except ProbeComplete:
        pass
    else:
        raise AssertionError("probe did not stop at the first resumed report")
    assert before == hashes(), "source checkpoint changed"
    report = {
        "status": "ok", "checkpoint": str(CHECKPOINT), "checkpoint_sha256": before,
        "scope": "normal trainer, same seed/order/recipe, one compiled update; no evaluation or long run",
        "source_checkpoint_unchanged": True, "report": history.report,
    }
    (HERE / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
