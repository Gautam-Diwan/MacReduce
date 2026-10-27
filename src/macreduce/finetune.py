"""QLoRA on one pinned dataset.

The dataset is Salesforce WikiText-2 raw v1, revision
``b08601e04326c79dfdd32d625aee71d232d685c3``. The base model is the pinned
Qwen3-0.6B 4-bit checkpoint. Training uses the installed ``mlx_lm`` LoRA
trainer. The default run is three epochs on the first 4,096 training lines,
batch size 8, and two gradient-accumulation steps, so validation perplexity
can be plotted without a pass over the full training split. The test split
stays complete. Perplexity is ``exp(mean token loss)``.
"""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from .serving import DRAFT_MODEL, MLX_LM_PIN, MODEL_REVISIONS

DATASET_ID = "Salesforce/wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
DATASET_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"
TRAIN_MODEL = DRAFT_MODEL
CURVE_TRAIN_EXAMPLES = 4096
CURVE_EPOCHS = 3
CURVE_BATCH_SIZE = 8
CURVE_ACCUMULATION = 2
PAGE_ROWS = 100
SPLITS = {
    "train": "train.jsonl",
    "validation": "valid.jsonl",
    "test": "test.jsonl",
}


def perplexity(loss: float) -> float:
    """Perplexity of a mean natural-log token loss."""

    if loss < 0:
        raise ValueError("loss must be non-negative")
    return math.exp(loss)


def epoch_steps(examples: int, batch_size: int, epochs: int) -> int:
    """Optimizer steps that cover ``epochs`` passes of a split."""

    if examples < 1 or batch_size < 1 or epochs < 1:
        raise ValueError("examples, batch size, and epochs must be positive")
    if examples < batch_size:
        raise ValueError("the split has fewer examples than the batch size")
    return epochs * (examples // batch_size)


def prefix_records(records: list[dict], count: int) -> list[dict]:
    """The first ``count`` records, in file order. This is the curve split."""

    if count < 1:
        raise ValueError("the training prefix must be positive")
    if count > len(records):
        raise ValueError("the training prefix is longer than the split")
    return records[:count]


def write_curve_split(source: Path, destination: Path, train_examples: int) -> dict:
    """Write a training prefix plus the full validation and test files."""

    train_lines = (source / SPLITS["train"]).read_text().splitlines()
    records = [json.loads(line) for line in train_lines if line]
    prefix = prefix_records(records, train_examples)
    destination.mkdir(parents=True, exist_ok=True)
    _write_jsonl(destination / SPLITS["train"], prefix)
    for split in ("validation", "test"):
        (destination / SPLITS[split]).write_bytes((source / SPLITS[split]).read_bytes())
    manifest = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "train_examples": len(prefix),
        "source": str(source),
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def records_from_texts(texts: list[str]) -> list[dict]:
    """Keep non-empty WikiText lines. Blank lines are document breaks."""

    records = []
    for text in texts:
        stripped = text.strip()
        if stripped:
            records.append({"text": stripped})
    return records


class MetricLog:
    """Collect the train and validation reports mlx-lm already emits."""

    def __init__(self) -> None:
        self.train: list[dict] = []
        self.validation: list[dict] = []

    def on_train_loss_report(self, train_info: dict) -> None:
        self.train.append(_plain(train_info))

    def on_val_loss_report(self, val_info: dict) -> None:
        row = _plain(val_info)
        if row.get("val_loss") is not None:
            row["val_perplexity"] = perplexity(float(row["val_loss"]))
        self.validation.append(row)


def prepare_dataset(destination: Path) -> dict:
    """Write train, valid, and test JSONL for the pinned WikiText revision."""

    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    if _manifest_matches(manifest_path):
        return json.loads(manifest_path.read_text())
    counts = {}
    for split, filename in SPLITS.items():
        texts = _fetch_split(split)
        records = records_from_texts(texts)
        _write_jsonl(destination / filename, records)
        counts[split] = len(records)
    manifest = {
        "dataset": DATASET_ID,
        "config": DATASET_CONFIG,
        "revision": DATASET_REVISION,
        "counts": counts,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_lora(
    output: Path,
    data_dir: Path,
    iters: int | None = None,
    epochs: int = CURVE_EPOCHS,
    test_batches: int = -1,
    batch_size: int = CURVE_BATCH_SIZE,
    max_seq_length: int = 512,
    learning_rate: float = 1e-4,
    seed: int = 0,
    train_examples: int | None = CURVE_TRAIN_EXAMPLES,
    grad_accumulation_steps: int = CURVE_ACCUMULATION,
) -> dict:
    """Train a QLoRA adapter and report test perplexity before and after.

    ``iters`` fixes the step count. When it is omitted, the run is
    ``epochs`` passes over the training lines. ``train_examples`` keeps the
    first lines of the training split and leaves validation and test whole.
    ``test_batches=-1`` scores the whole test split.
    """

    if batch_size < 1 or max_seq_length < 2 or epochs < 1 or grad_accumulation_steps < 1:
        raise ValueError("batch size, sequence length, epochs, and accumulation must be positive")
    if iters is not None and iters < 1:
        raise ValueError("iters must be positive")
    if train_examples is not None and train_examples < 1:
        raise ValueError("train examples must be positive")
    if test_batches == 0 or test_batches < -1:
        raise ValueError("test batches must be -1 or positive")
    try:
        return _run_lora(
            output,
            data_dir,
            iters,
            epochs,
            test_batches,
            batch_size,
            max_seq_length,
            learning_rate,
            seed,
            train_examples,
            grad_accumulation_steps,
        )
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "model": TRAIN_MODEL,
        }


def _run_lora(
    output: Path,
    data_dir: Path,
    iters: int | None,
    epochs: int,
    test_batches: int,
    batch_size: int,
    max_seq_length: int,
    learning_rate: float,
    seed: int,
    train_examples: int | None,
    grad_accumulation_steps: int,
) -> dict:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from mlx_lm.lora import train_model
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import evaluate
    from mlx_lm.utils import load

    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = prepare_dataset(data_dir)
    train_dir = data_dir
    if train_examples is not None:
        train_dir = data_dir.parent / f"{data_dir.name}-curve-{train_examples}"
        manifest = {**manifest, **write_curve_split(data_dir, train_dir, train_examples)}
    revision = MODEL_REVISIONS[TRAIN_MODEL]
    model, tokenizer = load(TRAIN_MODEL, revision=revision)
    args = _namespace(
        train_dir,
        output / "adapters",
        iters=1,
        epochs=epochs,
        test_batches=test_batches,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        learning_rate=learning_rate,
        seed=seed,
        grad_accumulation_steps=grad_accumulation_steps,
    )
    train_set, valid_set, test_set = load_dataset(args, tokenizer)
    steps = iters if iters is not None else epoch_steps(len(train_set), batch_size, epochs)
    args.iters = steps
    args.save_every = steps
    args.steps_per_eval = max(steps // epochs, 1)
    args.steps_per_report = 100 if steps >= 1000 else 10
    baseline = _measure(model, test_set, args, evaluate, CacheDataset)
    history = MetricLog()
    train_model(args, model, train_set, valid_set, history)
    adapted = _measure(model, test_set, args, evaluate, CacheDataset)
    trainable = sum(array.size for _, array in tree_flatten(model.trainable_parameters()))
    mx.eval(model.parameters())
    report = {
        "status": "ok",
        "mlx_lm": MLX_LM_PIN,
        "dataset": manifest,
        "model": TRAIN_MODEL,
        "model_revision": revision,
        "fine_tune_type": "lora",
        "quantized_base": True,
        "lora": {"rank": 8, "scale": 20.0, "dropout": 0.0, "num_layers": -1},
        "optimizer": "adamw",
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
        "max_seq_length": max_seq_length,
        "epochs": None if iters is not None else epochs,
        "iters": steps,
        "train_examples": len(train_set),
        "seed": seed,
        "grad_checkpoint": True,
        "val_batches": -1,
        "test_batches": test_batches,
        "seconds": time.perf_counter() - started,
        "trainable_parameters": int(trainable),
        "baseline_test": baseline,
        "adapted_test": adapted,
        "history": {"train": history.train, "validation": history.validation},
        "adapter_path": str(output / "adapters"),
    }
    return report


def _measure(model, dataset, args, evaluate, cache_dataset) -> dict:
    loss = float(
        evaluate(
            model=model,
            dataset=cache_dataset(dataset),
            batch_size=args.batch_size,
            num_batches=args.test_batches,
            max_seq_length=args.max_seq_length,
        )
    )
    scored = len(dataset) // args.batch_size if args.test_batches < 0 else args.test_batches
    return {"loss": loss, "perplexity": perplexity(loss), "batches": scored}


def _namespace(
    data_dir: Path,
    adapter_path: Path,
    iters: int,
    epochs: int,
    test_batches: int,
    batch_size: int,
    max_seq_length: int,
    learning_rate: float,
    seed: int,
    grad_accumulation_steps: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=TRAIN_MODEL,
        data=str(data_dir),
        mask_prompt=False,
        train=True,
        test=True,
        seed=seed,
        num_layers=-1,
        fine_tune_type="lora",
        lora_parameters={"rank": 8, "dropout": 0.0, "scale": 20.0},
        resume_adapter_file=None,
        adapter_path=str(adapter_path),
        save_every=iters,
        batch_size=batch_size,
        iters=iters,
        epochs=epochs,
        val_batches=-1,
        learning_rate=learning_rate,
        steps_per_report=10,
        steps_per_eval=iters,
        max_seq_length=max_seq_length,
        grad_checkpoint=True,
        grad_accumulation_steps=grad_accumulation_steps,
        clear_cache_threshold=0,
        lr_schedule=None,
        optimizer="adamw",
        optimizer_config={"adamw": {}},
        test_batches=test_batches,
    )


def _manifest_matches(path: Path) -> bool:
    if not path.is_file():
        return False
    manifest = json.loads(path.read_text())
    if manifest.get("revision") != DATASET_REVISION:
        return False
    return all((path.parent / name).is_file() for name in SPLITS.values())


def _fetch_split(split: str) -> list[str]:
    first = _page(split, 0)
    total = int(first["num_rows_total"])
    texts = [_row_text(row) for row in first["rows"]]
    offsets = list(range(PAGE_ROWS, total, PAGE_ROWS))
    if not offsets:
        return texts
    with ThreadPoolExecutor(max_workers=8) as pool:
        pages = pool.map(lambda offset: _page(split, offset), offsets)
    for page in pages:
        texts.extend(_row_text(row) for row in page["rows"])
    return texts


def _page(split: str, offset: int) -> dict:
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": split,
            "offset": offset,
            "length": PAGE_ROWS,
        }
    )
    url = f"https://datasets-server.huggingface.co/rows?{query}"
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def _row_text(row: dict) -> str:
    return str(row["row"]["text"])


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def _plain(info: dict) -> dict:
    plain = {}
    for key, value in info.items():
        if hasattr(value, "item"):
            value = value.item()
        plain[key] = value
    return plain
