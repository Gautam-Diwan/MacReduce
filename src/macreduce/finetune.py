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
import threading
import time
import traceback
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

# The LoRA recipe. mlx-lm pairs scale 20 with learning rate 1e-5; the
# LLaMA-Factory and ms-swift Qwen3 LoRA configs use alpha = 2r (mlx-lm scale 2
# at rank 8), learning rate 1e-4, warmup, cosine decay, and gradient clipping.
# With scale 20 and 1e-4 the adapter step was 5-10x those recipes, and small
# numerical perturbations sent otherwise identical runs into divergence.
DEFAULT_RECIPE = {
    "rank": 8,
    "scale": 2.0,
    "dropout": 0.0,
    "weight_decay": 0.01,
    "max_grad_norm": 1.0,
    "lr_schedule": {"name": "cosine_decay", "warmup": 300, "warmup_init": 1e-7, "end": 1e-6},
}


def recipe_from_config(config: dict) -> dict:
    """The LoRA recipe described by a ``configs/*.yaml`` mapping."""

    model = config.get("model") or {}
    training = config.get("training") or {}
    recipe = {
        "rank": int(model.get("lora_rank", DEFAULT_RECIPE["rank"])),
        "scale": float(model.get("lora_scale", DEFAULT_RECIPE["scale"])),
        "dropout": float(model.get("lora_dropout", DEFAULT_RECIPE["dropout"])),
        "weight_decay": float(training.get("weight_decay", DEFAULT_RECIPE["weight_decay"])),
        "max_grad_norm": training.get("max_grad_norm", DEFAULT_RECIPE["max_grad_norm"]),
        "lr_schedule": training.get("lr_schedule", DEFAULT_RECIPE["lr_schedule"]),
    }
    if recipe["max_grad_norm"] is not None:
        recipe["max_grad_norm"] = float(recipe["max_grad_norm"])
    return validate_recipe(recipe)


def validate_recipe(recipe: dict | None) -> dict:
    """Fill defaults and reject impossible values."""

    merged = {**DEFAULT_RECIPE, **(recipe or {})}
    if int(merged["rank"]) < 1:
        raise ValueError("lora rank must be positive")
    if float(merged["scale"]) <= 0:
        raise ValueError("lora scale must be positive")
    if not 0 <= float(merged["dropout"]) < 1:
        raise ValueError("lora dropout must be in [0, 1)")
    if float(merged["weight_decay"]) < 0:
        raise ValueError("weight decay must not be negative")
    if merged["max_grad_norm"] is not None and float(merged["max_grad_norm"]) <= 0:
        raise ValueError("max grad norm must be positive or null")
    schedule = merged["lr_schedule"]
    if schedule is not None:
        if not isinstance(schedule, dict) or "name" not in schedule:
            raise ValueError("lr_schedule needs a name")
        if int(schedule.get("warmup", 0)) < 0:
            raise ValueError("warmup must not be negative")
    merged["rank"] = int(merged["rank"])
    merged["scale"] = float(merged["scale"])
    merged["dropout"] = float(merged["dropout"])
    merged["weight_decay"] = float(merged["weight_decay"])
    return merged


def resolve_schedule(schedule: dict | None, learning_rate: float, iters: int) -> dict | None:
    """The mlx-lm ``build_schedule`` config for ``iters`` steps.

    ``cosine_decay`` warms up linearly from ``warmup_init`` to the peak over
    ``warmup`` steps, then decays to ``end`` by the final step. The decay
    length is derived from ``iters`` so a different epoch count or batch size
    keeps the same shape.
    """

    if schedule is None:
        return None
    if iters < 1:
        raise ValueError("iters must be positive")
    warmup = int(schedule.get("warmup", 0))
    if warmup >= iters:
        raise ValueError(f"warmup {warmup} must be shorter than {iters} steps")
    if "arguments" in schedule:
        arguments = list(schedule["arguments"])
    elif schedule["name"] == "cosine_decay":
        arguments = [float(learning_rate), iters - warmup, float(schedule.get("end", 0.0))]
    else:
        raise ValueError(f"lr_schedule {schedule['name']} needs explicit arguments")
    resolved = {"name": schedule["name"], "arguments": arguments}
    if warmup:
        resolved["warmup"] = warmup
        resolved["warmup_init"] = float(schedule.get("warmup_init", 0.0))
    return resolved
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

    def __init__(self, sink: Path | None = None) -> None:
        self.sink = sink
        self.train: list[dict] = []
        self.validation: list[dict] = []
        self._lock = threading.Lock()

    def snapshot(self) -> dict:
        with self._lock:
            return {"train": list(self.train), "validation": list(self.validation)}

    def on_train_loss_report(self, train_info: dict) -> None:
        self._keep("train", _plain(train_info))

    def on_val_loss_report(self, val_info: dict) -> None:
        row = _plain(val_info)
        if row.get("val_loss") is not None:
            row["val_perplexity"] = perplexity(float(row["val_loss"]))
        self._keep("validation", row)

    def _keep(self, kind: str, row: dict) -> None:
        with self._lock:
            getattr(self, kind).append(row)
            if self.sink is None:
                return
            self.sink.parent.mkdir(parents=True, exist_ok=True)
            with self.sink.open("a") as stream:
                stream.write(json.dumps({"kind": kind, **row}) + "\n")


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
    val_batches: int = -1,
    steps_per_report: int | None = None,
    metric_log: MetricLog | None = None,
    grad_checkpoint: bool = True,
    save_every: int | str | None = "epoch",
    resume: str | None = None,
    ane_lora: bool = False,
    recipe: dict | None = None,
) -> dict:
    """Train a QLoRA adapter and report test perplexity before and after.

    ``iters`` fixes the step count. When it is omitted, the run is
    ``epochs`` passes over the training lines. ``train_examples`` keeps the
    first lines of the training split and leaves validation and test whole.
    ``test_batches=-1`` scores the whole test split. ``recipe`` is the LoRA
    rank, scale, dropout, weight decay, gradient clip, and learning-rate
    schedule; omitted keys take :data:`DEFAULT_RECIPE`.
    """

    if batch_size < 1 or max_seq_length < 2 or epochs < 1 or grad_accumulation_steps < 1:
        raise ValueError("batch size, sequence length, epochs, and accumulation must be positive")
    if iters is not None and iters < 1:
        raise ValueError("iters must be positive")
    if train_examples is not None and train_examples < 1:
        raise ValueError("train examples must be positive")
    if test_batches == 0 or test_batches < -1:
        raise ValueError("test batches must be -1 or positive")
    if val_batches == 0 or val_batches < -1:
        raise ValueError("validation batches must be -1 or positive")
    if steps_per_report is not None and steps_per_report < 1:
        raise ValueError("steps per report must be positive")
    recipe = validate_recipe(recipe)
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
            val_batches,
            steps_per_report,
            metric_log,
            grad_checkpoint,
            save_every,
            resume,
            ane_lora,
            recipe,
        )
    except Exception as exc:
        traceback.print_exc()
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
    val_batches: int,
    steps_per_report: int | None,
    metric_log: MetricLog | None,
    grad_checkpoint: bool,
    save_every: int | str | None,
    resume: str | None,
    ane_lora: bool,
    recipe: dict,
) -> dict:
    import mlx.core as mx
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import evaluate
    from mlx_lm.utils import load

    from .train_loop import resolve_save_every, train_checked

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
        val_batches=val_batches,
        recipe=recipe,
    )
    train_set, valid_set, test_set = load_dataset(args, tokenizer)
    steps = iters if iters is not None else epoch_steps(len(train_set), batch_size, epochs)
    args.iters = steps
    schedule = resolve_schedule(recipe["lr_schedule"], learning_rate, steps)
    args.grad_checkpoint = grad_checkpoint
    args.steps_per_eval = max(steps // epochs, 1)
    args.steps_per_report = steps_per_report or (100 if steps >= 1000 else 10)
    every = resolve_save_every(save_every, steps, epochs)
    args.save_every = every
    baseline = _measure(model, test_set, args, evaluate, CacheDataset, "baseline")
    history = metric_log if metric_log is not None else MetricLog()
    resume_path = _resume_path(output / "checkpoints", resume)
    trained = train_checked(
        model,
        train_set,
        valid_set,
        args,
        history,
        output / "checkpoints",
        every,
        resume_path,
        ane_lora=ane_lora,
    )
    adapted = _measure(model, test_set, args, evaluate, CacheDataset, "test")
    trainable = trained["trainable_parameters"]
    mx.eval(model.parameters())
    report = {
        "status": "ok",
        "mlx_lm": MLX_LM_PIN,
        "dataset": manifest,
        "model": TRAIN_MODEL,
        "model_revision": revision,
        "fine_tune_type": "lora",
        "quantized_base": True,
        "lora": {
            "rank": recipe["rank"],
            "scale": recipe["scale"],
            "dropout": recipe["dropout"],
            "num_layers": args.num_layers,
        },
        "optimizer": "adamw",
        "learning_rate": learning_rate,
        "weight_decay": recipe["weight_decay"],
        "max_grad_norm": recipe["max_grad_norm"],
        "lr_schedule": schedule,
        "batch_size": batch_size,
        "grad_accumulation_steps": grad_accumulation_steps,
        "max_seq_length": max_seq_length,
        "epochs": None if iters is not None else epochs,
        "iters": steps,
        "train_examples": len(train_set),
        "seed": seed,
        "grad_checkpoint": grad_checkpoint,
        "save_every": every,
        "resumed_step": trained["resumed_step"],
        "checkpoint": trained["checkpoint"],
        "ane_lora": ane_lora,
        "ane": trained.get("ane"),
        "val_batches": val_batches,
        "test_batches": test_batches,
        "seconds": time.perf_counter() - started,
        "trainable_parameters": int(trainable),
        "baseline_test": baseline,
        "adapted_test": adapted,
        "history": {"train": history.train, "validation": history.validation},
        "adapter_path": str(output / "adapters"),
    }
    return report


def _measure(model, dataset, args, evaluate, cache_dataset, label: str) -> dict:
    from tqdm import tqdm

    scored = len(dataset) // args.batch_size
    if args.test_batches >= 0:
        scored = min(scored, args.test_batches)
    bar = tqdm(total=scored, desc=label, unit="batch", leave=False)
    try:
        loss = float(
            evaluate(
                model=model,
                dataset=cache_dataset(dataset),
                batch_size=args.batch_size,
                num_batches=args.test_batches,
                max_seq_length=args.max_seq_length,
                progress_callback=bar.update,
            )
        )
    finally:
        bar.close()
    return {"loss": loss, "perplexity": perplexity(loss), "batches": scored}


def _resume_path(checkpoint_root: Path, resume: str | None) -> Path | None:
    if resume in (None, "", "null"):
        return None
    if resume == "latest":
        pointer = checkpoint_root / "latest.json"
        if not pointer.is_file():
            raise FileNotFoundError(f"no checkpoint pointer at {pointer}")
        resume = json.loads(pointer.read_text())["path"]
    path = Path(resume)
    if not (path / "state.json").is_file():
        raise FileNotFoundError(f"no checkpoint state in {path}")
    return path


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
    val_batches: int,
    recipe: dict | None = None,
) -> SimpleNamespace:
    recipe = validate_recipe(recipe)
    return SimpleNamespace(
        model=TRAIN_MODEL,
        data=str(data_dir),
        mask_prompt=False,
        train=True,
        test=True,
        seed=seed,
        num_layers=-1,
        fine_tune_type="lora",
        lora_parameters={"rank": recipe["rank"], "dropout": recipe["dropout"], "scale": recipe["scale"]},
        resume_adapter_file=None,
        adapter_path=str(adapter_path),
        save_every=iters,
        batch_size=batch_size,
        iters=iters,
        epochs=epochs,
        val_batches=val_batches,
        learning_rate=learning_rate,
        steps_per_report=10,
        steps_per_eval=iters,
        max_seq_length=max_seq_length,
        grad_checkpoint=True,
        grad_accumulation_steps=grad_accumulation_steps,
        clear_cache_threshold=0,
        lr_schedule=recipe["lr_schedule"],
        max_grad_norm=recipe["max_grad_norm"],
        weight_decay=recipe["weight_decay"],
        optimizer="adamw",
        optimizer_config={"adamw": {"weight_decay": recipe["weight_decay"]}},
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
