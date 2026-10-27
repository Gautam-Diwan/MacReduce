"""QLoRA local steps for one federated client.

The GPU path calls the installed ``mlx_lm`` trainer and returns the LoRA
adapter. The Neural Engine path refuses to run until a stack is importable
and a Qwen3 adapter step is compiled for it. A missing stack is not replaced
by the GPU. Importing this module does not load a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .federated import Client, LocalResult, PeerRequired, run_federated
from .finetune import (
    CURVE_ACCUMULATION,
    CURVE_BATCH_SIZE,
    CURVE_EPOCHS,
    CURVE_TRAIN_EXAMPLES,
    SPLITS,
    TRAIN_MODEL,
    epoch_steps,
    perplexity,
    prefix_records,
)
from .serving import MODEL_REVISIONS

LORA = {"rank": 8, "dropout": 0.0, "scale": 20.0}
ANE_STACKS = ("coreml", "aneforge", "orion", "maderix")


class BackendUnavailable(RuntimeError):
    """The requested Neural Engine stack cannot take a QLoRA step."""


def write_client_shard(directory: Path, records: list[dict]) -> None:
    """Write one client's WikiText lines. Validation and test stay empty here."""

    if not records:
        raise ValueError("a client shard is empty")
    directory.mkdir(parents=True, exist_ok=True)
    _write_jsonl(directory / "train.jsonl", records)
    (directory / "valid.jsonl").write_text("")
    (directory / "test.jsonl").write_text("")


def load_curve_records(data_dir: Path, train_examples: int = CURVE_TRAIN_EXAMPLES) -> list[dict]:
    """The first training lines of a prepared WikiText directory."""

    path = data_dir / SPLITS["train"]
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    return prefix_records(records, train_examples)


def require_ane(stack: str) -> None:
    """Reject a Neural Engine step that would silently become a GPU step."""

    from .train_plan import probe_backends

    if stack not in ANE_STACKS:
        raise ValueError(f"unknown Neural Engine stack {stack}")
    rows = {row["backend"]: row for row in probe_backends()}
    status = rows[stack]["status"]
    if status != "importable":
        raise BackendUnavailable(f"{stack} is {status}")
    raise BackendUnavailable(f"{stack} imports, and no Qwen3 LoRA step is compiled for it")


class MlxLoraTrainer:
    """Local AdamW on the pinned Qwen3-0.6B adapter.

    ``setup`` loads the model. Optimizer moments are recreated for each local
    call so they follow the broadcast adapter instead of the previous round.
    """

    def __init__(
        self,
        work_dir: Path,
        learning_rate: float = 1e-4,
        batch_size: int = CURVE_BATCH_SIZE,
        max_seq_length: int = 512,
        seed: int = 0,
        grad_accumulation_steps: int = CURVE_ACCUMULATION,
    ) -> None:
        if batch_size < 1 or max_seq_length < 2 or grad_accumulation_steps < 1:
            raise ValueError("batch size, sequence length, and accumulation must be positive")
        if learning_rate <= 0:
            raise ValueError("learning rate must be positive")
        self.work_dir = Path(work_dir)
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_seq_length = max_seq_length
        self.seed = seed
        self.grad_accumulation_steps = grad_accumulation_steps
        self._model = None
        self._tokenizer = None

    def setup(self) -> dict[str, np.ndarray]:
        import mlx.core as mx
        from mlx_lm.tuner.trainer import grad_checkpoint
        from mlx_lm.tuner.utils import linear_to_lora_layers
        from mlx_lm.utils import load

        self.work_dir.mkdir(parents=True, exist_ok=True)
        model, tokenizer = load(TRAIN_MODEL, revision=MODEL_REVISIONS[TRAIN_MODEL])
        model.freeze()
        linear_to_lora_layers(model, -1, LORA, use_dora=False)
        grad_checkpoint(model.layers[0])
        mx.eval(model.parameters())
        self._model = model
        self._tokenizer = tokenizer
        return self.snapshot()

    def snapshot(self) -> dict[str, np.ndarray]:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        if self._model is None:
            raise RuntimeError("setup has not loaded the model")
        state = {}
        for name, value in tree_flatten(self._model.trainable_parameters()):
            mx.eval(value)
            state[name] = np.array(value)
        if not state:
            raise RuntimeError("the model has no trainable adapter")
        return state

    def load_state(self, state: dict[str, np.ndarray]) -> None:
        import mlx.core as mx

        current = self.snapshot()
        if set(current) != set(state):
            raise ValueError("adapter keys do not match the model")
        pairs = [(name, mx.array(np.asarray(value))) for name, value in state.items()]
        self._model.load_weights(pairs, strict=False)
        mx.eval(self._model.parameters())

    def frozen_nbytes(self) -> int:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        if self._model is None:
            raise RuntimeError("setup has not loaded the model")
        trainable = {name for name, _ in tree_flatten(self._model.trainable_parameters())}
        total = 0
        for name, value in tree_flatten(self._model.parameters()):
            if name in trainable:
                continue
            mx.eval(value)
            total += int(value.nbytes)
        return total

    def local_update(
        self,
        client: Client,
        state: dict[str, np.ndarray],
        records: list[dict],
        steps: int,
    ) -> LocalResult:
        import mlx.optimizers as optim
        from mlx_lm.tuner.datasets import load_dataset
        from mlx_lm.tuner.trainer import TrainingArgs, train

        from .finetune import MetricLog, _namespace

        if steps < 1:
            raise ValueError("local steps must be positive")
        if len(records) < self.batch_size:
            raise ValueError("the client shard is smaller than the batch")
        self.load_state(state)
        shard = self.work_dir / client.name
        write_client_shard(shard, records)
        args = _namespace(
            shard,
            shard / "adapters",
            iters=steps,
            epochs=1,
            test_batches=1,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            learning_rate=self.learning_rate,
            seed=self.seed,
            grad_accumulation_steps=self.grad_accumulation_steps,
        )
        args.train = True
        args.test = False
        train_set, _, _ = load_dataset(args, self._tokenizer)
        history = MetricLog()
        train(
            self._model,
            optim.AdamW(learning_rate=self.learning_rate),
            train_set,
            val_dataset=None,
            args=TrainingArgs(
                batch_size=self.batch_size,
                iters=steps,
                val_batches=0,
                steps_per_report=1,
                steps_per_eval=steps + 1,
                steps_per_save=steps + 1,
                max_seq_length=self.max_seq_length,
                adapter_file=str(shard / "adapters.safetensors"),
                grad_checkpoint=False,
                grad_accumulation_steps=self.grad_accumulation_steps,
            ),
            training_callback=history,
        )
        return LocalResult(self.snapshot(), steps, _reported_loss(history))

    def validation_perplexity(self, state: dict[str, np.ndarray], data_dir: Path) -> float:
        from mlx_lm.tuner.datasets import CacheDataset, load_dataset
        from mlx_lm.tuner.trainer import evaluate

        from .finetune import _namespace

        self.load_state(state)
        args = _namespace(
            data_dir,
            self.work_dir / "eval",
            iters=1,
            epochs=1,
            test_batches=1,
            batch_size=self.batch_size,
            max_seq_length=self.max_seq_length,
            learning_rate=self.learning_rate,
            seed=self.seed,
            grad_accumulation_steps=self.grad_accumulation_steps,
        )
        args.train = False
        args.test = False
        _, valid, _ = load_dataset(args, self._tokenizer)
        if len(valid) < self.batch_size:
            raise ValueError("the validation split is smaller than the batch")
        loss = float(
            evaluate(
                model=self._model,
                dataset=CacheDataset(valid),
                batch_size=self.batch_size,
                num_batches=-1,
                max_seq_length=self.max_seq_length,
            )
        )
        return perplexity(loss)


def execute_federated(
    work_dir: Path,
    data_dir: Path,
    *,
    n_macs: int = 1,
    mode: str = "gpu",
    algorithm: str = "fedavg",
    topology: str = "star",
    optimizer_steps: int | None = None,
    local_steps: int = 8,
    quantize: bool = False,
    normalize: bool = False,
    ane_stack: str = "aneforge",
    train_examples: int = CURVE_TRAIN_EXAMPLES,
    seed: int = 0,
) -> dict:
    """Fine-tune the pinned adapter. A second Mac raises ``PeerRequired``."""

    from .federated import pack_state

    if n_macs > 1:
        raise PeerRequired("a second Mac is not connected")
    records = load_curve_records(data_dir, train_examples)
    trainer = MlxLoraTrainer(work_dir)
    initial = trainer.setup()
    steps = optimizer_steps
    if steps is None:
        steps = epoch_steps(len(records), trainer.batch_size, CURVE_EPOCHS)

    def local_update(client, state, shard, requested):
        if client.device in {"ane", "gpu_ane"}:
            require_ane(ane_stack)
        return trainer.local_update(client, state, shard, requested)

    report = run_federated(
        records,
        initial,
        local_update,
        n_macs=n_macs,
        mode=mode,
        algorithm=algorithm,
        topology=topology,
        optimizer_steps=steps,
        local_steps=local_steps,
        quantize=quantize,
        normalize=normalize,
        seed=seed,
        evaluate=lambda state: trainer.validation_perplexity(state, data_dir),
        eval_every=epoch_steps(len(records), trainer.batch_size, 1),
        base_nbytes=trainer.frozen_nbytes(),
        trainer="mlx_qlora",
    )
    state = report.pop("state")
    packed = pack_state(state)
    destination = Path(work_dir) / "federated-adapter.npz"
    destination.write_bytes(packed)
    report["adapter_parameters"] = sum(int(np.asarray(value).size) for value in state.values())
    report["adapter_path"] = str(destination)
    report["measured"] = False
    return report


def _write_jsonl(path: Path, records: list[dict]) -> None:
    lines = []
    for record in records:
        text = record.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("each record needs non-empty text")
        lines.append(json.dumps({"text": text}) + "\n")
    path.write_text("".join(lines))


def _reported_loss(history) -> float | None:
    if not history.train:
        return None
    row = history.train[-1]
    for key in ("train_loss", "loss"):
        if key in row:
            return float(row[key])
    return None
