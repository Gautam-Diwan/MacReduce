"""
Resident local QLoRA client; hardware imports are delayed until setup.

A process owns one logical client, optimizer and sampler. The combined mode
uses the existing explicit ANE suffix. Standalone ANE is deliberately gated.
"""

from __future__ import annotations

import json
import math
from importlib.metadata import version
from pathlib import Path

import numpy as np

from .federation_plan import shard_indices
from .lora_client import BackendUnavailable


def sample_indices(size: int, cursor: int, count: int, seed: int) -> list[int]:
    if size < 1 or cursor < 0 or count < 1:
        raise ValueError("invalid sampler size/cursor/count")
    result = []
    while len(result) < count:
        epoch, offset = divmod(cursor, size)
        order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(size)
        take = min(count - len(result), size - offset)
        result.extend(int(index) for index in order[offset:offset + take])
        cursor += take
    return result


def collate(records: list[dict], tokenizer, max_length: int):
    """Same plain-text EOS/truncation convention as pinned mlx-lm TextDataset."""
    tokens = []
    for record in records:
        ids = list(tokenizer.encode(record["text"]))
        if not ids or ids[-1] != tokenizer.eos_token_id:
            ids.append(tokenizer.eos_token_id)
        ids = ids[:max_length]
        if len(ids) < 2:
            raise ValueError("training record must encode at least two tokens")
        tokens.append(ids)
    length = min(max_length, 1 + 32 * ((max(map(len, tokens)) + 31) // 32))
    batch = np.zeros((len(tokens), length), dtype=np.int32)
    for index, ids in enumerate(tokens):
        batch[index, :len(ids)] = ids
    lengths = np.array([(0, len(ids)) for ids in tokens], dtype=np.int32)
    return batch, lengths


class ResidentLoraClient:
    def __init__(self, plan: dict, spec: dict, records: list[dict]):
        self.plan, self.spec = plan, spec
        self.records = [records[i] for i in shard_indices(plan, spec["id"], len(records))]
        self.cursor = 0
        self.updates = 0
        self.model = None

    def setup(self) -> dict:
        if self.spec["device"] == "ane":
            raise BackendUnavailable(
                "standalone ANE Qwen3 training is not implemented; split cohort cannot start. "
                "GPU+ANE suffix placement is a different backend; no fallback was selected"
            )
        required = {"mlx": "0.32.4", "mlx-lm": "0.32.2"}
        if self.spec["device"] == "gpu_ane":
            required["aneforge"] = "0.4.0"
        for package, expected in required.items():
            if version(package) != expected:
                msg = f"{package} must be pinned to {expected}; run uv sync"
                raise BackendUnavailable(msg)
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx import nn
        from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
        from mlx_lm.tuner.utils import build_schedule, linear_to_lora_layers
        from mlx_lm.utils import load

        from .finetune import resolve_schedule
        from .memory import apply_memory_limit, resolve_memory_limit

        self.mx, self.optim = mx, optim
        mx.set_default_device(mx.gpu)
        apply_memory_limit(resolve_memory_limit("auto"))
        mx.random.seed(self.plan["seed"])
        self.model, self.tokenizer = load(self.plan["model"], revision=self.plan["model_revision"])
        self.model.freeze()
        recipe = self.plan["recipe"]
        linear_to_lora_layers(self.model, -1, {
            "rank": recipe["rank"], "scale": recipe["scale"], "dropout": recipe["dropout"],
        }, use_dora=False)
        if self.spec["device"] == "gpu_ane":
            from .ane_stack import enable_ane_suffix

            enable_ane_suffix(self.model, self.plan["max_seq_length"])
        if self.spec["grad_checkpoint"]:
            grad_checkpoint(self.model.layers[0])
        schedule = resolve_schedule(recipe["lr_schedule"], self.plan["learning_rate"],
                                    self.plan["rounds"] * self.plan["local_steps"])
        rate = build_schedule(schedule) if schedule is not None else self.plan["learning_rate"]
        self.optimizer = optim.AdamW(learning_rate=rate, weight_decay=recipe["weight_decay"])
        self.optimizer.init(self.model.trainable_parameters())
        self.loss_and_grad = nn.value_and_grad(self.model, default_loss)
        self.model.train()
        mx.eval(self.model.parameters())
        # A real forward AND backward must succeed before a client joins a cohort.
        before = self.placement()
        loss, grads, _tokens = self._grad(self.records[:1])
        mx.eval(loss, grads)
        if not math.isfinite(float(loss)) or not self._finite(grads):
            raise RuntimeError("backend preflight returned nonfinite loss/gradients")
        after = self.placement()
        if self.spec["device"] == "gpu_ane" and not (
            after["forward_calls"] > before["forward_calls"]
            and after["backward_calls"] > before["backward_calls"]
        ):
            raise BackendUnavailable("combined preflight did not execute ANE forward and backward")
        return self.snapshot()

    def placement(self):
        if self.spec["device"] == "gpu":
            return {"device": "gpu"}
        engine = self.model.model._ane_stack
        return {"device": "gpu_ane", "ane_mask": "0x4", "suffix_layers": engine.suffix_layers,
                "forward_calls": engine.calls["forward"], "backward_calls": engine.calls["backward"]}

    def _finite(self, tree):
        from mlx.utils import tree_flatten

        return all(np.isfinite(np.asarray(value)).all() for _, value in tree_flatten(tree))

    def snapshot(self):
        from mlx.utils import tree_flatten

        self.mx.eval(self.model.trainable_parameters())
        return {name: np.asarray(value).astype(np.float32)
                for name, value in tree_flatten(self.model.trainable_parameters())}

    def load_adapter(self, state):
        from mlx.utils import tree_flatten

        current = dict(tree_flatten(self.model.trainable_parameters()))
        if set(current) != set(state):
            raise ValueError("broadcast adapter keys differ from the local model")
        pairs = []
        for key, value in state.items():
            array = np.asarray(value)
            if array.shape != current[key].shape or not np.isfinite(array).all():
                raise ValueError("broadcast adapter shape/value mismatch")
            pairs.append((key, self.mx.array(array).astype(current[key].dtype)))
        self.model.load_weights(pairs, strict=False)
        self.mx.eval(self.model.parameters())

    def _grad(self, records):
        batch, lengths = collate(records, self.tokenizer, self.plan["max_seq_length"])
        (loss, tokens), grads = self.loss_and_grad(
            self.model, self.mx.array(batch), self.mx.array(lengths)
        )
        self.mx.eval(loss, tokens, grads)
        return loss, grads, int(tokens)

    def train_round(self, start: dict) -> tuple[dict, dict]:
        from mlx.utils import tree_map

        self.load_adapter(start)
        tokens_total, loss_total = 0, 0.0
        for _ in range(self.plan["local_steps"]):
            accumulated = None
            update_tokens = 0
            for _ in range(self.spec["accumulation"]):
                indices = sample_indices(len(self.records), self.cursor, self.spec["microbatch"],
                                         self.plan["seed"])
                examples = [self.records[index] for index in indices]
                loss, grad, tokens = self._grad(examples)
                if tokens < 1 or not math.isfinite(float(loss)) or not self._finite(grad):
                    raise RuntimeError("nonfinite local update; refusing to publish")
                weighted = tree_map(lambda value, tokens=tokens: value * tokens, grad)
                accumulated = weighted if accumulated is None else tree_map(
                    lambda left, right: left + right, accumulated, weighted
                )
                update_tokens += tokens
                tokens_total += tokens
                loss_total += float(loss) * tokens
                self.cursor += len(examples)
            grad = tree_map(
                lambda value, update_tokens=update_tokens: value / update_tokens, accumulated
            )
            clip = self.plan["recipe"]["max_grad_norm"]
            if clip is not None:
                grad, _ = self.optim.clip_grad_norm(grad, clip)
            self.optimizer.update(self.model, grad)
            self.mx.eval(self.model.parameters(), self.optimizer.state)
            self.updates += 1
        after = self.snapshot()
        if not all(np.isfinite(value).all() for value in after.values()):
            raise RuntimeError("nonfinite adapter after local optimizer update")
        return after, {"tokens": tokens_total, "loss": loss_total / tokens_total,
                       "optimizer_updates": self.updates, "cursor": self.cursor,
                       "placement": self.placement()}

    def save(self, path: Path):
        from .train_loop import _arrays

        path.mkdir(parents=True, exist_ok=True)
        adapter = _arrays(self.model.trainable_parameters())
        moments, dtypes = _arrays(self.optimizer.state, dtypes=True)
        self.mx.eval(adapter, moments)
        self.mx.save_safetensors(str(path / "adapter.safetensors"), adapter)
        self.mx.save_safetensors(str(path / "optimizer.safetensors"), moments)
        (path / "trainer.json").write_text(json.dumps({
            "cursor": self.cursor, "updates": self.updates, "dtypes": dtypes,
        }))

    def restore(self, path: Path):
        from mlx.utils import tree_unflatten

        info = json.loads((path / "trainer.json").read_text())
        self.model.load_weights(str(path / "adapter.safetensors"), strict=False)
        moments = self.mx.load(str(path / "optimizer.safetensors"))
        if set(moments) != set(info["dtypes"]):
            raise ValueError("optimizer checkpoint keys differ from its manifest")
        self.optimizer.state = tree_unflatten([
            (key, value.astype(getattr(self.mx, info["dtypes"][key])))
            for key, value in moments.items()
        ])
        # Flattening omits empty frozen-module containers. Rebuild those using
        # the public init API while retaining all restored nonempty moments.
        self.optimizer.init(self.model.trainable_parameters())
        self.mx.eval(self.model.parameters(), self.optimizer.state)
        self.cursor, self.updates = info["cursor"], info["updates"]
