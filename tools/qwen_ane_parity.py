"""Compare the real Qwen3 suffix with MLX on public WikiText batches.

Measures the backward at GPU-trained adapter states, including small LoRA-A
gradients during warmup. This is a gate, not a speed benchmark.
"""

import argparse
import json
from pathlib import Path

import aneforge as af
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models.base import create_attention_mask
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.utils import load

from macreduce.ane_numerics import scaled_rms_norm
from macreduce.ane_stack import SuffixEngine, _host, _lora_leaves
from macreduce.finetune import TRAIN_MODEL
from macreduce.memory import apply_memory_limit
from macreduce.serving import MODEL_REVISIONS


def comparison(actual, expected):
    a, b = np.asarray(actual, np.float64).ravel(), np.asarray(expected, np.float64).ravel()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    return {"reference_norm": nb, "ane_norm": na,
            "cosine": None if min(na, nb) == 0 else float(np.dot(a, b) / (na * nb)),
            "relative_l2": float(np.linalg.norm(a - b) / max(nb, 1e-30)),
            "max_abs_error": float(np.max(np.abs(a - b)))}


def main(args):
    apply_memory_limit(2 * 1024**3)
    mx.random.seed(0)
    model, tokenizer = load(TRAIN_MODEL, revision=MODEL_REVISIONS[TRAIN_MODEL])
    model.freeze()
    linear_to_lora_layers(model, -1, {"rank": 8, "scale": 2.0, "dropout": 0.0})
    initial_adapters = list(tree_flatten(model.trainable_parameters()))
    if len(args.records) != len(args.adapter_steps):
        raise ValueError("records and adapter-steps must have equal length")
    engine = SuffixEngine(model, args.sequence)
    engine.compile()
    records = [json.loads(line)["text"] for line in args.data.read_text().splitlines() if line]
    cases = []
    norm_input = af.input((args.sequence, engine.dim))
    norm_program = af.compile(scaled_rms_norm(norm_input,
        _host(model.layers[-1].input_layernorm.weight), model.args.rms_norm_eps))
    for index, step in zip(args.records, args.adapter_steps, strict=True):
        if step:
            model.load_weights(str(args.checkpoints / f"{step:07d}" / "adapters.safetensors"),
                               strict=False)
        else:
            model.load_weights(initial_adapters, strict=False)
        ids = tokenizer.encode(records[index])[:args.sequence + 1]
        tokens, targets = mx.array([ids[:-1]]), mx.array([ids[1:]])
        hidden = model.model.embed_tokens(tokens)
        mask = create_attention_mask(hidden, None)
        for layer in model.layers[:-1]:
            hidden = layer(hidden, mask, None)
        mx.eval(hidden)
        layer = model.layers[-1]
        reference = layer(hidden, mask, None)

        def tail_loss(value):
            value = model.model.norm(value)
            logits = (model.model.embed_tokens.as_linear(value) if model.args.tie_word_embeddings
                      else model.lm_head(value))
            return nn.losses.cross_entropy(logits.astype(mx.float32), targets, reduction="mean")

        cotangent = mx.grad(tail_loss)(reference)
        expected_hidden = mx.grad(lambda value: mx.sum(layer(value, mask, None) * cotangent))(hidden)
        _, gradients = nn.value_and_grad(
            layer, lambda module, value: mx.sum(module(value, mask, None) * cotangent)
        )(layer, hidden)
        names, expected = zip(*tree_flatten(gradients))
        by_name = dict(zip(names, expected))
        factor_names = [f"{parent}.{name}.lora_{factor}"
                        for parent, name in (("self_attn", "q_proj"), ("self_attn", "k_proj"),
                                             ("self_attn", "v_proj"), ("self_attn", "o_proj"),
                                             ("mlp", "gate_proj"), ("mlp", "up_proj"),
                                             ("mlp", "down_proj")) for factor in ("a", "b")]
        factors = _lora_leaves([layer])
        actual = engine.forward(hidden, factors)
        backward = engine.backward(hidden, factors, cotangent)
        produced = backward[1:]
        padded = np.zeros((args.sequence, engine.dim), np.float32)
        padded[:tokens.shape[1]] = _host(hidden)[0]
        norm_actual = np.asarray(norm_program(padded))[:tokens.shape[1]]
        norm_expected = _host(layer.input_layernorm(hidden))[0]
        row = {"record_index": index, "adapter_step": step, "tokens": tokens.shape[1],
               "hidden_abs_max": float(np.max(np.abs(_host(hidden)))),
               "cotangent_abs_max": float(np.max(np.abs(_host(cotangent)))),
               "ane_delta_norm": float(np.linalg.norm(_host(actual) - _host(hidden))),
               "reference_delta_norm": float(np.linalg.norm(_host(reference) - _host(hidden))),
               "input_normalization": comparison(norm_actual, norm_expected),
               "forward": comparison(_host(actual), _host(reference)),
               "hidden_gradient": comparison(_host(backward[0]), _host(expected_hidden)),
               "factors": {name: comparison(_host(grad), _host(by_name[name]))
                           for name, grad in zip(factor_names, produced)}}
        cases.append(row)
        print(json.dumps(row), flush=True)
    # Gate is explicit; tiny reference gradients are still reported rather than hidden.
    passed = all(row["forward"]["cosine"] is not None
                 and row["forward"]["cosine"] >= 0.99
                 and row["forward"]["relative_l2"] <= 0.02
                 and row["hidden_gradient"]["cosine"] is not None
                 and row["hidden_gradient"]["cosine"] >= 0.99
                 and row["hidden_gradient"]["relative_l2"] <= 0.15
                 and all((g["reference_norm"] == 0 and g["ane_norm"] == 0)
                         or (g["cosine"] is not None and g["cosine"] >= 0.99 and g["relative_l2"] <= 0.15)
                         for g in row["factors"].values()) for row in cases)
    result = {"status": "pass" if passed else "failed_gradient_gate", "cases": cases,
              "sequence": args.sequence,
              "criterion": "forward/hidden/LoRA cosine >=0.99; forward relative L2 <=0.02; gradient relative L2 <=0.15",
              "reference": "GPU quantized block; not an independent float64 full-block oracle",
              "scope": "real batches and saved adapter states; includes gradient to GPU prefix; not ANE-only"}
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", type=int, default=128)
    parser.add_argument("--records", type=int, nargs="+", default=[1, 5, 10])
    parser.add_argument("--adapter-steps", type=int, nargs="+", default=[0, 5, 10])
    main(parser.parse_args())
