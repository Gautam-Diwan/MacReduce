"""Offline codec and packing invariants; no Flower or MLX."""

from pathlib import Path

import numpy as np

from macreduce.federation.plan import load_plan
from macreduce.federation.wire import decode, encode, layout, quantize_delta


def test_plan_enables_int8_packed_local_steps():
    plan = load_plan(Path("configs/federation.yaml"))
    assert plan["wire"] == "int8"
    assert plan["pack_sequences"] is True
    assert plan["local_steps"] == 8
    assert plan["keep_generations"] == 2
    by_id = {row["id"]: row for row in plan["clients"]}
    assert by_id["m4-pro-gpu"]["compute_dtype"] == "bfloat16"
    assert by_id["m4-air-gpu"]["compute_dtype"] == "bfloat16"
    assert by_id["m1-air-gpu"]["compute_dtype"] == "float16"
    combined = load_plan(Path("configs/federation.yaml"), "gpu_ane")
    assert all(row["compute_dtype"] == "float16" for row in combined["clients"])


def test_int8_roundtrip_on_lora_shapes_and_error_feedback():
    tensors = {
        "lora_a": np.random.default_rng(0).normal(0, 1e-3, (1024, 8)).astype(np.float32),
        "lora_b": np.random.default_rng(1).normal(0, 1e-3, (8, 2048)).astype(np.float32),
    }
    schema = {name: list(value.shape) for name, value in tensors.items()}
    restored = decode(encode(tensors, "int8"), "int8", schema)
    rel = np.sqrt(sum(((restored[k] - tensors[k]) ** 2).sum() for k in tensors)
                  / sum((tensors[k] ** 2).sum() for k in tensors))
    assert rel < 0.01
    leftover = None
    sent = {k: np.zeros_like(v) for k, v in tensors.items()}
    true = {k: np.zeros_like(v) for k, v in tensors.items()}
    for _ in range(3):
        encoded, leftover = quantize_delta(tensors, leftover, "int8")
        piece = decode(encoded, "int8", schema)
        for key, value in tensors.items():
            sent[key] = sent[key] + piece[key]
            true[key] = true[key] + value
    with_fb = np.sqrt(sum(((sent[k] - true[k]) ** 2).sum() for k in tensors)
                      / sum((true[k] ** 2).sum() for k in tensors))
    plain = decode(encode(true, "int8"), "int8", schema)
    without = np.sqrt(sum(((plain[k] - true[k]) ** 2).sum() for k in tensors)
                      / sum((true[k] ** 2).sum() for k in tensors))
    assert with_fb < without
    form = layout(schema, "int8")
    packed = sum(np.zeros(form.shapes[name], form.dtypes[name]).nbytes for name in form.names)
    fp32 = sum(value.nbytes for value in tensors.values())
    assert packed < fp32 / 2


def test_bf16_is_lossless_at_lora_delta_scale():
    value = np.array([[1e-4, -2e-4]], dtype=np.float32)
    restored = decode(encode({"w": value}, "bf16"), "bf16", {"w": [1, 2]})
    rel = np.linalg.norm(restored["w"] - value) / np.linalg.norm(value)
    assert rel < 0.01
