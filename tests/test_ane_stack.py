import os

import numpy as np
import pytest

from macreduce.ane_stack import (
    _BACKWARD_PEAKS,
    _attention_mask,
    _candidate_scales,
    _host,
    _lora_leaves,
    _loss_scale,
    _widths,
)


def test_suffix_widths_leave_a_gpu_prefix_on_qwen():
    assert _widths(28) == [1]
    assert _widths(1) == [1]


def test_loss_scale_lifts_a_small_cotangent_to_unit_peak():
    small = np.array([1e-3, -2e-4], np.float32)
    assert _loss_scale(small) == pytest.approx(1000.0, rel=1e-3)
    assert _loss_scale(np.ones(4, np.float32)) == pytest.approx(1.0)
    assert _loss_scale(np.zeros(4, np.float32)) == 1.0
    scales = _candidate_scales(small)
    assert [target for target, _scale in scales] == list(_BACKWARD_PEAKS)
    assert scales[1][1] == pytest.approx(250.0, rel=1e-3)


def test_attention_mask_is_causal_and_hides_padding():
    mask = _attention_mask(4, 6)
    assert mask.shape == (6, 6)
    assert mask[0, 0] == 0
    assert mask[0, 1] < -100
    assert np.all(mask[:, 4:] < -100)


@pytest.mark.skipif(os.environ.get("MACREDUCE_TEST_MLX") != "1", reason="MLX test is opt-in")
def test_suffix_matches_mlx_on_one_block():
    import mlx.core as mx
    from mlx_lm.models.qwen3 import Model, ModelArgs
    from mlx_lm.tuner.utils import linear_to_lora_layers

    from macreduce.ane_stack import enable_ane_suffix

    mx.random.seed(0)
    args = ModelArgs(
        model_type="qwen3",
        hidden_size=64,
        num_hidden_layers=1,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-6,
        vocab_size=128,
        num_key_value_heads=2,
        max_position_embeddings=32,
        head_dim=16,
        tie_word_embeddings=True,
        rope_theta=10000.0,
    )
    engine = Model(args)
    reference = Model(args)
    config = {"rank": 4, "scale": 1.0, "dropout": 0.0}
    for model in (engine, reference):
        model.freeze()
        linear_to_lora_layers(model, 1, config)
    reference.update(engine.parameters())
    for layer in engine.layers:
        for proj in (
            layer.self_attn.q_proj,
            layer.self_attn.k_proj,
            layer.self_attn.v_proj,
            layer.self_attn.o_proj,
            layer.mlp.gate_proj,
            layer.mlp.up_proj,
            layer.mlp.down_proj,
        ):
            proj.linear.weight = proj.linear.weight * 0.05
            proj.lora_b = mx.random.normal(proj.lora_b.shape) * 0.02
    reference.update(engine.parameters())
    enable_ane_suffix(engine, 16)
    tokens = mx.random.randint(0, args.vocab_size, (1, 16))
    left = np.array(engine.model(tokens).astype(mx.float32))
    right = np.array(reference.model(tokens).astype(mx.float32))
    assert _cosine(left, right) > 0.98
    _assert_output_gradient(engine, tokens)


def _assert_output_gradient(model, tokens) -> None:
    """A ones cotangent of the block output must match a finite difference on the output projection."""

    from macreduce import ane_stack

    engine = ane_stack._ENGINE
    hidden = _host(model.model.embed_tokens(tokens))[0]
    factors = [_host(value) for value in _lora_leaves(model.layers)]
    mask = _attention_mask(tokens.shape[1], engine.max_seq)
    base = np.asarray(engine.forward_program(hidden, mask, *factors), dtype=np.float64)
    base = base.reshape(engine.max_seq, engine.dim)[: tokens.shape[1]]
    index = 7
    step = 1e-2
    trial = [value.copy() for value in factors]
    trial[index] = trial[index] + step
    moved = np.asarray(engine.forward_program(hidden, mask, *trial), dtype=np.float64)
    moved = moved.reshape(engine.max_seq, engine.dim)[: tokens.shape[1]]
    cotangent = np.ones_like(base)
    finite = float(np.sum((moved - base) / step * cotangent))
    upstream = np.zeros((engine.max_seq, engine.dim), np.float32)
    upstream[: tokens.shape[1]] = cotangent
    values = engine.backward_program(*engine._backward_feed(hidden, mask, factors, upstream))
    pieces = [np.asarray(values[name], dtype=np.float64) for _, name in engine.backward_program.output_ports]
    analytic = float(np.sum(pieces[1 + index]))
    assert abs(finite) > 1e-3
    assert abs(analytic - finite) / abs(finite) < 0.25


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = left.reshape(-1).astype(np.float64)
    right = right.reshape(-1).astype(np.float64)
    return float(np.dot(left, right) / (np.linalg.norm(left) * np.linalg.norm(right) + 1e-12))
