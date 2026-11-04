"""Later Qwen blocks on the Neural Engine, earlier blocks on the GPU.

A rank-8 product is under the engine's dispatch floor, and one call per
linear flushes the GPU graph. This path compiles a suffix of full decoder
blocks into one program: the frozen base is baked, and the LoRA factors are
the live inputs. The prefix stays 4-bit QLoRA on the GPU. Embedding, the
final norm, and the head stay on the GPU. A program whose device mask is not
``0x4`` raises.
"""

from __future__ import annotations

import time

import numpy as np

from .ane_forge import ANE_MASK

_ENGINE = None
_FN = None
_ORIGINAL = None
_PROJECTIONS = (
    ("self_attn", "q_proj"),
    ("self_attn", "k_proj"),
    ("self_attn", "v_proj"),
    ("self_attn", "o_proj"),
    ("mlp", "gate_proj"),
    ("mlp", "up_proj"),
    ("mlp", "down_proj"),
)


def ane_stats() -> dict:
    engine = _ENGINE
    if engine is None:
        return {"status": "absent", "qwen_suffix_blocks": 0}
    return {
        "status": "ok",
        "qwen_suffix_blocks": engine.suffix_layers,
        "gpu_blocks": engine.cut,
        "sequence": engine.max_seq,
        "device_mask": hex(ANE_MASK),
        "forward_calls": engine.calls["forward"],
        "backward_calls": engine.calls["backward"],
        "summary": engine.summary,
    }


def enable_ane_suffix(model, max_seq_length: int) -> dict:
    """Compile the longest suffix that fits on the Neural Engine and patch Qwen."""

    global _ENGINE
    if max_seq_length < 2:
        raise ValueError("max sequence length must be at least 2")
    inner = getattr(model, "model", None)
    if inner is None or type(inner).__name__ != "Qwen3Model":
        raise RuntimeError("the GPU plus Neural Engine path expects a Qwen3 model")
    if getattr(inner, "_ane_stack", None) is not None:
        return ane_stats()
    engine = SuffixEngine(model, max_seq_length)
    engine.compile()
    _ENGINE = engine
    inner._ane_stack = engine
    _patch_qwen()
    _install_function()
    return ane_stats()


class SuffixEngine:
    """One fused forward program and one fused backward program for the suffix."""

    def __init__(self, model, max_seq_length: int):
        args = model.model.args
        self.model = model
        self.max_seq = int(max_seq_length)
        self.n_layers = len(model.layers)
        self.dim = int(args.hidden_size)
        self.heads = int(args.num_attention_heads)
        self.kv_heads = int(args.num_key_value_heads)
        self.head_dim = int(args.head_dim)
        self.ffn = int(args.intermediate_size)
        self.eps = float(args.rms_norm_eps)
        self.rope_base = float(args.rope_theta)
        scaling = args.rope_scaling
        self.rope_scaling = scaling if isinstance(scaling, dict) and scaling.get("rope_type") == "llama3" else None
        self.cut = self.n_layers
        self.suffix_layers = 0
        self.forward_program = None
        self.backward_program = None
        self.specs: list[dict] = []
        self.lora_shapes: list[tuple[int, ...]] = []
        self.summary = ""
        self.calls = {"forward": 0, "backward": 0}
        self._reported: set[str] = set()
        self._cos = None
        self._sin = None
        self._bwd_hidden = None
        self._bwd_mask = None
        self._bwd_loras = ()
        self._bwd_upstream = None

    def compile(self) -> None:
        import aneforge as af

        widths = _widths(self.n_layers)
        errors = []
        for width in widths:
            try:
                self._compile_width(af, width)
                self.suffix_layers = width
                self.cut = self.n_layers - width
                self.summary = _summary(self.cut, self.n_layers, self.max_seq)
                return
            except Exception as exc:
                _release(self.forward_program)
                _release(self.backward_program)
                self.forward_program = None
                self.backward_program = None
                message = f"{width} blocks: {type(exc).__name__}: {exc}"
                errors.append(message)
                print(f"neural engine suffix did not compile ({message})", flush=True)
        raise RuntimeError("no Qwen suffix compiled for the Neural Engine:\n" + "\n".join(errors))

    def forward(self, hidden, loras):
        out, batch = self._rows(hidden, loras, None)
        self.calls["forward"] += batch
        return out

    def backward(self, hidden, loras, cotangent):
        out, batch = self._rows(hidden, loras, cotangent)
        self.calls["backward"] += batch
        return out

    def _engine_call(self, kind: str, fn):
        started = time.perf_counter()
        value = fn()
        if kind not in self._reported:
            self._reported.add(kind)
            elapsed = (time.perf_counter() - started) * 1e3
            print(f"neural engine suffix {kind} {elapsed:.1f} ms", flush=True)
        return value

    def run_model(self, inner, inputs, input_embeddings):
        from mlx_lm.models.base import create_attention_mask

        if input_embeddings is not None:
            hidden = input_embeddings
        else:
            hidden = inner.embed_tokens(inputs)
        mask = create_attention_mask(hidden, None)
        for layer in inner.layers[: self.cut]:
            hidden = layer(hidden, mask, None)
        loras = _lora_leaves(inner.layers[self.cut :])
        hidden = _FN(hidden, *loras)
        return inner.norm(hidden)

    def _compile_width(self, af, width: int) -> None:
        layers = self.model.layers[-width:]
        self.specs = [_layer_spec(layer) for layer in layers]
        self.lora_shapes = [shape for spec in self.specs for shape in spec["lora_shapes"]]
        self._cos, self._sin = _rope(self.max_seq, self.head_dim, self.rope_base, self.rope_scaling)
        print(
            f"compiling neural engine suffix: {width} blocks, sequence {self.max_seq}",
            flush=True,
        )
        hidden, mask, loras, _, out = self._tensors(af, False)
        program = af.compile(out)
        _require_ane(program, "forward")
        self.forward_program = program
        hidden, mask, loras, upstream, out = self._tensors(af, True)
        from aneforge.autograd import backward_from

        leaves = [hidden, *loras]
        grads = backward_from(upstream, out, leaves, stop=[*leaves, mask])
        from aneforge._compile import compile_multi

        program = compile_multi([grads[leaf] for leaf in leaves])
        _require_ane(program, "backward")
        self.backward_program = program
        self._bwd_hidden = hidden
        self._bwd_mask = mask
        self._bwd_loras = loras
        self._bwd_upstream = upstream

    def _tensors(self, af, with_upstream: bool):
        hidden = af.input((self.max_seq, self.dim))
        mask = af.input((self.max_seq, self.max_seq))
        loras = [af.input(shape) for shape in self.lora_shapes]
        out = _stack(hidden, mask, loras, self.specs, self._block_cfg(), self._cos, self._sin)
        upstream = af.input(out.shape) if with_upstream else None
        return hidden, mask, loras, upstream, out

    def _block_cfg(self) -> dict:
        return {
            "heads": self.heads,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "eps": self.eps,
        }

    def _rows(self, hidden, loras, cotangent):
        import mlx.core as mx

        host = _host(hidden)
        batch, length, _dim = host.shape
        if length > self.max_seq:
            raise ValueError(f"sequence {length} is longer than the compiled suffix {self.max_seq}")
        if len(loras) != len(self.lora_shapes):
            raise RuntimeError(f"expected {len(self.lora_shapes)} LoRA tensors, got {len(loras)}")
        padded = _pad(host, self.max_seq)
        factors = [_host(value) for value in loras]
        mask = _attention_mask(length, self.max_seq)
        upstream = None if cotangent is None else _pad(_host(cotangent), self.max_seq)
        rows = []
        accumulated = [np.zeros(shape, np.float32) for shape in self.lora_shapes]
        for index in range(batch):
            feed = [padded[index], mask, *factors]
            if upstream is None:
                produced = self._engine_call(
                    "forward",
                    lambda: np.asarray(self.forward_program(*feed), dtype=np.float32),
                )
                rows.append(produced.reshape(self.max_seq, self.dim))
                continue
            pieces = self._engine_call(
                "backward",
                lambda: self._run_backward(padded[index], mask, factors, upstream[index]),
            )
            rows.append(pieces[0].reshape(self.max_seq, self.dim))
            for slot, grad in enumerate(pieces[1:]):
                accumulated[slot] += grad.reshape(self.lora_shapes[slot])
        stacked = np.stack(rows, axis=0)[:, :length]
        hidden_out = mx.array(stacked).astype(hidden.dtype)
        if cotangent is None:
            return hidden_out, batch
        grads = [mx.array(value).astype(source.dtype) for value, source in zip(accumulated, loras)]
        return (hidden_out, *grads), batch

    def _run_backward(self, row, mask, factors, upstream):
        tried = []
        for target, scale in _candidate_scales(upstream):
            scaled = np.asarray(upstream * np.float32(scale), dtype=np.float32)
            if not np.isfinite(scaled).all():
                tried.append(target)
                continue
            try:
                values = self.backward_program(*self._backward_feed(row, mask, factors, scaled))
            except Exception:
                tried.append(target)
                continue
            pieces = [
                np.asarray(values[name], dtype=np.float32) / np.float32(scale)
                for _, name in self.backward_program.output_ports
            ]
            if all(np.isfinite(piece).all() for piece in pieces):
                self._note_peak(target)
                return pieces
            tried.append(target)
        raise RuntimeError(f"neural engine backward is not finite at peaks {tried}")

    def _note_peak(self, target: float) -> None:
        if target == _BACKWARD_PEAKS[0]:
            return
        key = f"peak-{target}"
        if key in self._reported:
            return
        self._reported.add(key)
        print(f"neural engine backward peak {target}", flush=True)

    def _backward_feed(self, row, mask, factors, upstream):
        known = {
            id(self._bwd_hidden): row,
            id(self._bwd_mask): mask,
            id(self._bwd_upstream): upstream,
        }
        for tensor, value in zip(self._bwd_loras, factors):
            known[id(tensor)] = value
        feed = []
        for tensor in self.backward_program.input_tensors:
            if id(tensor) in known:
                feed.append(known[id(tensor)])
                continue
            value = np.asarray(tensor.attrs["value"], dtype=np.float32)
            feed.append(value.reshape(tensor.shape))
        return feed


def _widths(n_layers: int) -> list[int]:
    """How many trailing blocks to fuse onto the Neural Engine.

    On this M4 a two-block backward overflowed and a four-block backward
    failed to execute. One full block stays finite, and its compile is above
    the dispatch floor.
    """

    if n_layers < 1:
        raise ValueError("the model has no decoder blocks")
    return [1]


def _summary(cut: int, n_layers: int, sequence: int) -> str:
    if cut == 0:
        ane = f"blocks 0-{n_layers - 1}"
    elif n_layers - cut == 1:
        ane = f"block {cut}"
    else:
        ane = f"blocks {cut}-{n_layers - 1}"
    if cut == 0:
        return f"qwen {ane} on the neural engine, embedding and head on the gpu, sequence {sequence}"
    return (
        f"qwen blocks 0-{cut - 1} on the gpu, {ane} on the neural engine, sequence {sequence}"
    )


def _patch_qwen() -> None:
    global _ORIGINAL
    from mlx_lm.models.qwen3 import Qwen3Model

    if getattr(Qwen3Model, "_macreduce_suffix", False):
        return
    _ORIGINAL = Qwen3Model.__call__

    def _call(self, inputs, cache=None, input_embeddings=None):
        engine = getattr(self, "_ane_stack", None)
        if engine is None:
            return _ORIGINAL(self, inputs, cache, input_embeddings)
        if cache is not None:
            raise RuntimeError("the neural engine suffix is the training prefill, not a cached decode")
        return engine.run_model(self, inputs, input_embeddings)

    Qwen3Model.__call__ = _call
    Qwen3Model._macreduce_suffix = True


def _install_function() -> None:
    global _FN
    if _FN is not None:
        return
    import mlx.core as mx

    @mx.custom_function
    def suffix_hidden(hidden, *loras):
        return _ENGINE.forward(hidden, loras)

    @suffix_hidden.vjp
    def _suffix_vjp(primals, cotangent, output):
        del output
        hidden, *loras = primals
        return _ENGINE.backward(hidden, loras, cotangent)

    _FN = suffix_hidden


def _stack(hidden, mask, loras, specs, cfg, cos, sin):
    cursor = 0
    value = hidden
    for spec in specs:
        value, cursor = _block(value, mask, loras, cursor, spec, cfg, cos, sin)
    if cursor != len(loras):
        raise RuntimeError("LoRA inputs were not consumed by the suffix")
    return value


def _block(hidden, mask, loras, cursor, spec, cfg, cos, sin):
    import aneforge as af

    heads = cfg["heads"]
    kv_heads = cfg["kv_heads"]
    head_dim = cfg["head_dim"]
    seq = hidden.shape[0]
    groups = heads // kv_heads
    normed = hidden.rms_norm(spec["attn_norm"], cfg["eps"])
    query, cursor = _project(normed, spec["weights"]["q"], spec["scales"][0], loras, cursor)
    key, cursor = _project(normed, spec["weights"]["k"], spec["scales"][1], loras, cursor)
    value, cursor = _project(normed, spec["weights"]["v"], spec["scales"][2], loras, cursor)
    query = query.reshape(seq, heads, head_dim)
    key = key.reshape(seq, kv_heads, head_dim)
    value = value.reshape(seq, kv_heads, head_dim)
    query = query.reshape(seq * heads, head_dim).rms_norm(spec["q_norm"], cfg["eps"]).reshape(seq, heads, head_dim)
    key = key.reshape(seq * kv_heads, head_dim).rms_norm(spec["k_norm"], cfg["eps"]).reshape(seq, kv_heads, head_dim)
    query = query.transpose((1, 0, 2)).reshape(1, heads, seq, head_dim)
    key = key.transpose((1, 0, 2)).reshape(1, kv_heads, seq, head_dim)
    value = value.transpose((1, 0, 2)).reshape(1, kv_heads, seq, head_dim)
    query = af.llm.rope(query, cos, sin)
    key = af.llm.rope(key, cos, sin)
    key = _repeat_kv(key, groups)
    value = _repeat_kv(value, groups)
    scores = (query @ key.transpose((0, 1, 3, 2))) * (head_dim**-0.5) + mask
    mixed = (scores.softmax(-1) @ value).reshape(heads, seq, head_dim)
    mixed = mixed.transpose((1, 0, 2)).reshape(seq, heads * head_dim)
    attended, cursor = _project(mixed, spec["weights"]["o"], spec["scales"][3], loras, cursor)
    hidden = hidden + attended
    normed = hidden.rms_norm(spec["mlp_norm"], cfg["eps"])
    gate, cursor = _project(normed, spec["weights"]["gate"], spec["scales"][4], loras, cursor)
    up, cursor = _project(normed, spec["weights"]["up"], spec["scales"][5], loras, cursor)
    down, cursor = _project(gate.silu() * up, spec["weights"]["down"], spec["scales"][6], loras, cursor)
    return hidden + down, cursor


def _project(hidden, weight, scale, loras, cursor):
    left = loras[cursor]
    right = loras[cursor + 1]
    base = hidden.linear(weight)
    update = ((hidden @ left) @ right) * float(scale)
    return base + update, cursor + 2


def _repeat_kv(key, groups: int):
    if groups == 1:
        return key
    _batch, kv_heads, seq, head_dim = key.shape
    matrix = np.repeat(np.eye(kv_heads, dtype=np.float16), groups, axis=0)
    mixed = key.reshape(kv_heads, seq, head_dim).transpose((1, 2, 0)).linear(matrix)
    return mixed.transpose((2, 0, 1)).reshape(1, kv_heads * groups, seq, head_dim)


def _layer_spec(layer) -> dict:
    weights = {}
    scales = []
    shapes = []
    names = ("q", "k", "v", "o", "gate", "up", "down")
    for name, proj in zip(names, _projections(layer)):
        weights[name] = _dense(proj)
        scales.append(float(proj.scale))
        shapes.append(tuple(int(dim) for dim in proj.lora_a.shape))
        shapes.append(tuple(int(dim) for dim in proj.lora_b.shape))
    return {
        "weights": weights,
        "scales": scales,
        "lora_shapes": shapes,
        "attn_norm": _vector(layer.input_layernorm.weight),
        "mlp_norm": _vector(layer.post_attention_layernorm.weight),
        "q_norm": _vector(layer.self_attn.q_norm.weight),
        "k_norm": _vector(layer.self_attn.k_norm.weight),
    }


def _projections(layer):
    found = []
    for parent, name in _PROJECTIONS:
        module = getattr(layer, parent)
        found.append(getattr(module, name))
    return found


def _lora_leaves(layers):
    leaves = []
    for layer in layers:
        for proj in _projections(layer):
            leaves.append(proj.lora_a)
            leaves.append(proj.lora_b)
    return leaves


def _dense(proj) -> np.ndarray:
    import mlx.core as mx

    linear = proj.linear if hasattr(proj, "linear") else proj
    if hasattr(linear, "scales"):
        weight = mx.dequantize(
            linear.weight,
            scales=linear.scales,
            biases=getattr(linear, "biases", None),
            group_size=linear.group_size,
            bits=linear.bits,
            mode=linear.mode,
        )
    else:
        weight = linear.weight
    return _vector(weight)


def _vector(array) -> np.ndarray:
    import mlx.core as mx

    mx.eval(array)
    return np.ascontiguousarray(np.array(array.astype(mx.float32), copy=True))


def _host(array) -> np.ndarray:
    import mlx.core as mx

    mx.eval(array)
    return np.array(array.astype(mx.float32), copy=True)


def _pad(array: np.ndarray, sequence: int) -> np.ndarray:
    length = array.shape[1]
    if length == sequence:
        return array
    pad = np.zeros((array.shape[0], sequence - length, array.shape[2]), np.float32)
    return np.concatenate([array, pad], axis=1)


_BACKWARD_PEAKS = (1.0, 0.25, 0.0625, 0.015625)


def _candidate_scales(upstream: np.ndarray) -> list[tuple[float, float]]:
    """Scales that bring the cotangent peak down this ladder.

    A WikiText cotangent on block 27 peaks near 1e-3. Left there, the fp16
    query gradient pointed the wrong way. Peak 1 matched float32, and peak
    0.25 did too. Peak 16 overflowed, and a later training batch overflowed
    at peak 1, so the backward tries the smaller peaks before it gives up.
    """

    if upstream.size == 0:
        return [(1.0, 1.0)]
    peak = float(np.max(np.abs(upstream)))
    if not np.isfinite(peak):
        raise RuntimeError("neural engine cotangent is not finite")
    if peak == 0.0:
        return [(1.0, 1.0)]
    return [(target, target / peak) for target in _BACKWARD_PEAKS]


def _loss_scale(upstream: np.ndarray) -> float:
    """The first backward scale, which targets a cotangent peak of 1."""

    return _candidate_scales(upstream)[0][1]


def _attention_mask(length: int, sequence: int) -> np.ndarray:
    mask = np.triu(np.full((sequence, sequence), -1e4, np.float32), 1)
    if length < sequence:
        mask[:, length:] = -1e4
    return mask


def _rope(sequence: int, head_dim: int, base: float, scaling):
    import aneforge as af

    return af.llm.rope_tables(sequence, head_dim, base=base, scaling=scaling)


def _require_ane(program, kind: str) -> None:
    masks = _device_masks(program)
    if not masks or any(mask != ANE_MASK for mask in masks):
        raise RuntimeError(f"{kind} suffix compiled off the Neural Engine: {[hex(item) for item in masks]}")


def _device_masks(program) -> list[int]:
    compiled = getattr(program, "_prog", None)
    if compiled is None:
        compiled = getattr(program, "prog", None)
    if compiled is not None and hasattr(compiled, "_device_mask"):
        return [int(compiled._device_mask)]
    masks = []
    for stage in getattr(program, "_stages", ()):
        inner = stage.get("prog") if isinstance(stage, dict) else None
        if inner is not None and hasattr(inner, "_device_mask"):
            masks.append(int(inner._device_mask))
    return masks


def _release(program) -> None:
    if program is None:
        return
    try:
        program.release()
    except Exception:
        return
