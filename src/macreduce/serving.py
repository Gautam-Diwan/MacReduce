"""Serving trace, A-B-B-A comparison, and the pinned MLX-LM executor.

Unit tests use ``token_diff`` and the harness without loading a model. The
MLX-LM path imports ``mlx_lm`` only when a command asks for generation, and it
fails if that package or the pinned version is absent. CPU and ANE drafters
remain policy choices without an executor. The Core ML probe is a separate
convolution, not a draft model.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from .placement import DEPTHS, Placement, fixed

MLX_LM_PIN = "0.32.2"
TARGET_MODEL = "mlx-community/Qwen3-1.7B-4bit"
DRAFT_MODEL = "mlx-community/Qwen3-0.6B-4bit"
MODEL_REVISIONS = {
    TARGET_MODEL: "3b1b1768f8f8cf8351c712464f906e86c2b8269e",
    DRAFT_MODEL: "73e3e38d981303bc594367cd910ea6eb48349da8",
}


@dataclass(frozen=True)
class Prompt:
    segment: str
    text: str


@dataclass(frozen=True)
class Trial:
    arm: str
    segment: str
    ttft_seconds: float
    decode_tokens_per_second: float
    exact_match: bool
    generated_tokens: int


def builtin_trace() -> list[Prompt]:
    """Short, long, and code prompts. Token counts are assigned after tokenization."""

    paragraph = (
        "MacReduce places each request by measured draft cost, acceptance, "
        "and prefix-transfer cost. The placement is a hypothesis until the "
        "benchmark compares it with the best fixed configuration. "
    )
    code = (
        "def place(depth, acceptance, draft_s, verify_s):\n"
        "    if depth == 0:\n"
        "        return verify_s\n"
        "    tokens = (1 - acceptance ** (depth + 1)) / (1 - acceptance)\n"
        "    return (depth * draft_s + verify_s) / tokens\n"
    )
    return [
        Prompt("short", "Reply with one word: ready"),
        Prompt("long", paragraph * 40),
        Prompt("code", "Continue this function without changing its name.\n" + code),
    ]


def abba(cycles: int) -> list[str]:
    if cycles < 1:
        raise ValueError("cycles must be positive")
    return ["A", "B", "B", "A"] * cycles


def sweep_order(levels: tuple[int, ...] = DEPTHS, cycles: int = 1) -> list[int]:
    """Forward then reversed sweeps so a slow thermal drift does not favor one end."""

    if cycles < 1:
        raise ValueError("cycles must be positive")
    order: list[int] = []
    forward = list(levels)
    for index in range(cycles):
        order.extend(forward if index % 2 == 0 else list(reversed(forward)))
    return order


def noise_floor(values: list[float]) -> float | None:
    """Median absolute deviation. One sample has no floor; missing data is not zero."""

    if len(values) < 2:
        return None
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def token_diff(reference: tuple[int, ...], other: tuple[int, ...]) -> dict:
    matched = reference == other
    limit = min(len(reference), len(other))
    mismatch = next((index for index in range(limit) if reference[index] != other[index]), None)
    if mismatch is None and len(reference) != len(other):
        mismatch = limit
    return {
        "exact": matched,
        "first_mismatch": None if matched else mismatch,
        "reference_length": len(reference),
        "other_length": len(other),
    }


def beats(
    adaptive: list[Trial],
    static: list[Trial],
    *,
    ttft_floor: float | None,
    decode_floor: float | None,
) -> dict:
    """True only when both gains clear the measured floors and every trial matches."""

    if not adaptive or not static:
        raise ValueError("both arms need trials")
    if any(not trial.exact_match for trial in adaptive + static):
        return {
            "beats": False,
            "reason": "greedy outputs differ",
            "ttft_gain": None,
            "decode_gain": None,
        }
    if ttft_floor is None or decode_floor is None:
        return {
            "beats": False,
            "reason": "noise floor unavailable",
            "ttft_gain": None,
            "decode_gain": None,
        }
    adaptive_ttft = statistics.median(trial.ttft_seconds for trial in adaptive)
    static_ttft = statistics.median(trial.ttft_seconds for trial in static)
    adaptive_decode = statistics.median(trial.decode_tokens_per_second for trial in adaptive)
    static_decode = statistics.median(trial.decode_tokens_per_second for trial in static)
    ttft_gain = static_ttft - adaptive_ttft
    decode_gain = adaptive_decode - static_decode
    won = ttft_gain > ttft_floor and decode_gain > decode_floor
    return {
        "beats": won,
        "reason": "both gains exceed the noise floor" if won else "within noise or wrong direction",
        "ttft_gain": ttft_gain,
        "decode_gain": decode_gain,
        "ttft_floor": ttft_floor,
        "decode_floor": decode_floor,
    }


def run_abba(run, prompts: list[Prompt], cycles: int) -> list[Trial]:
    """``run(arm, prompt) -> Trial``. Arms alternate A, B, B, A for each prompt."""

    trials = []
    for prompt in prompts:
        for arm in abba(cycles):
            trial = run(arm, prompt)
            if trial.arm != arm or trial.segment != prompt.segment:
                raise ValueError("runner returned a trial for a different arm or segment")
            trials.append(trial)
    return trials


def static_arms() -> list[Placement]:
    """The sweep the adaptive policy has to beat, one depth at a time on the GPU."""

    return [fixed("none", 0, "recompute")] + [
        fixed("gpu", depth, "recompute") for depth in DEPTHS if depth
    ]


class MlxLmExecutor:
    """Greedy generation through stock ``mlx_lm``. Same-device GPU draft only."""

    def __init__(self, target: str = TARGET_MODEL, draft: str = DRAFT_MODEL) -> None:
        self.target_id = target
        self.draft_id = draft
        self._target = None
        self._tokenizer = None
        self._draft = None
        self.versions: dict[str, str] = {}

    def require_pin(self) -> dict[str, str]:
        import importlib.metadata as metadata

        try:
            found = metadata.version("mlx-lm")
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "mlx-lm is not installed; install the serve extra and do not substitute another backend"
            ) from exc
        if found != MLX_LM_PIN:
            raise RuntimeError(f"mlx-lm {found} does not match pin {MLX_LM_PIN}")
        self.versions = {"mlx-lm": found, "mlx": metadata.version("mlx")}
        return dict(self.versions)

    def load(self) -> None:
        self.require_pin()
        if self._target is not None:
            return
        from mlx_lm import load

        self._target, self._tokenizer = load(
            self.target_id, revision=MODEL_REVISIONS.get(self.target_id)
        )
        self._draft, draft_tokenizer = load(
            self.draft_id, revision=MODEL_REVISIONS.get(self.draft_id)
        )
        if getattr(self._tokenizer, "vocab_size", None) != getattr(
            draft_tokenizer, "vocab_size", None
        ):
            raise RuntimeError("target and draft tokenizers report different vocab sizes")

    def generate(self, prompt: str, *, max_tokens: int, placement: Placement) -> dict:
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if placement.engine in {"cpu", "ane"}:
            raise NotImplementedError(
                f"{placement.engine} drafter has no executor; GPU and no-draft are the implemented paths"
            )
        self.load()
        from mlx_lm import stream_generate

        assert self._target is not None and self._tokenizer is not None
        draft = None if placement.engine == "none" else self._draft
        kwargs = {"max_tokens": max_tokens}
        if draft is not None:
            kwargs["draft_model"] = draft
            kwargs["num_draft_tokens"] = placement.depth
        tokens: list[int] = []
        drafted: list[bool] = []
        text = ""
        last = None
        for response in stream_generate(self._target, self._tokenizer, prompt, **kwargs):
            tokens.append(int(response.token))
            drafted.append(bool(response.from_draft))
            text += response.text
            last = response
        if last is None:
            raise RuntimeError("mlx-lm produced no generation response")
        prompt_tokens = int(last.prompt_tokens)
        prompt_rate = float(last.prompt_tps)
        ttft = None if prompt_rate <= 0 else prompt_tokens / prompt_rate
        return {
            "tokens": tuple(tokens),
            "text": text,
            "ttft_seconds": ttft,
            "decode_tokens_per_second": float(last.generation_tps),
            "prompt_tokens": prompt_tokens,
            "generated_tokens": int(last.generation_tokens),
            "from_draft_tokens": sum(drafted),
            "peak_memory_gb": float(last.peak_memory),
            "finish_reason": last.finish_reason,
            "placement": asdict(placement),
            "versions": dict(self.versions),
            "target": self.target_id,
            "draft": self.draft_id,
        }

    def equivalence(self, prompts: list[Prompt], max_tokens: int, depth: int = 2) -> dict:
        """Token identity of greedy speculative decoding against greedy generation."""

        rows = []
        for prompt in prompts:
            plain = self.generate(
                prompt.text, max_tokens=max_tokens, placement=fixed("none", 0, "recompute")
            )
            spec = self.generate(
                prompt.text,
                max_tokens=max_tokens,
                placement=fixed("gpu", depth, "recompute"),
            )
            diff = token_diff(plain["tokens"], spec["tokens"])
            rows.append({"segment": prompt.segment, **diff})
        return {
            "exact": all(row["exact"] for row in rows),
            "depth": depth,
            "max_tokens": max_tokens,
            "rows": rows,
            "versions": dict(self.versions),
            "target": self.target_id,
            "draft": self.draft_id,
            "revisions": {
                "target": MODEL_REVISIONS.get(self.target_id),
                "draft": MODEL_REVISIONS.get(self.draft_id),
            },
        }

    def _encode(self, prompt: str):
        tokenizer = self._tokenizer
        assert tokenizer is not None
        add_special = tokenizer.bos_token is None or not prompt.startswith(
            tokenizer.bos_token or ""
        )
        return list(tokenizer.encode(prompt, add_special_tokens=add_special))

    def prefix_trial(self, prompt: str, directory, continuation_tokens: int = 4) -> dict:
        """Time full prefill against loading a saved prefix cache, and compare tokens.

        The suffix is the last 8 prompt tokens. The cache holds everything before
        that suffix. This does not measure a peer transfer.
        """

        from pathlib import Path

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.load()
        import time

        import mlx.core as mx
        from mlx_lm.generate import generate_step, stream_generate
        from mlx_lm.models.cache import load_prompt_cache, make_prompt_cache, save_prompt_cache

        ids = self._encode(prompt)
        if len(ids) <= 8:
            raise ValueError("prompt is too short to split a prefix from an 8-token suffix")
        prefix, suffix = ids[:-8], ids[-8:]
        cache = make_prompt_cache(self._target)
        started = time.perf_counter()
        for _ in generate_step(mx.array(prefix), self._target, max_tokens=0, prompt_cache=cache):
            pass
        mx.synchronize()
        prefill_seconds = time.perf_counter() - started
        path = directory / "prefix.safetensors"
        save_prompt_cache(str(path), cache, {"model": self.target_id})
        started = time.perf_counter()
        loaded = load_prompt_cache(str(path))
        mx.synchronize()
        load_seconds = time.perf_counter() - started
        kv_bytes = path.stat().st_size

        def continuations(prompt_cache):
            produced = []
            for response in stream_generate(
                self._target,
                self._tokenizer,
                suffix,
                max_tokens=continuation_tokens,
                prompt_cache=prompt_cache,
            ):
                produced.append(int(response.token))
            return tuple(produced)

        reused = continuations(loaded)
        recomputed_cache = make_prompt_cache(self._target)
        for _ in generate_step(
            mx.array(prefix), self._target, max_tokens=0, prompt_cache=recomputed_cache
        ):
            pass
        mx.synchronize()
        recomputed = continuations(recomputed_cache)
        return {
            "exact": reused == recomputed,
            "prefix_tokens": len(prefix),
            "suffix_tokens": len(suffix),
            "prefill_seconds": prefill_seconds,
            "load_seconds": load_seconds,
            "kv_bytes": kv_bytes,
            "prefill_tokens_per_second": None
            if prefill_seconds <= 0
            else len(prefix) / prefill_seconds,
            "disk_bytes_per_second": None if load_seconds <= 0 else kv_bytes / load_seconds,
            "link_bytes_per_second": None,
            "recomputed_tokens": recomputed,
            "reused_tokens": reused,
            "target": self.target_id,
            "versions": dict(self.versions),
        }

    def cpu_draft_trial(self, prompt: str, max_tokens: int, depth: int) -> dict:
        """Draft weights loaded on the CPU. The stock loop still uses one stream.

        Token identity is the caller's job. Running this does not show that the
        CPU and GPU executed at the same time.
        """

        import mlx.core as mx
        from mlx_lm import load, stream_generate

        self.require_pin()
        previous = mx.default_device()
        try:
            mx.set_default_device(mx.cpu)
            draft, draft_tokenizer = load(self.draft_id)
            mx.set_default_device(mx.gpu)
            if self._target is None:
                self._target, self._tokenizer = load(self.target_id)
            if getattr(self._tokenizer, "vocab_size", None) != getattr(
                draft_tokenizer, "vocab_size", None
            ):
                raise RuntimeError("target and draft tokenizers report different vocab sizes")
            tokens: list[int] = []
            drafted = 0
            last = None
            for response in stream_generate(
                self._target,
                self._tokenizer,
                prompt,
                max_tokens=max_tokens,
                draft_model=draft,
                num_draft_tokens=depth,
            ):
                tokens.append(int(response.token))
                drafted += int(bool(response.from_draft))
                last = response
            if last is None:
                raise RuntimeError("cpu draft produced no tokens")
            prompt_rate = float(last.prompt_tps)
            return {
                "status": "ran",
                "placement_note": "draft parameters were created on mx.cpu; one mlx-lm stream ran both models",
                "tokens": tokens,
                "from_draft_tokens": drafted,
                "ttft_seconds": None if prompt_rate <= 0 else int(last.prompt_tokens) / prompt_rate,
                "decode_tokens_per_second": float(last.generation_tps),
                "generated_tokens": int(last.generation_tokens),
                "versions": dict(self.versions),
            }
        finally:
            mx.set_default_device(previous)
