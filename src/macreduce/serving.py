"""Serving trace, A-B-B-A comparison, and the pinned MLX-LM executor.

Unit tests use ``token_diff`` and the harness without loading a model. The
MLX-LM path imports ``mlx_lm`` only when a command asks for generation, and it
fails if that package or the pinned version is absent. CPU and ANE drafters
are policy choices without an executor until a placement check exists.
"""

from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass

from .placement import DEPTHS, Placement, fixed

MLX_LM_PIN = "0.32.2"
TARGET_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DRAFT_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"


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

        self._target, self._tokenizer = load(self.target_id)
        self._draft, draft_tokenizer = load(self.draft_id)
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
        text = ""
        last = None
        for response in stream_generate(self._target, self._tokenizer, prompt, **kwargs):
            tokens.append(int(response.token))
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
        }
