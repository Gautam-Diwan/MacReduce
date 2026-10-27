"""Per-request placement for drafter engine, speculation depth, and prefix source.

The cost model is a planning rule, not a measured speedup. Missing rates stay
unavailable. An ANE engine is eligible only when the caller sets
``placement_verified``; a low draft time does not imply the Neural Engine ran.
"""

from __future__ import annotations

from dataclasses import dataclass

ENGINES = ("none", "gpu", "cpu", "ane")
DEPTHS = (0, 1, 2, 4, 8)
PREFIX_SOURCES = ("recompute", "disk", "peer")


@dataclass(frozen=True)
class EngineCost:
    """One drafter implementation. ``draft_seconds_per_token`` is None when unmeasured."""

    draft_seconds_per_token: float | None = None
    available: bool = False
    placement_verified: bool = False


@dataclass(frozen=True)
class CostTable:
    """Online measurements the scheduler is allowed to use.

    Rates are seconds or bytes per second. ``None`` means the probe did not
    return a value. A missing link is not a zero-cost peer transfer.
    """

    target_seconds_per_token: float
    verify_seconds: float
    verify_seconds_per_extra_token: float
    acceptance: float | None
    engines: dict[str, EngineCost]
    prefix_tokens: int = 0
    kv_bytes: int | None = None
    prefill_tokens_per_second: float | None = None
    disk_bytes_per_second: float | None = None
    link_bytes_per_second: float | None = None

    def __post_init__(self) -> None:
        if self.target_seconds_per_token <= 0:
            raise ValueError("target_seconds_per_token must be positive")
        if self.verify_seconds < 0 or self.verify_seconds_per_extra_token < 0:
            raise ValueError("verify costs must be non-negative")
        if self.acceptance is not None and not 0 <= self.acceptance <= 1:
            raise ValueError("acceptance must be in [0, 1] or unknown")
        if self.prefix_tokens < 0:
            raise ValueError("prefix_tokens must be non-negative")
        unknown = set(self.engines) - set(ENGINES)
        if unknown:
            raise ValueError(f"unknown engines: {sorted(unknown)}")


@dataclass(frozen=True)
class Placement:
    engine: str
    depth: int
    prefix: str | None
    seconds_per_token: float | None
    prefix_seconds: float | None
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.engine not in ENGINES:
            raise ValueError(f"unknown engine {self.engine}")
        if self.depth not in DEPTHS:
            raise ValueError(f"depth must be one of {DEPTHS}")
        if self.prefix is not None and self.prefix not in PREFIX_SOURCES:
            raise ValueError(f"unknown prefix source {self.prefix}")
        if self.engine == "none" and self.depth != 0:
            raise ValueError("engine 'none' requires depth 0")
        if self.depth == 0 and self.engine != "none":
            raise ValueError("depth 0 requires engine 'none'")


def expected_tokens(depth: int, acceptance: float) -> float:
    """Draft tokens accepted geometrically, plus the target's correction token."""

    if depth not in DEPTHS:
        raise ValueError(f"depth must be one of {DEPTHS}")
    if not 0 <= acceptance <= 1:
        raise ValueError("acceptance must be in [0, 1]")
    if depth == 0 or acceptance == 1:
        return float(depth + 1)
    return (1 - acceptance ** (depth + 1)) / (1 - acceptance)


def _draft_engines(table: CostTable) -> list[tuple[str, EngineCost]]:
    chosen = []
    for name in ("gpu", "cpu", "ane"):
        cost = table.engines.get(name)
        if cost is None or not cost.available or cost.draft_seconds_per_token is None:
            continue
        if cost.draft_seconds_per_token <= 0:
            raise ValueError(f"{name} draft_seconds_per_token must be positive when set")
        if name == "ane" and not cost.placement_verified:
            continue
        chosen.append((name, cost))
    return chosen


def _depth_cost(depth: int, draft_seconds: float, table: CostTable) -> tuple[float, float]:
    tokens = expected_tokens(depth, table.acceptance if table.acceptance is not None else 0)
    if depth == 0:
        return table.target_seconds_per_token, table.target_seconds_per_token / tokens
    elapsed = (
        depth * draft_seconds + table.verify_seconds + depth * table.verify_seconds_per_extra_token
    )
    return elapsed, elapsed / tokens


def choose(table: CostTable) -> Placement:
    """Pick the measured minimum of seconds per expected token, then the prefix source."""

    reasons: list[str] = []
    best: tuple[float, str, int] | None = None
    if table.acceptance is None:
        reasons.append("acceptance unknown; speculation disabled")
        best = (table.target_seconds_per_token, "none", 0)
    else:
        _, rate = _depth_cost(0, 0, table)
        best = (rate, "none", 0)
        for name, cost in _draft_engines(table):
            for depth in DEPTHS:
                if depth == 0:
                    continue
                _, rate = _depth_cost(depth, cost.draft_seconds_per_token or 0, table)
                if rate < best[0]:
                    best = (rate, name, depth)
        ane = table.engines.get("ane")
        if ane is not None and ane.available and not ane.placement_verified:
            reasons.append("ANE cost ignored until placement_verified")
    assert best is not None
    prefix, prefix_seconds, prefix_reasons = choose_prefix(table)
    reasons.extend(prefix_reasons)
    return Placement(
        engine=best[1],
        depth=best[2],
        prefix=prefix,
        seconds_per_token=best[0],
        prefix_seconds=prefix_seconds,
        reasons=tuple(reasons),
    )


def choose_prefix(table: CostTable) -> tuple[str | None, float | None, tuple[str, ...]]:
    """Compare recompute, local disk, and peer transfer. Unavailable rates are omitted."""

    options: dict[str, float] = {}
    notes: list[str] = []
    if table.prefix_tokens == 0:
        return "recompute", 0.0, ("empty prefix",)
    if table.prefill_tokens_per_second is None:
        notes.append("prefill rate unavailable")
    elif table.prefill_tokens_per_second <= 0:
        raise ValueError("prefill_tokens_per_second must be positive when set")
    else:
        options["recompute"] = table.prefix_tokens / table.prefill_tokens_per_second
    if table.kv_bytes is None:
        notes.append("kv size unavailable")
    elif table.kv_bytes < 0:
        raise ValueError("kv_bytes must be non-negative")
    else:
        if table.disk_bytes_per_second is None:
            notes.append("disk rate unavailable")
        elif table.disk_bytes_per_second <= 0:
            raise ValueError("disk_bytes_per_second must be positive when set")
        else:
            options["disk"] = table.kv_bytes / table.disk_bytes_per_second
        if table.link_bytes_per_second is None:
            notes.append("link rate unavailable")
        elif table.link_bytes_per_second <= 0:
            raise ValueError("link_bytes_per_second must be positive when set")
        else:
            options["peer"] = table.kv_bytes / table.link_bytes_per_second
    if not options:
        return None, None, tuple(notes)
    source = min(options, key=options.get)
    return source, options[source], tuple(notes)


def fixed(engine: str, depth: int, prefix: str | None) -> Placement:
    """A static baseline. It does not consult measurements."""

    return Placement(engine, depth, prefix, None, None, ("static configuration",))


def choose_device(seconds_per_forward: dict[str, float | None]) -> dict:
    """Pick the fastest device that has a positive measured forward time.

    Omitted and null times stay out of the ranking. This is the placement rule
    for one matched workload, not the speculative depth rule.
    """

    measured: dict[str, float] = {}
    for name, value in seconds_per_forward.items():
        if name not in {"cpu", "gpu", "ane"}:
            raise ValueError(f"unknown device {name}")
        if value is None:
            continue
        if value <= 0:
            raise ValueError(f"{name} seconds_per_forward must be positive when set")
        measured[name] = float(value)
    if not measured:
        return {
            "engine": None,
            "seconds_per_forward": None,
            "measured": {},
            "reason": "no measured device times",
        }
    engine = min(measured, key=measured.get)
    return {
        "engine": engine,
        "seconds_per_forward": measured[engine],
        "measured": measured,
        "reason": "minimum measured seconds per forward",
    }
