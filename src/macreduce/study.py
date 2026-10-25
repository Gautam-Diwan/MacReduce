"""Depth sweep, adaptive choice, and the local prefix decision.

Records are plain dictionaries so a fresh process can write one trial and the
parent can score it. A missing rate stays absent. The peer arm is not implied
by a local disk measurement.
"""

from __future__ import annotations

import statistics
from collections import defaultdict

from .placement import DEPTHS, CostTable, EngineCost, choose, choose_prefix
from .serving import Prompt, Trial, beats, builtin_trace, noise_floor, token_diff


def acceptance_from_counts(from_draft: int, generated: int, depth: int) -> float | None:
    """Invert the geometric draft model. Undetermined counts stay unknown."""

    if depth not in DEPTHS or depth == 0 or generated <= 0 or from_draft < 0:
        return None
    rounds = generated - from_draft
    if rounds <= 0:
        return None
    per_round = from_draft / rounds
    if per_round >= depth:
        return 1.0
    low, high = 0.0, 1.0
    for _ in range(40):
        mid = (low + high) / 2
        if mid == 1:
            expected = float(depth)
        else:
            expected = mid * (1 - mid**depth) / (1 - mid)
        if expected < per_round:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def draft_seconds(
    decode_tokens_per_second: float,
    generated: int,
    from_draft: int,
    depth: int,
    target_seconds: float,
) -> float | None:
    """Estimate one draft step from a fused speculative loop.

    The verify pass is charged as one target step. A non-positive remainder
    means this approximation does not fit, and the caller must not invent a cost.
    """

    rounds = generated - from_draft
    if (
        depth not in DEPTHS
        or depth == 0
        or rounds <= 0
        or generated <= 0
        or decode_tokens_per_second <= 0
        or target_seconds <= 0
    ):
        return None
    per_round = (generated / decode_tokens_per_second) / rounds
    step = (per_round - target_seconds) / depth
    if step <= 0:
        return None
    return step


def annotate_exact(records: list[dict]) -> list[dict]:
    """Mark speculative tokens against the first no-draft token sequence of that segment."""

    references: dict[str, tuple[int, ...]] = {}
    for record in records:
        if record.get("engine") == "none" and record.get("depth") == 0 and "tokens" in record:
            references.setdefault(record["segment"], tuple(record["tokens"]))
    annotated = []
    for record in records:
        row = dict(record)
        reference = references.get(record["segment"])
        if reference is None or "tokens" not in record:
            row["exact_match"] = False
            row["first_mismatch"] = None
        else:
            diff = token_diff(reference, tuple(record["tokens"]))
            row["exact_match"] = diff["exact"]
            row["first_mismatch"] = diff["first_mismatch"]
        annotated.append(row)
    return annotated


def _group(records: list[dict]):
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("status") not in (None, "ok"):
            continue
        grouped[(record["segment"], record["engine"], int(record["depth"]))].append(record)
    return grouped


def depth_table(records: list[dict]) -> dict:
    """Median time-to-first-token and decode rate for exact GPU or no-draft trials."""

    table: dict[str, dict[str, dict]] = {}
    for (segment, engine, depth), rows in _group(records).items():
        if engine not in {"none", "gpu"}:
            continue
        exact = [row for row in rows if row.get("exact_match")]
        entry = {
            "n": len(rows),
            "exact_n": len(exact),
            "ttft_seconds": _median(exact, "ttft_seconds"),
            "decode_tokens_per_second": _median(exact, "decode_tokens_per_second"),
            "ttft_floor": noise_floor(
                [row["ttft_seconds"] for row in exact if row.get("ttft_seconds") is not None]
            ),
            "decode_floor": noise_floor(
                [
                    row["decode_tokens_per_second"]
                    for row in exact
                    if row.get("decode_tokens_per_second") is not None
                ]
            ),
        }
        table.setdefault(segment, {})[str(depth)] = entry
    return table


def _median(rows: list[dict], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None
    return statistics.median(values)


def hindsight_depth(table_for_segment: dict, metric: str) -> int | None:
    """Best exact depth. Decode rate is maximized. Time-to-first-token is minimized."""

    best_depth = None
    best_value = None
    for depth in DEPTHS:
        entry = table_for_segment.get(str(depth))
        if entry is None or entry["exact_n"] == 0:
            continue
        value = entry["decode_tokens_per_second" if metric == "decode" else "ttft_seconds"]
        if value is None:
            continue
        better = best_value is None or (
            value > best_value if metric == "decode" else value < best_value
        )
        if better:
            best_depth, best_value = depth, value
    return best_depth


def _calibration_rows(records: list[dict], segment: str) -> list[dict]:
    rows = [
        row
        for row in records
        if row["segment"] == segment and row.get("exact_match") and row.get("pass") == "calibrate"
    ]
    if rows:
        return rows
    return [row for row in records if row["segment"] == segment and row.get("exact_match")]


def cost_table_for_segment(
    records: list[dict], segment: str, prefix: dict | None
) -> CostTable | None:
    """Fit one segment. Returns None when the no-draft decode rate was not measured."""

    rows = _calibration_rows(records, segment)
    base = [
        row
        for row in rows
        if row.get("engine") == "none"
        and int(row.get("depth", -1)) == 0
        and row.get("decode_tokens_per_second")
    ]
    if not base:
        return None
    target_rate = statistics.median(row["decode_tokens_per_second"] for row in base)
    target_seconds = 1 / target_rate
    probe = [
        row
        for row in rows
        if row.get("engine") == "gpu"
        and int(row.get("depth", 0)) > 0
        and row.get("from_draft_tokens") is not None
    ]
    probe.sort(key=lambda row: int(row["depth"]))
    acceptance = None
    fitted_draft = None
    probe_depth = None
    if probe:
        sample = probe[-1]
        probe_depth = int(sample["depth"])
        acceptance = acceptance_from_counts(
            int(sample["from_draft_tokens"]), int(sample["generated_tokens"]), probe_depth
        )
        if sample.get("decode_tokens_per_second"):
            fitted_draft = draft_seconds(
                float(sample["decode_tokens_per_second"]),
                int(sample["generated_tokens"]),
                int(sample["from_draft_tokens"]),
                probe_depth,
                target_seconds,
            )
    engines = {}
    if fitted_draft is not None:
        engines["gpu"] = EngineCost(fitted_draft, True, True)
    prefix_tokens = 0 if prefix is None else int(prefix.get("prefix_tokens") or 0)
    return CostTable(
        target_seconds_per_token=target_seconds,
        verify_seconds=target_seconds,
        verify_seconds_per_extra_token=0.0,
        acceptance=acceptance,
        engines=engines,
        prefix_tokens=prefix_tokens,
        kv_bytes=None if prefix is None else prefix.get("kv_bytes"),
        prefill_tokens_per_second=None
        if prefix is None
        else prefix.get("prefill_tokens_per_second"),
        disk_bytes_per_second=None if prefix is None else prefix.get("disk_bytes_per_second"),
        link_bytes_per_second=None,
    )


def score_segment(records: list[dict], segment: str, prefix: dict | None) -> dict:
    """Choose from the calibration pass and score the later evaluation pass."""

    table = depth_table([row for row in records if row["segment"] == segment])
    segment_table = table.get(segment, {})
    fitted = cost_table_for_segment(records, segment, prefix)
    placement = None if fitted is None else choose(fitted)
    decode_best = hindsight_depth(segment_table, "decode")
    ttft_best = hindsight_depth(segment_table, "ttft")
    evaluation = [
        row for row in records if row["segment"] == segment and row.get("pass") == "evaluate"
    ]
    if not evaluation:
        evaluation = [row for row in records if row["segment"] == segment]

    def arm(depth: int | None, engine: str) -> list[Trial]:
        if depth is None:
            return []
        return [
            Trial(
                "arm",
                segment,
                row["ttft_seconds"],
                row["decode_tokens_per_second"],
                bool(row.get("exact_match")),
                int(row["generated_tokens"]),
            )
            for row in evaluation
            if int(row["depth"]) == depth
            and row.get("engine") == engine
            and row.get("ttft_seconds") is not None
            and row.get("decode_tokens_per_second") is not None
        ]

    adaptive_engine = "none" if placement is None else placement.engine
    adaptive_depth = 0 if placement is None else placement.depth
    if adaptive_engine not in {"none", "gpu"}:
        adaptive_engine, adaptive_depth = "none", 0
    hindsight_engine = "none" if decode_best in (None, 0) else "gpu"
    adaptive_trials = arm(adaptive_depth, adaptive_engine)
    hindsight_trials = arm(decode_best, hindsight_engine)
    comparison = None
    if adaptive_trials and hindsight_trials:
        comparison = beats(
            adaptive_trials,
            hindsight_trials,
            ttft_floor=noise_floor([trial.ttft_seconds for trial in hindsight_trials]),
            decode_floor=noise_floor(
                [trial.decode_tokens_per_second for trial in hindsight_trials]
            ),
        )
    prefix_choice = None
    if prefix is not None and prefix.get("exact") and fitted is not None:
        source, seconds, reasons = choose_prefix(fitted)
        prefix_choice = {"source": source, "seconds": seconds, "reasons": reasons}
    return {
        "depths": segment_table,
        "hindsight_decode_depth": decode_best,
        "hindsight_ttft_depth": ttft_best,
        "adaptive": None
        if placement is None
        else {
            "engine": placement.engine,
            "depth": placement.depth,
            "prefix": placement.prefix,
            "seconds_per_token": placement.seconds_per_token,
            "reasons": placement.reasons,
        },
        "comparison": comparison,
        "same_depth_as_hindsight": placement is not None and placement.depth == decode_best,
        "prefix_choice": prefix_choice,
        "note": (
            "One depth chosen before the evaluation pass cannot beat the best depth on that "
            "pass. A tie means the cost model selected the hindsight depth. Per-step depth "
            "changes are not measured here."
        ),
    }


def assign_passes(records: list[dict]) -> list[dict]:
    """The first replicate of each setting calibrates. Later replicates are scored."""

    seen: set[tuple] = set()
    assigned = []
    for record in records:
        row = dict(record)
        key = (row.get("segment"), row.get("engine"), row.get("depth"))
        row["pass"] = "evaluate" if key in seen else "calibrate"
        seen.add(key)
        assigned.append(row)
    return assigned


def ane_gate() -> dict:
    """Record whether a Core ML drafter can be placed. Absence is not zero utilization."""

    try:
        import importlib

        importlib.import_module("coremltools")
    except ImportError:
        return {
            "placement_verified": False,
            "available": False,
            "reason": "coremltools is not installed; the draft model was not converted and no ANE power was read",
        }
    return {
        "placement_verified": False,
        "available": False,
        "reason": "coremltools imports, but this run does not convert the draft model or read an ANE compute plan",
    }


def trial_plan(prompts: list[Prompt] | None = None, cycles: int = 2) -> list[dict]:
    prompts = builtin_trace() if prompts is None else prompts
    if cycles < 2:
        raise ValueError("cycles must be at least 2 so one pass can calibrate and one can score")
    plan = []
    for cycle in range(cycles):
        depths = DEPTHS if cycle % 2 == 0 else tuple(reversed(DEPTHS))
        for prompt in prompts:
            for depth in depths:
                plan.append(
                    {
                        "segment": prompt.segment,
                        "engine": "none" if depth == 0 else "gpu",
                        "depth": depth,
                        "mode": "generate",
                    }
                )
    return plan


def render_report(records: list[dict], prefix: dict | None, cpu: dict | None) -> dict:
    annotated = assign_passes(annotate_exact(records))
    segments = []
    for segment in dict.fromkeys(row["segment"] for row in annotated):
        segments.append({"segment": segment, **score_segment(annotated, segment, prefix)})
    return {
        "segments": segments,
        "peer": {
            "status": "not_run",
            "reason": "no second Mac was available; link rate was not measured",
        },
        "ane_gate": ane_gate(),
        "cpu_draft": cpu,
        "prefix": _public_prefix(prefix),
        "trials": [
            {key: value for key, value in row.items() if key != "tokens"} for row in annotated
        ],
    }


def execute(output, cycles: int, max_tokens: int, executable: str) -> dict:
    """Run each trial in a new interpreter so compile state does not leak across arms."""

    import json
    import subprocess
    from pathlib import Path

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for index, spec in enumerate(trial_plan(cycles=cycles)):
        destination = output / f"trial-{index:03d}.json"
        print(f"trial {index + 1} {spec['segment']} {spec['engine']} depth {spec['depth']}", flush=True)
        completed = subprocess.run(
            [
                executable,
                "-m",
                "macreduce",
                "bench-trial",
                "--segment",
                spec["segment"],
                "--engine",
                spec["engine"],
                "--depth",
                str(spec["depth"]),
                "--max-tokens",
                str(max_tokens),
                "--output",
                str(destination),
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        if destination.is_file():
            records.append(json.loads(destination.read_text()))
        else:
            records.append(
                {
                    **spec,
                    "status": "error",
                    "returncode": completed.returncode,
                    "stderr": completed.stderr[-2000:],
                }
            )
    prefix_path = output / "prefix.json"
    prefix_run = subprocess.run(
        [
            executable,
            "-m",
            "macreduce",
            "bench-trial",
            "--mode",
            "prefix",
            "--segment",
            "long",
            "--output",
            str(prefix_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    prefix = (
        json.loads(prefix_path.read_text())
        if prefix_path.is_file()
        else {
            "status": "error",
            "returncode": prefix_run.returncode,
            "stderr": prefix_run.stderr[-2000:],
        }
    )
    cpu_path = output / "cpu.json"
    cpu_run = subprocess.run(
        [
            executable,
            "-m",
            "macreduce",
            "bench-trial",
            "--mode",
            "cpu",
            "--segment",
            "short",
            "--depth",
            "2",
            "--max-tokens",
            str(max_tokens),
            "--output",
            str(cpu_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    cpu = (
        json.loads(cpu_path.read_text())
        if cpu_path.is_file()
        else {
            "status": "error",
            "returncode": cpu_run.returncode,
            "stderr": cpu_run.stderr[-2000:],
        }
    )
    reference = next(
        (
            row.get("tokens")
            for row in records
            if row.get("segment") == "short"
            and row.get("engine") == "none"
            and row.get("depth") == 0
        ),
        None,
    )
    if reference is not None and cpu.get("tokens") is not None:
        cpu["exact_match"] = list(cpu["tokens"]) == list(reference)
    usable_prefix = prefix if prefix.get("status") != "error" else None
    report = render_report(records, usable_prefix, cpu)
    if prefix.get("status") == "error":
        report["prefix"] = prefix
    report["protocol"] = {
        "cycles": cycles,
        "max_tokens": max_tokens,
        "process": "one fresh interpreter per trial",
        "models": {
            "target": "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
            "draft": "mlx-community/Qwen2.5-0.5B-Instruct-4bit",
        },
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def _public_prefix(prefix: dict | None) -> dict | None:
    if prefix is None:
        return None
    hidden = {"recomputed_tokens", "reused_tokens"}
    return {key: value for key, value in prefix.items() if key not in hidden}
