"""Training figures: loss, perplexity, power, memory, and latency.

Matplotlib writes the files. Seaborn sets the theme and draws the lines.
Power and system memory are the mactop readings at each sample. Loss, MLX
peak memory, and token rate come from the mlx-lm reports.
"""

from __future__ import annotations

import threading
from pathlib import Path

_LOCK = threading.Lock()


def write_run_figures(directory: Path, history: dict | None, samples: list[dict] | None) -> list[Path]:
    """Write every figure that has at least two points. Return the paths."""

    directory.mkdir(parents=True, exist_ok=True)
    history = history or {}
    samples = samples or []
    written: list[Path] = []
    with _LOCK:
        _use_backend()
        jobs = (
            ("loss.png", lambda ax: _loss(ax, history)),
            ("perplexity.png", lambda ax: _perplexity(ax, history)),
            ("energy.png", lambda ax: _power(ax, samples)),
            ("memory.png", lambda ax: _memory(ax, history, samples)),
            ("latency.png", lambda ax: _latency(ax, history)),
        )
        for name, draw in jobs:
            if _save(directory / name, draw):
                written.append(directory / name)
    return written


def plot_energy_comparison(path: Path, cases: dict) -> Path | None:
    """Grouped GPU and Neural Engine joules for the three one-Mac cases."""

    gpu = (cases.get("1_gpu") or {}).get("energy") or {}
    both = (cases.get("1_gpu_plus_ane") or {}).get("energy") or {}
    split = cases.get("1_gpu_and_1_ane") or {}
    rows = [
        ("1 GPU", gpu.get("gpu_joules"), gpu.get("ane_joules")),
        ("1 GPU + ANE", both.get("gpu_joules"), both.get("ane_joules")),
        ("1 GPU and 1 ANE", split.get("gpu_joules"), split.get("ane_joules")),
    ]
    if any(item[1] is None and item[2] is None for item in rows):
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        _use_backend()
        import matplotlib.pyplot as plt
        import seaborn as sns

        _theme()
        frame: dict[str, list] = {"case": [], "rail": [], "joules": []}
        for label, gpu_j, ane_j in rows:
            frame["case"].extend([label, label])
            frame["rail"].extend(["GPU", "ANE"])
            frame["joules"].extend([gpu_j or 0, ane_j or 0])
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.barplot(data=frame, x="case", y="joules", hue="rail", ax=ax)
        ax.set_title("Energy by one-Mac case")
        ax.set_xlabel("Case")
        ax.set_ylabel("Energy (J)")
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
    return path


def _use_backend() -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)


def _theme() -> None:
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="notebook")


def _save(path: Path, draw) -> bool:
    import matplotlib.pyplot as plt

    _theme()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if not draw(ax):
        plt.close(fig)
        return False
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def _pairs(rows: list[dict], x_key: str, y_key: str) -> tuple[list, list]:
    xs, ys = [], []
    for row in rows:
        x, y = row.get(x_key), row.get(y_key)
        if x is None or y is None:
            continue
        xs.append(x)
        ys.append(float(y))
    return xs, ys


def _loss(ax, history: dict) -> bool:
    import seaborn as sns

    train_x, train_y = _pairs(history.get("train") or [], "iteration", "train_loss")
    val_x, val_y = _pairs(history.get("validation") or [], "iteration", "val_loss")
    if len(train_x) < 1 and len(val_x) < 1:
        return False
    if train_x:
        sns.lineplot(x=train_x, y=train_y, ax=ax, label="Train loss", marker="o")
    if val_x:
        sns.lineplot(x=val_x, y=val_y, ax=ax, label="Validation loss", marker="o")
    ax.set_title("QLoRA loss")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean token loss")
    return True


def _perplexity(ax, history: dict) -> bool:
    import seaborn as sns

    xs, ys = _pairs(history.get("validation") or [], "iteration", "val_perplexity")
    if len(xs) < 1:
        return False
    sns.lineplot(x=xs, y=ys, ax=ax, marker="o", color="#D55E00")
    ax.set_title("Validation perplexity")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Perplexity")
    return True


def _relative_time(samples: list[dict]) -> list[dict]:
    stamped = [row for row in samples if row.get("t") is not None]
    if not stamped:
        return []
    origin = float(stamped[0]["t"])
    return [{**row, "seconds": float(row["t"]) - origin} for row in stamped]


def _power(ax, samples: list[dict]) -> bool:
    import seaborn as sns

    timed = _relative_time(samples)
    gpu_x, gpu_y = _pairs(timed, "seconds", "gpu")
    ane_x, ane_y = _pairs(timed, "seconds", "ane")
    if len(gpu_x) < 2 and len(ane_x) < 2:
        return False
    if len(gpu_x) >= 2:
        sns.lineplot(x=gpu_x, y=gpu_y, ax=ax, label="GPU")
    if len(ane_x) >= 2:
        sns.lineplot(x=ane_x, y=ane_y, ax=ax, label="ANE")
    ax.set_title("GPU and Neural Engine power")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    return True


def _memory(ax, history: dict, samples: list[dict]) -> bool:
    import seaborn as sns

    train_x, train_y = _pairs(history.get("train") or [], "iteration", "peak_memory")
    timed = _relative_time(samples)
    used = []
    for row in timed:
        if row.get("memory_used_bytes") is None:
            continue
        used.append({**row, "used_gb": float(row["memory_used_bytes"]) / 1e9})
    mem_x, mem_y = _pairs(used, "seconds", "used_gb")
    if len(train_x) < 1 and len(mem_x) < 2:
        return False
    if train_x:
        sns.lineplot(x=train_x, y=train_y, ax=ax, label="MLX peak (GB)", marker="o")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("MLX peak memory (GB)")
    if len(mem_x) >= 2:
        host = ax.twinx() if train_x else ax
        sns.lineplot(x=mem_x, y=mem_y, ax=host, label="System used (GB)", color="#009E73")
        host.set_ylabel("System memory used (GB)")
        if not train_x:
            ax.set_xlabel("Time (s)")
    ax.set_title("Memory during the run")
    return True


def _latency(ax, history: dict) -> bool:
    import seaborn as sns

    rows = []
    for row in history.get("train") or []:
        rate = row.get("iterations_per_second")
        if row.get("iteration") is None or not rate:
            continue
        rows.append({**row, "iteration_ms": 1000.0 / float(rate)})
    step_x, step_y = _pairs(rows, "iteration", "iteration_ms")
    token_x, token_y = _pairs(history.get("train") or [], "iteration", "tokens_per_second")
    val_x, val_y = _pairs(history.get("validation") or [], "iteration", "val_time")
    if len(step_x) < 1 and len(token_x) < 1 and len(val_x) < 1:
        return False
    if step_x:
        sns.lineplot(x=step_x, y=step_y, ax=ax, label="Iteration (ms)", marker="o")
        ax.set_ylabel("Iteration latency (ms)")
    if val_x:
        sns.lineplot(
            x=val_x,
            y=[value * 1000 for value in val_y],
            ax=ax,
            label="Validation (ms)",
            marker="s",
        )
    if token_x:
        host = ax.twinx() if step_x or val_x else ax
        sns.lineplot(x=token_x, y=token_y, ax=host, label="Tokens/s", color="#0072B2")
        host.set_ylabel("Tokens per second")
    ax.set_title("Training and validation latency")
    ax.set_xlabel("Iteration")
    return True
