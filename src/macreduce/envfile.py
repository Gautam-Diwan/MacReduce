"""Load a local .env without printing values or overriding the shell."""

from __future__ import annotations

import os
from pathlib import Path


def env_path(start: Path | None = None) -> Path | None:
    """
    Nearest ``.env`` walking up from cwd, else the repo root.

    If ``start`` is set, only that directory is checked (tests).
    """
    if start is not None:
        candidate = Path(start).resolve() / ".env"
        return candidate if candidate.is_file() else None
    here = Path.cwd().resolve()
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
        if (directory / "pyproject.toml").is_file() and (directory / "src" / "macreduce").is_dir():
            break
    fallback = Path(__file__).resolve().parents[2] / ".env"
    return fallback if fallback.is_file() else None


def load_project_env(start: Path | None = None) -> Path | None:
    """Apply KEY=VALUE lines with setdefault. Returns the file that was read."""
    path = env_path(start)
    if path is None:
        return None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)
    return path
