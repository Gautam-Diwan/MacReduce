"""MLX memory budget for this Mac.

``auto`` is the GPU's recommended working set. That is the resident budget
MLX reports for the device, not a fixed 8 GB cap. The wired limit is raised
to the same recommendation so those pages stay in unified memory.
"""

from __future__ import annotations


def recommended_working_set_bytes() -> int:
    """Bytes MLX recommends keeping resident on this GPU."""

    import mlx.core as mx

    size = mx.device_info().get("max_recommended_working_set_size")
    if not size:
        raise RuntimeError("MLX did not report a recommended working set")
    return int(size)


def resolve_memory_limit(value: int | str | None) -> int:
    """Turn ``auto`` into the recommended working set. Other values pass through."""

    if value in (None, "auto"):
        return recommended_working_set_bytes()
    limit = int(value)
    if limit < 1:
        raise ValueError("memory limit must be positive")
    return limit


def apply_memory_limit(limit: int) -> int:
    """Raise the wired limit to the recommendation and cap the MLX allocator."""

    import mlx.core as mx
    from mlx_lm.utils import maybe_set_recommended_wired_limit

    if limit < 1:
        raise ValueError("memory limit must be positive")
    maybe_set_recommended_wired_limit()
    mx.set_memory_limit(limit)
    mx.clear_cache()
    mx.reset_peak_memory()
    return limit
