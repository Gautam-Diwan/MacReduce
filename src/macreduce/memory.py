"""MLX memory budget for this Mac.

``auto`` is the GPU's recommended working set. That is the resident budget
MLX reports for the device, not a fixed 8 GB cap. An explicit smaller budget
also bounds the per-process wired allowance and cache.
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
    """Cap MLX allocations, wired allowance, and cache to the selected budget.

    This does not cap total process memory or ANE compiler allocations.
    """

    import mlx.core as mx

    if limit < 1:
        raise ValueError("memory limit must be positive")
    mx.set_wired_limit(min(limit, recommended_working_set_bytes()))
    mx.set_memory_limit(limit)
    mx.set_cache_limit(min(limit // 8, 256 * 1024**2))
    mx.clear_cache()
    mx.reset_peak_memory()
    return limit
