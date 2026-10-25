"""A public Accelerate BLAS probe; it does not claim which matrix unit dispatched."""

import ctypes
import time

import numpy as np


def accelerate_matmul(size=512, repeats=10):
    if not 16 <= size <= 2048 or not 1 <= repeats <= 1000:
        raise ValueError("size must be 16–2048 and repeats 1–1000")
    library = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
    fn = library.cblas_sgemm
    ptr = ctypes.POINTER(ctypes.c_float)
    fn.argtypes = [ctypes.c_int] * 6 + [
        ctypes.c_float,
        ptr,
        ctypes.c_int,
        ptr,
        ctypes.c_int,
        ctypes.c_float,
        ptr,
        ctypes.c_int,
    ]
    fn.restype = None
    rng = np.random.default_rng(11)
    a, b = [rng.standard_normal((size, size)).astype(np.float32) for _ in range(2)]
    c = np.zeros_like(a)

    def run():
        fn(
            101,
            111,
            111,
            size,
            size,
            size,
            1.0,
            a.ctypes.data_as(ptr),
            size,
            b.ctypes.data_as(ptr),
            size,
            0.0,
            c.ctypes.data_as(ptr),
            size,
        )

    for _ in range(3):
        run()
    timings = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        run()
        timings.append((time.perf_counter_ns() - start) / 1e9)
    # Double-precision sampled dot products validate outputs independently of SGEMM.
    error = max(
        abs(float(c[i, j]) - float(np.dot(a[i].astype(np.float64), b[:, j].astype(np.float64))))
        for i, j in [(0, 0), (size - 1, size - 1), (3, 7)]
    )
    if error > 1e-3 * size**0.5:
        raise ValueError("Accelerate output failed numerical check")
    median = float(np.median(timings))
    return {
        "backend": "Apple Accelerate cblas_sgemm",
        "size": size,
        "dtype": "float32",
        "repeats": repeats,
        "warmup_calls": 3,
        "median_seconds": median,
        "p95_seconds": float(np.percentile(timings, 95)),
        "median_gflops": 2 * size**3 / median / 1e9,
        "sampled_max_absolute_error": error,
        "matrix_unit_dispatch": "not_verified",
        "note": "Public CPU BLAS executed. This does not prove AMX/SME/SME2 dispatch or ANE use.",
    }
