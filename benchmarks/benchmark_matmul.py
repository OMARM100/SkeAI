"""Isolated GEMM baseline benchmark for SkeAI's native C++ engine."""

from __future__ import annotations

import argparse
import math
import platform
import random
import statistics
from time import perf_counter

from src.skeai import engine


SHAPES = (
    ("tiny", 1, 1, 1),
    ("small_rectangular", 2, 3, 4),
    ("irregular_16", 17, 19, 21),
    ("irregular_32", 31, 33, 17),
    ("square_32", 32, 32, 32),
    ("square_64", 64, 64, 64),
    ("rectangular_96x64x128", 96, 64, 128),
)


def _reference_matmul(
    left: list[float],
    right: list[float],
    rows: int,
    inner: int,
    cols: int,
) -> list[float]:
    output: list[float] = []
    for row in range(rows):
        for col in range(cols):
            total = 0.0
            for k in range(inner):
                total += left[row * inner + k] * right[k * cols + col]
            output.append(total)
    return output


def _measure(call, warmup: int, repeats: int) -> tuple[float, float, float]:
    for _ in range(warmup):
        result = call()
        del result

    measurements: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        result = call()
        measurements.append((perf_counter() - start) * 1000.0)
        del result

    return (
        statistics.median(measurements),
        min(measurements),
        max(measurements),
    )


def _max_abs_error(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Output lengths do not match.")
    return max(
        (abs(a - b) for a, b in zip(left, right)),
        default=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark and validate the C++ GEMM baseline and dispatch."
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")

    if not engine.NATIVE_AVAILABLE:
        raise RuntimeError("C++ engine is unavailable; run python -m tools.build_cpp first.")

    print("=== SkeAI Isolated C++ GEMM Baseline ===")
    print(f"platform={platform.platform()}")
    print(f"machine={platform.machine()}")
    print(f"python={platform.python_version()}")
    print(f"warmup={args.warmup}")
    print(f"repeats={args.repeats}")
    print(f"seed={args.seed}")

    rng = random.Random(args.seed)
    for name, rows, inner, cols in SHAPES:
        left_values = [rng.uniform(-1.0, 1.0) for _ in range(rows * inner)]
        right_values = [rng.uniform(-1.0, 1.0) for _ in range(inner * cols)]
        left = engine.storage_from_flat(left_values)
        right = engine.storage_from_flat(right_values)

        baseline_call = lambda: engine.matmul_baseline(
            left, right, rows, inner, inner, cols
        )
        dispatch_call = lambda: engine.matmul(
            left, right, rows, inner, inner, cols
        )

        expected = _reference_matmul(
            left_values,
            right_values,
            rows,
            inner,
            cols,
        )
        baseline_values = engine.storage_to_flat(baseline_call())
        dispatch_values = engine.storage_to_flat(dispatch_call())
        baseline_error = _max_abs_error(baseline_values, expected)
        dispatch_error = _max_abs_error(dispatch_values, expected)
        baseline_dispatch_error = _max_abs_error(
            baseline_values,
            dispatch_values,
        )

        for actual, reference in (
            (baseline_values, expected),
            (dispatch_values, expected),
        ):
            if any(
                not math.isclose(value, target, rel_tol=1e-9, abs_tol=1e-10)
                for value, target in zip(actual, reference)
            ):
                raise AssertionError(
                    f"{name}: GEMM output differs from the Python reference "
                    f"(max_abs_error={_max_abs_error(actual, reference):.12g})"
                )

        baseline_median, baseline_min, baseline_max = _measure(
            baseline_call, args.warmup, args.repeats
        )
        dispatch_median, dispatch_min, dispatch_max = _measure(
            dispatch_call, args.warmup, args.repeats
        )
        speedup = baseline_median / max(dispatch_median, 1e-12)

        print(
            f"case={name} shape={rows}x{inner}x{cols} "
            f"multiply_accumulate_terms={rows * inner * cols}"
        )
        print(
            f"  baseline_median_ms={baseline_median:.6f} "
            f"baseline_min_ms={baseline_min:.6f} "
            f"baseline_max_ms={baseline_max:.6f}"
        )
        print(
            f"  dispatch_median_ms={dispatch_median:.6f} "
            f"dispatch_min_ms={dispatch_min:.6f} "
            f"dispatch_max_ms={dispatch_max:.6f} "
            f"baseline_over_dispatch={speedup:.4f}x"
        )
        print(
            f"  baseline_max_abs_error={baseline_error:.12g} "
            f"dispatch_max_abs_error={dispatch_error:.12g} "
            f"baseline_dispatch_max_abs_error={baseline_dispatch_error:.12g}"
        )

    print("correctness=PASS")


if __name__ == "__main__":
    main()
