"""Small, dependency-free benchmarking helpers for SkeAI."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable


@dataclass(frozen=True)
class BenchmarkResult:
    """Result for one repeated benchmark operation."""

    name: str
    repeats: int
    elapsed_seconds: float
    average_seconds: float

    @property
    def average_milliseconds(self) -> float:
        return self.average_seconds * 1000.0


def benchmark(
    name: str,
    function: Callable[[], object],
    *,
    repeats: int = 5,
    warmups: int = 1,
) -> BenchmarkResult:
    """Run a callable repeatedly and return average execution time."""
    if repeats <= 0:
        raise ValueError("repeats must be greater than zero.")
    if warmups < 0:
        raise ValueError("warmups cannot be negative.")

    for _ in range(warmups):
        function()

    start = perf_counter()
    for _ in range(repeats):
        function()

    elapsed = perf_counter() - start

    return BenchmarkResult(
        name=name,
        repeats=repeats,
        elapsed_seconds=elapsed,
        average_seconds=elapsed / repeats,
    )


__all__ = ["BenchmarkResult", "benchmark"]
