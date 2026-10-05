"""Loss functions implemented through the SkeAI C++ engine."""
from __future__ import annotations

from typing import Sequence

from . import engine
from .tensor import Tensor


class MeanSquaredError:
    def forward(self, predictions: Tensor, targets: Tensor) -> float:
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets must have the same shape.")
        return engine.mse_forward(
            predictions._storage,
            targets._storage,
        )

    def backward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets must have the same shape.")

        gradient = Tensor.zeros(predictions.shape)
        engine.mse_backward(
            predictions._storage,
            targets._storage,
            gradient._storage,
        )
        return gradient


class CrossEntropyLoss:
    """Softmax cross-entropy for integer class targets."""

    def __init__(self) -> None:
        self._gradient: Tensor | None = None
        self._targets: list[int] | None = None

    def forward(self, logits: Tensor, targets: Sequence[int]) -> float:
        if logits.ndim != 2:
            raise ValueError("CrossEntropyLoss expects rank-2 logits.")

        batch_size, class_count = logits.shape

        if len(targets) != batch_size:
            raise ValueError("Number of targets must match batch size.")

        if self._gradient is None or self._gradient.shape != logits.shape:
            self._gradient = Tensor.zeros(logits.shape)

        value = engine.cross_entropy_forward(
            logits._storage,
            targets,
            self._gradient._storage,
            batch_size,
            class_count,
        )

        self._targets = list(targets)
        return value

    def backward(self) -> Tensor:
        if self._gradient is None:
            raise RuntimeError("forward must be called before backward.")
        return self._gradient


__all__ = ["CrossEntropyLoss", "MeanSquaredError"]
