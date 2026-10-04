"""Loss functions for SkeAI 0.1.

The first language-model experiments will use categorical cross-entropy on
logits. Mean-squared error is included as a simple debugging loss.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from .tensor import Tensor


class MeanSquaredError:
    """Mean squared error with an analytical gradient."""

    def forward(self, predictions: Tensor, targets: Tensor) -> float:
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets must have the same shape.")

        values = [
            (prediction - target) ** 2
            for prediction, target in zip(
                predictions.flatten(),
                targets.flatten(),
            )
        ]
        return sum(values) / len(values) if values else 0.0

    def backward(self, predictions: Tensor, targets: Tensor) -> Tensor:
        if predictions.shape != targets.shape:
            raise ValueError("Predictions and targets must have the same shape.")

        count = predictions.size
        if count == 0:
            return Tensor([])

        scale = 2.0 / count
        return Tensor(
            [
                [
                    (prediction - target) * scale
                    for prediction, target in zip(pred_row, target_row)
                ]
                for pred_row, target_row in zip(
                    predictions.to_list(),
                    targets.to_list(),
                )
            ]
        )


class CrossEntropyLoss:
    """Softmax cross-entropy for integer class targets.

    Logits shape: [batch, classes]
    Targets: one integer class index per batch item
    """

    def __init__(self) -> None:
        self._probabilities: List[List[float]] | None = None
        self._targets: List[int] | None = None

    @staticmethod
    def _softmax(row: Sequence[float]) -> List[float]:
        if not row:
            return []

        maximum = max(row)
        exponentials = [math.exp(value - maximum) for value in row]
        total = sum(exponentials)

        if total <= 0.0 or not math.isfinite(total):
            raise ValueError("Invalid softmax normalization.")

        return [value / total for value in exponentials]

    def forward(self, logits: Tensor, targets: Sequence[int]) -> float:
        if logits.ndim != 2:
            raise ValueError("CrossEntropyLoss expects rank-2 logits.")

        batch_size, class_count = logits.shape

        if len(targets) != batch_size:
            raise ValueError("Number of targets must match batch size.")

        probabilities = [
            self._softmax(row)
            for row in logits.to_list()
        ]

        total_loss = 0.0

        for probability_row, target in zip(probabilities, targets):
            if not isinstance(target, int):
                raise TypeError("Class targets must be integers.")
            if target < 0 or target >= class_count:
                raise ValueError(f"Target class out of range: {target}")

            probability = max(probability_row[target], 1e-12)
            total_loss -= math.log(probability)

        self._probabilities = probabilities
        self._targets = list(targets)

        return total_loss / batch_size if batch_size else 0.0

    def backward(self) -> Tensor:
        if self._probabilities is None or self._targets is None:
            raise RuntimeError("forward must be called before backward.")

        batch_size = len(self._targets)
        if batch_size == 0:
            return Tensor([])

        gradient = [row[:] for row in self._probabilities]

        for row_index, target in enumerate(self._targets):
            gradient[row_index][target] -= 1.0

        scale = 1.0 / batch_size
        gradient = [
            [value * scale for value in row]
            for row in gradient
        ]

        return Tensor(gradient)


__all__ = ["CrossEntropyLoss", "MeanSquaredError"]
