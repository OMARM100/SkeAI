"""Loss functions for SkeAI 0.2."""

from __future__ import annotations

import math
from typing import List, Sequence

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
        self._gradient: Tensor | None = None

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

        logits_data = logits._data  # type: ignore[attr-defined]
        probabilities: List[List[float]] = []
        gradient: List[List[float]] = []
        total_loss = 0.0

        for row_index in range(batch_size):
            row = logits_data[row_index]
            maximum = max(row)

            probability_row = [
                math.exp(value - maximum)
                for value in row
            ]
            total = sum(probability_row)

            if total <= 0.0 or not math.isfinite(total):
                raise ValueError("Invalid softmax normalization.")

            inverse_total = 1.0 / total
            probability_row = [
                value * inverse_total
                for value in probability_row
            ]

            target = targets[row_index]
            if not isinstance(target, int):
                raise TypeError("Class targets must be integers.")
            if target < 0 or target >= class_count:
                raise ValueError(f"Target class out of range: {target}")

            probability = max(probability_row[target], 1e-12)
            total_loss -= math.log(probability)

            probabilities.append(probability_row)
            gradient.append(probability_row[:])
            gradient[row_index][target] -= 1.0

        self._probabilities = probabilities
        self._targets = list(targets)

        scale = 1.0 / batch_size if batch_size else 0.0
        if scale:
            for row in gradient:
                for index in range(len(row)):
                    row[index] *= scale

        self._gradient = Tensor(gradient)
        return total_loss / batch_size if batch_size else 0.0

    def backward(self) -> Tensor:
        if self._gradient is None:
            raise RuntimeError("forward must be called before backward.")
        return self._gradient


__all__ = ["CrossEntropyLoss", "MeanSquaredError"]
