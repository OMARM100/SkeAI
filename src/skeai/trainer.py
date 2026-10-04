"""Minimal training loop for SkeAI 0.1."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from .loss import CrossEntropyLoss
from .model import Sequential
from .optimizer import SGD
from .tensor import Tensor


class Trainer:
    """Train a Sequential model with cross-entropy and SGD."""

    def __init__(
        self,
        model: Sequential,
        optimizer: SGD,
        loss: CrossEntropyLoss | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss or CrossEntropyLoss()

    def train_step(
        self,
        inputs: Tensor,
        targets: Sequence[int],
    ) -> float:
        logits = self.model.forward(inputs)
        loss_value = self.loss.forward(logits, targets)

        gradient = self.loss.backward()
        self.model.backward(gradient)
        self.optimizer.step(
            self.model.parameters(),
            self.model.gradients(),
        )

        return loss_value

    def train_batches(
        self,
        batches,
        epochs: int,
        *,
        callback: Callable[[int, int, float], None] | None = None,
    ) -> List[float]:
        """Train repeatedly over an iterable of (inputs, targets) batches."""
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        history: List[float] = []

        for epoch in range(1, epochs + 1):
            epoch_losses: List[float] = []
            for batch_index, (inputs, targets) in enumerate(batches, start=1):
                loss_value = self.train_step(inputs, targets)
                epoch_losses.append(loss_value)
                if callback is not None:
                    callback(epoch, batch_index, loss_value)

            if not epoch_losses:
                raise ValueError("Training batches cannot be empty.")

            history.append(sum(epoch_losses) / len(epoch_losses))

        return history

    def train(
        self,
        inputs: Tensor,
        targets: Sequence[int],
        epochs: int,
        *,
        callback: Callable[[int, float], None] | None = None,
    ) -> List[float]:
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        history: List[float] = []

        for epoch in range(1, epochs + 1):
            loss_value = self.train_step(inputs, targets)
            history.append(loss_value)

            if callback is not None:
                callback(epoch, loss_value)

        return history


__all__ = ["Trainer"]
