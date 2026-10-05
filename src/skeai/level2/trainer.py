"""Training utilities for SkeAI Level 2 Transformer."""

from __future__ import annotations

from typing import Sequence

from ..loss import CrossEntropyLoss
from ..optimizer import SGD
from .transformer import TinyTransformerLM


class Level2Trainer:
    """Minimal single-sequence trainer used to validate Level 2 learning."""

    def __init__(
        self,
        model: TinyTransformerLM,
        optimizer: SGD,
        loss: CrossEntropyLoss | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss or CrossEntropyLoss()

    def train_step(
        self,
        inputs: Sequence[int],
        targets: Sequence[int],
    ) -> float:
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets must have the same length.")
        if not inputs:
            raise ValueError("training sequence cannot be empty.")

        logits = self.model.forward(list(inputs))
        loss_value = self.loss.forward(logits, list(targets))
        gradients = self.model.backward(self.loss.backward())
        self.optimizer.step(self.model.parameters(), gradients)
        return loss_value

    def evaluate(
        self,
        inputs: Sequence[int],
        targets: Sequence[int],
    ) -> float:
        logits = self.model.forward(list(inputs))
        return self.loss.forward(logits, list(targets))


__all__ = ["Level2Trainer"]
