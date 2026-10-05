"""Training utilities for SkeAI Level 2 Transformer."""

from __future__ import annotations

from typing import Sequence

from .. import engine
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

        if (
            engine.NATIVE_AVAILABLE
            and isinstance(self.optimizer, SGD)
        ):
            parameters = list(self.model.parameters().values())
            return engine.transformer_train_step(
                [parameter._storage for parameter in parameters],
                list(inputs),
                list(targets),
                self.model.vocab_size,
                self.model.config.context_length,
                self.model.config.d_model,
                self.model.config.n_heads,
                self.model.config.feed_forward_size,
                self.model.config.n_layers,
                self.optimizer.learning_rate,
                self.optimizer.weight_decay,
            )

        logits = self.model.forward(list(inputs))
        loss_value = self.loss.forward(logits, list(targets))
        gradients = self.model.backward(self.loss.backward())
        self.optimizer.step(self.model.parameters(), gradients)
        return loss_value

    def train_batch(
        self,
        inputs: Sequence[Sequence[int]],
        targets: Sequence[Sequence[int]],
    ) -> float:
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets batch sizes must match.")
        if not inputs:
            raise ValueError("training batch cannot be empty.")
        if (
            engine.NATIVE_AVAILABLE
            and isinstance(self.optimizer, SGD)
        ):
            parameters = list(self.model.parameters().values())

            return engine.transformer_train_batch(
                [parameter._storage for parameter in parameters],
                inputs,
                targets,
                self.model.vocab_size,
                self.model.config.context_length,
                self.model.config.d_model,
                self.model.config.n_heads,
                self.model.config.feed_forward_size,
                self.model.config.n_layers,
                self.optimizer.learning_rate,
                self.optimizer.weight_decay,
            )

        total_loss = 0.0
        for batch_input, batch_target in zip(inputs, targets):
            total_loss += self.train_step(batch_input, batch_target)
        return total_loss / len(inputs)

    def evaluate(
        self,
        inputs: Sequence[int],
        targets: Sequence[int],
    ) -> float:
        logits = self.model.forward(list(inputs))
        return self.loss.forward(logits, list(targets))


__all__ = ["Level2Trainer"]
