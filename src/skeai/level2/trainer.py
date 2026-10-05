"""Training utilities for SkeAI Level 2 Transformer."""

from __future__ import annotations

import math
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
            and getattr(self.optimizer, "weight_decay", 0.0) == 0.0
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
        target_weights: Sequence[Sequence[float]] | None = None,
    ) -> float:
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets batch sizes must match.")
        if target_weights is not None and len(target_weights) != len(inputs):
            raise ValueError("target_weights batch size must match inputs.")
        if not inputs:
            raise ValueError("training batch cannot be empty.")
        if (
            engine.NATIVE_AVAILABLE
            and isinstance(self.optimizer, SGD)
            and getattr(self.optimizer, "weight_decay", 0.0) == 0.0
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
                target_weights,
            )

        total_loss = 0.0
        for index, (batch_input, batch_target) in enumerate(zip(inputs, targets)):
            weights = None if target_weights is None else target_weights[index]
            if weights is None:
                total_loss += self.train_step(batch_input, batch_target)
            else:
                total_loss += self.train_step_weighted(
                    batch_input,
                    batch_target,
                    weights,
                )
        return total_loss / len(inputs)

    def train_step_weighted(
        self,
        inputs: Sequence[int],
        targets: Sequence[int],
        target_weights: Sequence[float],
    ) -> float:
        if (
            len(inputs) != len(targets)
            or len(inputs) != len(target_weights)
        ):
            raise ValueError(
                "inputs, targets, and target_weights must have the same length."
            )
        if not inputs:
            raise ValueError("training sequence cannot be empty.")

        if (
            engine.NATIVE_AVAILABLE
            and isinstance(self.optimizer, SGD)
            and getattr(self.optimizer, "weight_decay", 0.0) == 0.0
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
                [float(weight) for weight in target_weights],
            )

        logits = self.model.forward(list(inputs)).to_list()
        weights = [float(weight) for weight in target_weights]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            raise ValueError(
                "target_weights must contain at least one positive value."
            )

        total_loss = 0.0
        for row, target in enumerate(targets):
            row_logits = logits[row]
            maximum = max(row_logits)
            exponentials = [
                math.exp(value - maximum)
                for value in row_logits
            ]
            denominator = sum(exponentials)
            probability = max(
                exponentials[target] / denominator,
                1e-12,
            )
            total_loss -= weights[row] * math.log(probability)
        return total_loss / total_weight

    def evaluate(
        self,
        inputs: Sequence[int],
        targets: Sequence[int],
    ) -> float:
        logits = self.model.forward(list(inputs))
        return self.loss.forward(logits, list(targets))

    def evaluate_weighted(
        self,
        inputs: Sequence[int],
        targets: Sequence[int],
        target_weights: Sequence[float],
    ) -> float:
        if (
            len(inputs) != len(targets)
            or len(inputs) != len(target_weights)
        ):
            raise ValueError(
                "inputs, targets, and target_weights must have the same length."
            )
        logits = self.model.forward(list(inputs)).to_list()
        weights = [float(weight) for weight in target_weights]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            raise ValueError(
                "target_weights must contain at least one positive value."
            )
        total_loss = 0.0
        for row, target in enumerate(targets):
            row_logits = logits[row]
            maximum = max(row_logits)
            exponentials = [
                math.exp(value - maximum)
                for value in row_logits
            ]
            denominator = sum(exponentials)
            probability = max(
                exponentials[target] / denominator,
                1e-12,
            )
            total_loss -= weights[row] * math.log(probability)
        return total_loss / total_weight


__all__ = ["Level2Trainer"]
