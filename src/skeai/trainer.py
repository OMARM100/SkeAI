"""Training loop and performance telemetry for SkeAI."""

from __future__ import annotations

import random
from time import perf_counter
from typing import Callable, Dict, List, Sequence

from .dataset import IndexedBatch
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
        *,
        enable_timing: bool = False,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss = loss or CrossEntropyLoss()
        self.enable_timing = enable_timing
        self._parameters = model.parameters()
        self._gradients = model.gradients()
        self.last_step_timing: Dict[str, float] = {
            k: 0.0
            for k in (
                "forward_ms",
                "loss_ms",
                "backward_ms",
                "optimizer_ms",
                "total_ms",
            )
        }
        self.last_epoch_timing: Dict[str, float] = {
            k: 0.0
            for k in (
                "seconds",
                "batches",
                "examples",
                "examples_per_second",
                "average_step_ms",
            )
        }

    def _forward_batch(self, inputs: Tensor | IndexedBatch) -> Tensor:
        if isinstance(inputs, IndexedBatch):
            return self.model.forward_indexed(inputs.indices, inputs.batch_size)
        return self.model.forward(inputs)
    def train_step(self, inputs: Tensor | IndexedBatch, targets: Sequence[int] | None = None) -> float:
        total_start = perf_counter()
        if self.enable_timing:
            stage_start = perf_counter()

        if isinstance(inputs, IndexedBatch):
            targets = inputs.targets
        if targets is None:
            raise ValueError("targets are required for a dense Tensor batch.")

        logits = self._forward_batch(inputs)

        if self.enable_timing:
            forward_ms = (perf_counter() - stage_start) * 1000.0
            stage_start = perf_counter()
        else:
            forward_ms = 0.0

        loss_value = self.loss.forward(logits, targets)

        if self.enable_timing:
            loss_ms = (perf_counter() - stage_start) * 1000.0
            stage_start = perf_counter()
        else:
            loss_ms = 0.0

        gradient = self.loss.backward()
        self.model.backward(gradient, compute_input_gradient=False)

        if self.enable_timing:
            backward_ms = (perf_counter() - stage_start) * 1000.0
            stage_start = perf_counter()
        else:
            backward_ms = 0.0

        self.optimizer.step(self._parameters, self._gradients)
        total_ms = (perf_counter() - total_start) * 1000.0

        if self.enable_timing:
            optimizer_ms = (perf_counter() - stage_start) * 1000.0
            self.last_step_timing = {
                "forward_ms": forward_ms,
                "loss_ms": loss_ms,
                "backward_ms": backward_ms,
                "optimizer_ms": optimizer_ms,
                "total_ms": total_ms,
            }

        return loss_value

    def evaluate_batches(self, batches) -> float:
        """Evaluate without updating model parameters.

        The result is a true example-weighted mean, so a smaller final batch
        cannot distort validation loss more than a full batch.
        """
        total_loss = 0.0
        total_examples = 0

        for inputs, targets in batches:
            logits = self._forward_batch(inputs)
            batch_loss = self.loss.forward(logits, targets)
            batch_examples = inputs.shape[0]
            total_loss += batch_loss * batch_examples
            total_examples += batch_examples

        if total_examples == 0:
            raise ValueError("Evaluation batches cannot be empty.")

        return total_loss / total_examples

    def train_batches(
        self,
        batches,
        epochs: int,
        *,
        callback: Callable[[int, int, float], None] | None = None,
        epoch_callback: Callable[[int, float, float, float], bool | None] | None = None,
        shuffle: bool = True,
        seed: int = 1234,
    ) -> List[float]:
        """Train batches in a different order each epoch and allow early stop."""
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        batch_list = list(batches)
        if not batch_list:
            raise ValueError("Training batches cannot be empty.")

        history: List[float] = []
        rng = random.Random(seed)
        order = list(range(len(batch_list)))

        for epoch in range(1, epochs + 1):
            epoch_start = perf_counter()
            epoch_loss_total = 0.0
            epoch_examples = 0

            if shuffle:
                rng.shuffle(order)

            for batch_number, batch_index in enumerate(order, start=1):
                inputs, targets = batch_list[batch_index]
                loss_value = self.train_step(inputs, targets)

                batch_examples = inputs.shape[0]
                epoch_loss_total += loss_value * batch_examples
                epoch_examples += batch_examples

                if callback is not None:
                    callback(epoch, batch_number, loss_value)

            elapsed = perf_counter() - epoch_start
            average_loss = (
                epoch_loss_total / epoch_examples
                if epoch_examples
                else 0.0
            )
            examples_per_second = (
                epoch_examples / elapsed
                if elapsed > 0.0
                else 0.0
            )
            average_step_ms = (
                elapsed * 1000.0 / len(order)
            )

            self.last_epoch_timing = {
                "seconds": elapsed,
                "batches": float(len(order)),
                "examples": float(epoch_examples),
                "examples_per_second": examples_per_second,
                "average_step_ms": average_step_ms,
            }
            history.append(average_loss)

            should_stop = (
                bool(epoch_callback(
                    epoch,
                    average_loss,
                    elapsed,
                    examples_per_second,
                ))
                if epoch_callback is not None
                else False
            )

            if should_stop:
                break

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
