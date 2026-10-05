"""Training loop and performance telemetry for SkeAI 0.2."""

from __future__ import annotations

from time import perf_counter
from typing import Callable, Dict, List, Sequence

from .loss import CrossEntropyLoss
from .model import Sequential
from .optimizer import SGD
from .tensor import Tensor


class Trainer:
    """Train a Sequential model with cross-entropy and SGD.

    Optional timing telemetry measures the hot stages without changing the
    public train_step return value.
    """

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
        self.last_step_timing: Dict[str, float] = {
            "forward_ms": 0.0,
            "loss_ms": 0.0,
            "backward_ms": 0.0,
            "optimizer_ms": 0.0,
            "total_ms": 0.0,
        }
        self.last_epoch_timing: Dict[str, float] = {
            "seconds": 0.0,
            "batches": 0.0,
            "examples": 0.0,
            "examples_per_second": 0.0,
            "average_step_ms": 0.0,
        }

    def train_step(
        self,
        inputs: Tensor,
        targets: Sequence[int],
    ) -> float:
        total_start = perf_counter()

        if self.enable_timing:
            stage_start = perf_counter()

        logits = self.model.forward(inputs)

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
        self.model.backward(gradient)

        if self.enable_timing:
            backward_ms = (perf_counter() - stage_start) * 1000.0
            stage_start = perf_counter()
        else:
            backward_ms = 0.0

        self.optimizer.step(
            self.model.parameters(),
            self.model.gradients(),
        )

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

    def train_batches(
        self,
        batches,
        epochs: int,
        *,
        callback: Callable[[int, int, float], None] | None = None,
        epoch_callback: Callable[[int, float, float, float], None] | None = None,
    ) -> List[float]:
        """Train repeatedly over an iterable of (inputs, targets) batches.

        epoch_callback receives:
            epoch, average_loss, elapsed_seconds, examples_per_second
        """
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero.")

        history: List[float] = []

        for epoch in range(1, epochs + 1):
            epoch_start = perf_counter()
            epoch_losses: List[float] = []
            epoch_examples = 0

            for batch_index, (inputs, targets) in enumerate(batches, start=1):
                loss_value = self.train_step(inputs, targets)
                epoch_losses.append(loss_value)
                epoch_examples += inputs.shape[0]

                if callback is not None:
                    callback(epoch, batch_index, loss_value)

            if not epoch_losses:
                raise ValueError("Training batches cannot be empty.")

            elapsed = perf_counter() - epoch_start
            average_loss = sum(epoch_losses) / len(epoch_losses)
            examples_per_second = (
                epoch_examples / elapsed if elapsed > 0.0 else 0.0
            )
            average_step_ms = (
                elapsed * 1000.0 / len(epoch_losses)
                if epoch_losses
                else 0.0
            )

            self.last_epoch_timing = {
                "seconds": elapsed,
                "batches": float(len(epoch_losses)),
                "examples": float(epoch_examples),
                "examples_per_second": examples_per_second,
                "average_step_ms": average_step_ms,
            }

            history.append(average_loss)

            if epoch_callback is not None:
                epoch_callback(
                    epoch,
                    average_loss,
                    elapsed,
                    examples_per_second,
                )

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
