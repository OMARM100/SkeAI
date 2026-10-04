"""Basic neural-network layers for SkeAI 0.1.

The implementation is deliberately small and explicit. It uses SkeAI's own
Tensor class instead of a machine-learning framework.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Tuple

from .tensor import Tensor


class Dense:
    """A fully connected layer for rank-2 [batch, features] tensors."""

    def __init__(self, input_size: int, output_size: int, seed: int = 42) -> None:
        if input_size <= 0 or output_size <= 0:
            raise ValueError("Layer sizes must be positive integers.")

        rng = random.Random(seed)
        limit = math.sqrt(6.0 / (input_size + output_size))

        self.weights = Tensor(
            [
                [rng.uniform(-limit, limit) for _ in range(output_size)]
                for _ in range(input_size)
            ]
        )
        self.bias = Tensor([0.0 for _ in range(output_size)])
        self.grad_weights = Tensor(
            [[0.0 for _ in range(output_size)] for _ in range(input_size)]
        )
        self.grad_bias = Tensor([0.0 for _ in range(output_size)])
        self._cached_input: Tensor | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                "Dense.forward expects [batch, input_size] tensor."
            )

        result = inputs.matmul(self.weights)
        bias = self.bias.to_list()
        result_data = result.to_list()

        for row in result_data:
            for index in range(len(row)):  # type: ignore[arg-type]
                row[index] += bias[index]  # type: ignore[index]

        self._cached_input = inputs
        return Tensor(result_data)  # type: ignore[arg-type]

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._cached_input is None:
            raise RuntimeError("forward must be called before backward.")

        if grad_output.ndim != 2:
            raise ValueError("grad_output must be rank-2.")

        inputs = self._cached_input
        if grad_output.shape[0] != inputs.shape[0]:
            raise ValueError("Batch size mismatch in Dense.backward.")

        x = inputs.to_list()
        go = grad_output.to_list()
        w = self.weights.to_list()

        batch_size, input_size = inputs.shape
        _, output_size = grad_output.shape

        grad_w = [
            [
                sum(
                    x[batch][input_index] * go[batch][output_index]
                    for batch in range(batch_size)
                )
                for output_index in range(output_size)
            ]
            for input_index in range(input_size)
        ]

        grad_b = [
            sum(go[batch][output_index] for batch in range(batch_size))
            for output_index in range(output_size)
        ]

        grad_x = [
            [
                sum(
                    go[batch][output_index] * w[input_index][output_index]
                    for output_index in range(output_size)
                )
                for input_index in range(input_size)
            ]
            for batch in range(batch_size)
        ]

        self.grad_weights = Tensor(grad_w)
        self.grad_bias = Tensor(grad_b)
        return Tensor(grad_x)

    def parameters(self) -> Dict[str, Tensor]:
        return {
            "weights": self.weights,
            "bias": self.bias,
        }

    def gradients(self) -> Dict[str, Tensor]:
        return {
            "weights": self.grad_weights,
            "bias": self.grad_bias,
        }


class ReLU:
    """Rectified Linear Unit activation."""

    def __init__(self) -> None:
        self._cached_input: Tensor | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        self._cached_input = inputs
        return inputs.map(lambda value: max(0.0, value))

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._cached_input is None:
            raise RuntimeError("forward must be called before backward.")

        if grad_output.shape != self._cached_input.shape:
            raise ValueError("Gradient shape must match the cached input.")

        return Tensor(
            [
                [
                    grad if value > 0.0 else 0.0
                    for value, grad in zip(input_row, grad_row)
                ]
                for input_row, grad_row in zip(
                    self._cached_input.to_list(),
                    grad_output.to_list(),
                )
            ]
        )


class Tanh:
    """Hyperbolic tangent activation."""

    def __init__(self) -> None:
        self._cached_output: Tensor | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs.map(math.tanh)
        self._cached_output = output
        return output

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._cached_output is None:
            raise RuntimeError("forward must be called before backward.")

        if grad_output.shape != self._cached_output.shape:
            raise ValueError("Gradient shape must match the cached output.")

        return Tensor(
            [
                [
                    grad * (1.0 - output_value * output_value)
                    for output_value, grad in zip(output_row, grad_row)
                ]
                for output_row, grad_row in zip(
                    self._cached_output.to_list(),
                    grad_output.to_list(),
                )
            ]
        )


__all__ = ["Dense", "ReLU", "Tanh"]
