"""Basic neural-network layers for SkeAI 0.3.

Common training paths reuse validated buffers and perform their math directly
against internal storage to reduce Python object allocation overhead.
"""

from __future__ import annotations

import math
import random
from typing import Dict

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
        self._cached_output: Tensor | None = None
        self._cached_grad_input: Tensor | None = None

        self._parameter_cache = {
            "weights": self.weights,
            "bias": self.bias,
        }
        self._gradient_cache = {
            "weights": self.grad_weights,
            "bias": self.grad_bias,
        }

    def _ensure_output_buffers(self, batch_size: int) -> None:
        output_size = self.weights.shape[1]
        input_size = self.weights.shape[0]

        if self._cached_output is None or self._cached_output.shape != (
            batch_size,
            output_size,
        ):
            self._cached_output = Tensor._from_data(
                [[0.0] * output_size for _ in range(batch_size)],
                (batch_size, output_size),
            )

        if self._cached_grad_input is None or self._cached_grad_input.shape != (
            batch_size,
            input_size,
        ):
            self._cached_grad_input = Tensor._from_data(
                [[0.0] * input_size for _ in range(batch_size)],
                (batch_size, input_size),
            )

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.weights.shape[0]:
            raise ValueError(
                "Dense.forward expects [batch, input_size] tensor."
            )

        batch_size, input_size = inputs.shape
        output_size = self.weights.shape[1]
        self._ensure_output_buffers(batch_size)

        x = inputs._data  # type: ignore[attr-defined]
        w = self.weights._data  # type: ignore[attr-defined]
        b = self.bias._data  # type: ignore[attr-defined]
        result = self._cached_output._data  # type: ignore[union-attr]

        # Reuse the output buffer and compute matrix multiplication directly.
        # The loop order keeps each weight row contiguous.
        for batch in range(batch_size):
            x_row = x[batch]
            result_row = result[batch]

            # The output buffer is reused across steps, so reset it before
            # accumulating the new matrix product.
            for output_index in range(output_size):
                result_row[output_index] = 0.0

            for input_index in range(input_size):
                value = x_row[input_index]
                w_row = w[input_index]
                for output_index in range(output_size):
                    result_row[output_index] += value * w_row[output_index]

            for output_index in range(output_size):
                result_row[output_index] += b[output_index]

        self._cached_input = inputs
        return self._cached_output

    def backward(
        self,
        grad_output: Tensor,
        *,
        compute_input_gradient: bool = True,
    ) -> Tensor | None:
        if self._cached_input is None:
            raise RuntimeError("forward must be called before backward.")

        if grad_output.ndim != 2:
            raise ValueError("grad_output must be rank-2.")

        inputs = self._cached_input
        if grad_output.shape[0] != inputs.shape[0]:
            raise ValueError("Batch size mismatch in Dense.backward.")

        batch_size, input_size = inputs.shape
        _, output_size = grad_output.shape

        self._ensure_output_buffers(batch_size)

        x = inputs._data  # type: ignore[attr-defined]
        go = grad_output._data  # type: ignore[attr-defined]
        w = self.weights._data  # type: ignore[attr-defined]
        grad_w = self.grad_weights._data  # type: ignore[attr-defined]
        grad_b = self.grad_bias._data  # type: ignore[attr-defined]
        grad_x = self._cached_grad_input._data  # type: ignore[union-attr]

        # Clear reusable gradient buffers.
        for input_index in range(input_size):
            grad_w_row = grad_w[input_index]
            for output_index in range(output_size):
                grad_w_row[output_index] = 0.0

        for output_index in range(output_size):
            grad_b[output_index] = 0.0

        # Fuse grad_W, grad_b, and grad_X into one batch traversal.
        # This keeps x/go rows hot and removes a second full traversal of the
        # batch. The arithmetic is unchanged; only the Python loop structure
        # is optimized.
        input_range = range(input_size)
        output_range = range(output_size)

        for batch in range(batch_size):
            x_row = x[batch]
            go_row = go[batch]
            grad_x_row = grad_x[batch]

            for output_index in output_range:
                grad_b[output_index] += go_row[output_index]

            for input_index in input_range:
                x_value = x_row[input_index]
                weight_row = w[input_index]
                grad_w_row = grad_w[input_index]
                total = 0.0

                for output_index in output_range:
                    grad_value = go_row[output_index]
                    grad_w_row[output_index] += x_value * grad_value
                    total += grad_value * weight_row[output_index]

                grad_x_row[input_index] = total

        return self._cached_grad_input

    def parameters(self) -> Dict[str, Tensor]:
        # Keep the fast cached mapping during normal training, but refresh it
        # if a caller replaces a parameter tensor.
        if (
            self._parameter_cache["weights"] is not self.weights
            or self._parameter_cache["bias"] is not self.bias
        ):
            self._parameter_cache = {
                "weights": self.weights,
                "bias": self.bias,
            }
        return self._parameter_cache

    def gradients(self) -> Dict[str, Tensor]:
        if (
            self._gradient_cache["weights"] is not self.grad_weights
            or self._gradient_cache["bias"] is not self.grad_bias
        ):
            self._gradient_cache = {
                "weights": self.grad_weights,
                "bias": self.grad_bias,
            }
        return self._gradient_cache


class ReLU:
    """Rectified Linear Unit activation."""

    def __init__(self) -> None:
        self._cached_input: Tensor | None = None
        self._cached_output: Tensor | None = None
        self._cached_gradient: Tensor | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        if self._cached_output is None or self._cached_output.shape != inputs.shape:
            self._cached_output = Tensor._from_data(
                [
                    [0.0] * inputs.shape[1]
                    for _ in range(inputs.shape[0])
                ],
                inputs.shape,
            )

        input_data = inputs._data  # type: ignore[attr-defined]
        output_data = self._cached_output._data  # type: ignore[attr-defined]

        for row_index in range(inputs.shape[0]):
            input_row = input_data[row_index]
            output_row = output_data[row_index]
            for col_index in range(inputs.shape[1]):
                value = input_row[col_index]
                output_row[col_index] = value if value > 0.0 else 0.0

        self._cached_input = inputs
        return self._cached_output

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._cached_input is None:
            raise RuntimeError("forward must be called before backward.")

        if grad_output.shape != self._cached_input.shape:
            raise ValueError("Gradient shape must match the cached input.")

        if self._cached_gradient is None or self._cached_gradient.shape != grad_output.shape:
            self._cached_gradient = Tensor._from_data(
                [
                    [0.0] * grad_output.shape[1]
                    for _ in range(grad_output.shape[0])
                ],
                grad_output.shape,
            )

        input_data = self._cached_input._data  # type: ignore[attr-defined]
        grad_data = grad_output._data  # type: ignore[attr-defined]
        output_data = self._cached_gradient._data  # type: ignore[attr-defined]

        for row_index in range(grad_output.shape[0]):
            input_row = input_data[row_index]
            grad_row = grad_data[row_index]
            output_row = output_data[row_index]
            for col_index in range(grad_output.shape[1]):
                output_row[col_index] = (
                    grad_row[col_index]
                    if input_row[col_index] > 0.0
                    else 0.0
                )

        return self._cached_gradient


class Tanh:
    """Hyperbolic tangent activation."""

    def __init__(self) -> None:
        self._cached_output: Tensor | None = None
        self._cached_gradient: Tensor | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        if self._cached_output is None or self._cached_output.shape != inputs.shape:
            self._cached_output = Tensor._from_data(
                [
                    [0.0] * inputs.shape[1]
                    for _ in range(inputs.shape[0])
                ],
                inputs.shape,
            )

        input_data = inputs._data  # type: ignore[attr-defined]
        output_data = self._cached_output._data  # type: ignore[attr-defined]

        for row_index in range(inputs.shape[0]):
            input_row = input_data[row_index]
            output_row = output_data[row_index]
            for col_index in range(inputs.shape[1]):
                output_row[col_index] = math.tanh(input_row[col_index])

        return self._cached_output

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._cached_output is None:
            raise RuntimeError("forward must be called before backward.")

        if grad_output.shape != self._cached_output.shape:
            raise ValueError("Gradient shape must match the cached output.")

        if self._cached_gradient is None or self._cached_gradient.shape != grad_output.shape:
            self._cached_gradient = Tensor._from_data(
                [
                    [0.0] * grad_output.shape[1]
                    for _ in range(grad_output.shape[0])
                ],
                grad_output.shape,
            )

        output_data = self._cached_output._data  # type: ignore[attr-defined]
        grad_data = grad_output._data  # type: ignore[attr-defined]
        result_data = self._cached_gradient._data  # type: ignore[attr-defined]

        for row_index in range(grad_output.shape[0]):
            output_row = output_data[row_index]
            grad_row = grad_data[row_index]
            result_row = result_data[row_index]
            for col_index in range(grad_output.shape[1]):
                output_value = output_row[col_index]
                result_row[col_index] = (
                    grad_row[col_index] * (1.0 - output_value * output_value)
                )

        return self._cached_gradient


__all__ = ["Dense", "ReLU", "Tanh"]
