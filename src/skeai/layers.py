"""Neural-network layers backed by the SkeAI C++ engine."""
from __future__ import annotations

import random
from typing import Dict, Sequence

from . import engine
from .tensor import Tensor


class Dense:
    """Fully connected layer for rank-2 [batch, features] tensors."""

    def __init__(self, input_size: int, output_size: int, seed: int = 42) -> None:
        if input_size <= 0 or output_size <= 0:
            raise ValueError("Layer sizes must be positive integers.")

        rng = random.Random(seed)
        limit = (6.0 / (input_size + output_size)) ** 0.5

        self.weights = Tensor([
            [rng.uniform(-limit, limit) for _ in range(output_size)]
            for _ in range(input_size)
        ])
        self.bias = Tensor.zeros((output_size,))
        self.grad_weights = Tensor.zeros((input_size, output_size))
        self.grad_bias = Tensor.zeros((output_size,))

        self._cached_input: Tensor | None = None
        self._cached_indexed_input: list[int] | None = None
        self._cached_output: Tensor | None = None
        self._cached_grad_input: Tensor | None = None

        self._parameter_cache = {"weights": self.weights, "bias": self.bias}
        self._gradient_cache = {"weights": self.grad_weights, "bias": self.grad_bias}

    def _ensure_output_buffers(self, batch_size: int) -> None:
        input_size = self.weights.shape[0]
        output_size = self.weights.shape[1]

        output_shape = (batch_size, output_size)
        grad_input_shape = (batch_size, input_size)

        if self._cached_output is None or self._cached_output.shape != output_shape:
            self._cached_output = Tensor.zeros(output_shape)

        if self._cached_grad_input is None or self._cached_grad_input.shape != grad_input_shape:
            self._cached_grad_input = Tensor.zeros(grad_input_shape)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.weights.shape[0]:
            raise ValueError("Dense.forward expects [batch, input_size] tensor.")

        batch_size, input_size = inputs.shape
        output_size = self.weights.shape[1]
        self._ensure_output_buffers(batch_size)

        engine.dense_forward(
            inputs._storage,
            self.weights._storage,
            self.bias._storage,
            self._cached_output._storage,  # type: ignore[union-attr]
            batch_size,
            input_size,
            output_size,
        )

        self._cached_input = inputs
        self._cached_indexed_input = None
        return self._cached_output  # type: ignore[return-value]

    def forward_indexed(self, indices: Sequence[int], batch_size: int) -> Tensor:
        """Forward pass for compact indexed one-hot input."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if not indices or len(indices) % batch_size != 0:
            raise ValueError("Indexed inputs must divide evenly across the batch.")

        input_size = self.weights.shape[0]
        output_size = self.weights.shape[1]
        normalized = [int(value) for value in indices]

        if any(value < 0 or value >= input_size for value in normalized):
            raise ValueError("Indexed feature is out of range.")

        self._ensure_output_buffers(batch_size)
        engine.dense_indexed_forward(
            normalized,
            self.weights._storage,
            self.bias._storage,
            self._cached_output._storage,  # type: ignore[union-attr]
            batch_size,
            input_size,
            output_size,
        )

        self._cached_input = None
        self._cached_indexed_input = normalized
        return self._cached_output  # type: ignore[return-value]
    def backward(
        self,
        grad_output: Tensor,
        *,
        compute_input_gradient: bool = True,
    ) -> Tensor | None:
        if self._cached_indexed_input is not None:
            if compute_input_gradient:
                raise ValueError(
                    "Indexed dense inputs do not have a differentiable input gradient."
                )

            batch_size = grad_output.shape[0]
            input_size = self.weights.shape[0]
            output_size = self.weights.shape[1]

            if grad_output.shape[1] != output_size:
                raise ValueError("Output feature count mismatch in Dense.backward.")

            engine.dense_indexed_backward(
                self._cached_indexed_input,
                grad_output._storage,
                self.grad_weights._storage,
                self.grad_bias._storage,
                batch_size,
                input_size,
                output_size,
            )
            return None
        if self._cached_input is None:
            raise RuntimeError("forward must be called before backward.")
        if grad_output.ndim != 2:
            raise ValueError("grad_output must be rank-2.")

        inputs = self._cached_input

        if grad_output.shape[0] != inputs.shape[0]:
            raise ValueError("Batch size mismatch in Dense.backward.")

        batch_size, input_size = inputs.shape
        output_size = grad_output.shape[1]

        if output_size != self.weights.shape[1]:
            raise ValueError("Output feature count mismatch in Dense.backward.")

        self._ensure_output_buffers(batch_size)

        engine.dense_backward(
            inputs._storage,
            grad_output._storage,
            self.weights._storage,
            self.grad_weights._storage,
            self.grad_bias._storage,
            self._cached_grad_input._storage,  # type: ignore[union-attr]
            compute_input_gradient,
            batch_size,
            input_size,
            output_size,
        )

        return self._cached_grad_input

    def parameters(self) -> Dict[str, Tensor]:
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
    """Rectified linear unit activation."""

    def __init__(self) -> None:
        self._cached_input: Tensor | None = None
        self._cached_output: Tensor | None = None
        self._cached_gradient: Tensor | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        if self._cached_output is None or self._cached_output.shape != inputs.shape:
            self._cached_output = Tensor.zeros(inputs.shape)

        engine.relu_forward(
            inputs._storage,
            self._cached_output._storage,
        )
        self._cached_input = inputs
        return self._cached_output

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._cached_input is None:
            raise RuntimeError("forward must be called before backward.")
        if grad_output.shape != self._cached_input.shape:
            raise ValueError("Gradient shape must match the cached input.")

        if self._cached_gradient is None or self._cached_gradient.shape != grad_output.shape:
            self._cached_gradient = Tensor.zeros(grad_output.shape)

        engine.relu_backward(
            self._cached_input._storage,
            grad_output._storage,
            self._cached_gradient._storage,
        )
        return self._cached_gradient


class Tanh:
    """Hyperbolic tangent activation."""

    def __init__(self) -> None:
        self._cached_output: Tensor | None = None
        self._cached_gradient: Tensor | None = None

    def forward(self, inputs: Tensor) -> Tensor:
        if self._cached_output is None or self._cached_output.shape != inputs.shape:
            self._cached_output = Tensor.zeros(inputs.shape)

        engine.tanh_forward(
            inputs._storage,
            self._cached_output._storage,
        )
        return self._cached_output

    def backward(self, grad_output: Tensor) -> Tensor:
        if self._cached_output is None:
            raise RuntimeError("forward must be called before backward.")
        if grad_output.shape != self._cached_output.shape:
            raise ValueError("Gradient shape must match the cached output.")

        if self._cached_gradient is None or self._cached_gradient.shape != grad_output.shape:
            self._cached_gradient = Tensor.zeros(grad_output.shape)

        engine.tanh_backward(
            self._cached_output._storage,
            grad_output._storage,
            self._cached_gradient._storage,
        )
        return self._cached_gradient


__all__ = ["Dense", "ReLU", "Tanh"]
