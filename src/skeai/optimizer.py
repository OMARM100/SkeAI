"""Optimizers for SkeAI 0.2."""

from __future__ import annotations

from typing import Dict

from .tensor import Tensor


class SGD:
    """Plain stochastic gradient descent with in-place parameter updates."""

    def __init__(self, learning_rate: float = 0.01) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero.")
        self.learning_rate = float(learning_rate)

    def step(
        self,
        parameters: Dict[str, Tensor],
        gradients: Dict[str, Tensor],
    ) -> None:
        if parameters.keys() != gradients.keys():
            raise ValueError("Parameters and gradients must have matching keys.")

        for name, parameter in parameters.items():
            gradient = gradients[name]

            if parameter.shape != gradient.shape:
                raise ValueError(f"Gradient shape mismatch for parameter '{name}'.")

            parameter_data = parameter._data  # type: ignore[attr-defined]
            gradient_data = gradient._data  # type: ignore[attr-defined]

            if parameter.ndim == 1:
                for index in range(parameter.shape[0]):
                    parameter_data[index] -= (
                        self.learning_rate * gradient_data[index]
                    )
            elif parameter.ndim == 2:
                rows, cols = parameter.shape
                for row in range(rows):
                    parameter_row = parameter_data[row]
                    gradient_row = gradient_data[row]
                    for col in range(cols):
                        parameter_row[col] -= (
                            self.learning_rate * gradient_row[col]
                        )
            else:
                raise ValueError(
                    f"SGD currently supports rank-1 and rank-2 tensors, got {parameter.shape}."
                )


__all__ = ["SGD"]
