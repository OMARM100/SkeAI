"""Optimizers for SkeAI 0.1."""

from __future__ import annotations

from typing import Dict

from .tensor import Tensor


class SGD:
    """Plain stochastic gradient descent."""

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

            # Parameters are intentionally updated through a new Tensor. This
            # keeps the Tensor class simple and makes optimizer behavior clear.
            updated = parameter - (gradient * self.learning_rate)
            parameters[name]._data = updated.to_list()  # type: ignore[attr-defined]


__all__ = ["SGD"]
