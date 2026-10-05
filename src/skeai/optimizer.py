"""Optimizers implemented through the SkeAI C++ engine."""
from __future__ import annotations

from typing import Dict

from . import engine
from .tensor import Tensor


class SGD:
    """Plain stochastic gradient descent with in-place parameter updates."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
    ) -> None:
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than zero.")
        if weight_decay < 0.0:
            raise ValueError("weight_decay cannot be negative.")
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)

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

            if parameter.ndim not in (1, 2):
                raise ValueError(
                    "SGD currently supports rank-1 and rank-2 tensors."
                )

            engine.sgd_step(
                parameter._storage,
                gradient._storage,
                self.learning_rate,
                self.weight_decay,
            )


__all__ = ["SGD"]
