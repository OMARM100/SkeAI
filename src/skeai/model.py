"""Small sequential neural network for SkeAI 0.1."""

from __future__ import annotations

from typing import Any, Dict, List

from .layers import Dense, ReLU, Tanh
from .tensor import Tensor


Layer = Dense | ReLU | Tanh


class Sequential:
    """Run a list of layers in order and reverse order for backpropagation."""

    def __init__(self, layers: List[Layer]) -> None:
        if not layers:
            raise ValueError("Sequential requires at least one layer.")
        self.layers = layers

    def forward(self, inputs: Tensor) -> Tensor:
        output = inputs
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, gradient: Tensor) -> Tensor:
        output = gradient
        for layer in reversed(self.layers):
            output = layer.backward(output)
        return output

    def parameters(self) -> Dict[str, Tensor]:
        parameters: Dict[str, Tensor] = {}

        for index, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                for name, parameter in layer.parameters().items():
                    parameters[f"layer{index}.{name}"] = parameter

        return parameters

    def gradients(self) -> Dict[str, Tensor]:
        gradients: Dict[str, Tensor] = {}

        for index, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                for name, gradient in layer.gradients().items():
                    gradients[f"layer{index}.{name}"] = gradient

        return gradients

    def state_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable trainable parameters."""
        return {
            name: parameter.to_list()
            for name, parameter in self.parameters().items()
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load trainable parameters after validating their shapes."""
        parameters = self.parameters()

        if set(state) != set(parameters):
            raise ValueError("Checkpoint parameters do not match the model.")

        for name, parameter in parameters.items():
            loaded = Tensor(state[name])
            if loaded.shape != parameter.shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{name}': "
                    f"expected {parameter.shape}, got {loaded.shape}."
                )
            parameter._data = loaded.to_list()  # type: ignore[attr-defined]


__all__ = ["Sequential"]
