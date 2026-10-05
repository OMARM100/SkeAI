"""Small sequential neural network for SkeAI 0.3."""

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
        self._parameter_cache: Dict[str, Tensor] = {}
        self._gradient_cache: Dict[str, Tensor] = {}

        for index, layer in enumerate(self.layers):
            if isinstance(layer, Dense):
                self._parameter_cache[f"layer{index}.weights"] = layer.weights
                self._parameter_cache[f"layer{index}.bias"] = layer.bias
                self._gradient_cache[f"layer{index}.weights"] = layer.grad_weights
                self._gradient_cache[f"layer{index}.bias"] = layer.grad_bias

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
        return self._parameter_cache

    def gradients(self) -> Dict[str, Tensor]:
        return self._gradient_cache

    def parameter_count(self) -> int:
        """Return the total number of scalar trainable parameters."""
        return sum(parameter.size for parameter in self._parameter_cache.values())

    def state_dict(self) -> Dict[str, Any]:
        """Return JSON-serializable trainable parameters."""
        return {
            name: parameter.to_list()
            for name, parameter in self._parameter_cache.items()
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Load trainable parameters after validating their shapes."""
        parameters = self._parameter_cache

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
