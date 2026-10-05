"""Python boundary for the SkeAI C++ numerical engine."""
from __future__ import annotations

from typing import Any, Sequence

try:
    from . import _cpp
except ImportError as exc:  # pragma: no cover
    _cpp = None
    NATIVE_IMPORT_ERROR = exc
else:
    NATIVE_IMPORT_ERROR = None

NATIVE_AVAILABLE = _cpp is not None
BACKEND_NAME = "cpp" if NATIVE_AVAILABLE else "unavailable"
SCALAR_TYPE = "float64"


def _require_cpp() -> Any:
    if _cpp is None:
        raise RuntimeError(
            "SkeAI requires the C++ engine. Run python -m tools.build_cpp first."
        ) from NATIVE_IMPORT_ERROR
    return _cpp


def storage_from_flat(values: Sequence[float]) -> Any:
    return _require_cpp().storage_from_flat(values)


def storage_zeros(size: int) -> Any:
    return _require_cpp().storage_zeros(int(size))


def storage_to_flat(storage: Any) -> list[float]:
    return storage.to_list()


def add(left: Any, right: Any) -> Any:
    return _require_cpp().add(left, right)


def subtract(left: Any, right: Any) -> Any:
    return _require_cpp().subtract(left, right)


def multiply(left: Any, right: Any) -> Any:
    return _require_cpp().multiply(left, right)


def scalar_add(storage: Any, scalar: float) -> Any:
    return _require_cpp().scalar_add(storage, float(scalar))


def scalar_subtract(storage: Any, scalar: float) -> Any:
    return _require_cpp().scalar_subtract(storage, float(scalar))


def scalar_reverse_subtract(storage: Any, scalar: float) -> Any:
    return _require_cpp().scalar_reverse_subtract(storage, float(scalar))


def scalar_multiply(storage: Any, scalar: float) -> Any:
    return _require_cpp().scalar_multiply(storage, float(scalar))


def scalar_divide(storage: Any, scalar: float) -> Any:
    return _require_cpp().scalar_divide(storage, float(scalar))


def matmul(left: Any, right: Any, rows: int, inner: int, right_rows: int, cols: int) -> Any:
    return _require_cpp().matmul(
        left, right, int(rows), int(inner), int(right_rows), int(cols)
    )

def causal_softmax(storage: Any, rows: int, cols: int) -> Any:
    return _require_cpp().causal_softmax(storage, int(rows), int(cols))


def softmax_backward(
    probabilities: Any,
    gradient: Any,
    rows: int,
    cols: int,
) -> Any:
    return _require_cpp().softmax_backward(
        probabilities,
        gradient,
        int(rows),
        int(cols),
    )



def transpose(storage: Any, rows: int, cols: int) -> Any:
    return _require_cpp().transpose(storage, int(rows), int(cols))


def dense_forward(
    inputs: Any,
    weights: Any,
    bias: Any,
    output: Any,
    batch: int,
    input_size: int,
    output_size: int,
) -> None:
    _require_cpp().dense_forward(
        inputs, weights, bias, output,
        int(batch), int(input_size), int(output_size)
    )


def dense_indexed_forward(
    indices: Sequence[int],
    weights: Any,
    bias: Any,
    output: Any,
    batch: int,
    input_size: int,
    output_size: int,
) -> None:
    _require_cpp().dense_indexed_forward(
        indices,
        weights,
        bias,
        output,
        int(batch),
        int(input_size),
        int(output_size),
    )


def dense_indexed_backward(
    indices: Sequence[int],
    grad_output: Any,
    grad_weights: Any,
    grad_bias: Any,
    batch: int,
    input_size: int,
    output_size: int,
) -> None:
    _require_cpp().dense_indexed_backward(
        indices,
        grad_output,
        grad_weights,
        grad_bias,
        int(batch),
        int(input_size),
        int(output_size),
    )


def dense_backward(
    inputs: Any,
    grad_output: Any,
    weights: Any,
    grad_weights: Any,
    grad_bias: Any,
    grad_input: Any,
    compute_input_gradient: bool,
    batch: int,
    input_size: int,
    output_size: int,
) -> None:
    _require_cpp().dense_backward(
        inputs, grad_output, weights,
        grad_weights, grad_bias, grad_input,
        bool(compute_input_gradient),
        int(batch), int(input_size), int(output_size),
    )


def relu_forward(inputs: Any, output: Any) -> None:
    _require_cpp().relu_forward(inputs, output)


def relu_backward(inputs: Any, grad_output: Any, output: Any) -> None:
    _require_cpp().relu_backward(inputs, grad_output, output)


def tanh_forward(inputs: Any, output: Any) -> None:
    _require_cpp().tanh_forward(inputs, output)


def tanh_backward(cached_output: Any, grad_output: Any, output: Any) -> None:
    _require_cpp().tanh_backward(cached_output, grad_output, output)


def mse_forward(predictions: Any, targets: Any) -> float:
    return float(_require_cpp().mse_forward(predictions, targets))


def mse_backward(predictions: Any, targets: Any, gradient: Any) -> None:
    _require_cpp().mse_backward(predictions, targets, gradient)


def cross_entropy_forward(
    logits: Any,
    targets: Sequence[int],
    gradient: Any,
    batch: int,
    classes: int,
) -> float:
    return float(
        _require_cpp().cross_entropy_forward(
            logits, targets, gradient, int(batch), int(classes)
        )
    )


def sgd_step(
    parameter: Any,
    gradient: Any,
    learning_rate: float,
    weight_decay: float = 0.0,
) -> None:
    _require_cpp().sgd_step(
        parameter,
        gradient,
        float(learning_rate),
        float(weight_decay),
    )


def transformer_train_batch(
    parameters: Sequence[Any],
    inputs: Sequence[Sequence[int]],
    targets: Sequence[Sequence[int]],
    vocabulary_size: int,
    context_length: int,
    d_model: int,
    n_heads: int,
    feed_forward_size: int,
    n_layers: int,
    learning_rate: float,
) -> float:
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets batch sizes must match.")
    if not inputs:
        raise ValueError("training batch cannot be empty.")

    return float(
        _require_cpp().transformer_train_batch(
            [parameter for parameter in parameters],
            [list(sequence) for sequence in inputs],
            [list(sequence) for sequence in targets],
            int(vocabulary_size),
            int(context_length),
            int(d_model),
            int(n_heads),
            int(feed_forward_size),
            int(n_layers),
            len(inputs[0]),
            float(learning_rate),
        )
    )


def transformer_train_step(
    parameters: Sequence[Any],
    token_ids: Sequence[int],
    target_ids: Sequence[int],
    vocabulary_size: int,
    context_length: int,
    d_model: int,
    n_heads: int,
    feed_forward_size: int,
    n_layers: int,
    learning_rate: float,
) -> float:
    native_parameters = list(parameters)
    native_tokens = list(token_ids)
    native_targets = list(target_ids)

    if len(native_tokens) != len(native_targets):
        raise ValueError("token_ids and target_ids must have the same length.")
    if not native_tokens:
        raise ValueError("token_ids cannot be empty.")

    return float(
        _require_cpp().transformer_train_step(
            native_parameters,
            native_tokens,
            native_targets,
            int(vocabulary_size),
            int(context_length),
            int(d_model),
            int(n_heads),
            int(feed_forward_size),
            int(n_layers),
            len(native_tokens),
            float(learning_rate),
        )
    )


__all__ = [
    "BACKEND_NAME",
    "NATIVE_AVAILABLE",
    "NATIVE_IMPORT_ERROR",
    "SCALAR_TYPE",
    "add",
    "subtract",
    "multiply",
    "scalar_add",
    "scalar_subtract",
    "scalar_reverse_subtract",
    "scalar_multiply",
    "scalar_divide",
    "matmul",
    "causal_softmax",
    "softmax_backward",
    "transpose",
    "dense_forward",
    "dense_indexed_forward",
    "dense_indexed_backward",
    "dense_backward",
    "relu_forward",
    "relu_backward",
    "tanh_forward",
    "tanh_backward",
    "mse_forward",
    "mse_backward",
    "cross_entropy_forward",
    "sgd_step",
    "transformer_train_step",
    "transformer_train_batch",
    "storage_from_flat",
    "storage_zeros",
    "storage_to_flat",
]
