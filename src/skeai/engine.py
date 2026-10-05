"""Execution engine for SkeAI numerical kernels.

The public model/layer APIs remain Python-level, while expensive numerical
kernels live behind this dispatch layer. A native C backend is used when it
has been built; otherwise the same operations fall back to Python.
"""

from __future__ import annotations

from typing import Any

try:
    from . import _native
except ImportError:  # pragma: no cover
    _native = None

NATIVE_AVAILABLE = _native is not None
BACKEND_NAME = "native-c" if NATIVE_AVAILABLE else "python"


def matmul(left: Any, right: Any) -> Any:
    if NATIVE_AVAILABLE:
        return _native.matmul(left, right)

    rows = len(left)
    inner = len(right)
    cols = len(right[0]) if inner else 0
    result = [[0.0] * cols for _ in range(rows)]

    for row_index in range(rows):
        left_row = left[row_index]
        result_row = result[row_index]
        for inner_index in range(inner):
            value = left_row[inner_index]
            right_row = right[inner_index]
            for col_index in range(cols):
                result_row[col_index] += value * right_row[col_index]

    return result


def dense_forward(inputs: Any, weights: Any, bias: Any, output: Any) -> None:
    if NATIVE_AVAILABLE:
        _native.dense_forward(inputs, weights, bias, output)
        return

    batch_size = len(inputs)
    input_size = len(weights)
    output_size = len(bias)

    for batch in range(batch_size):
        x_row = inputs[batch]
        result_row = output[batch]

        for output_index in range(output_size):
            result_row[output_index] = bias[output_index]

        for input_index in range(input_size):
            value = x_row[input_index]
            weight_row = weights[input_index]
            for output_index in range(output_size):
                result_row[output_index] += value * weight_row[output_index]


def dense_backward(
    inputs: Any,
    grad_output: Any,
    weights: Any,
    grad_weights: Any,
    grad_bias: Any,
    grad_input: Any,
    compute_input_gradient: bool,
) -> None:
    if NATIVE_AVAILABLE:
        _native.dense_backward(
            inputs,
            grad_output,
            weights,
            grad_weights,
            grad_bias,
            grad_input,
            bool(compute_input_gradient),
        )
        return

    batch_size = len(inputs)
    input_size = len(weights)
    output_size = len(grad_bias)

    for input_index in range(input_size):
        grad_weight_row = grad_weights[input_index]
        for output_index in range(output_size):
            grad_weight_row[output_index] = 0.0

    for output_index in range(output_size):
        grad_bias[output_index] = 0.0

    for batch in range(batch_size):
        x_row = inputs[batch]
        go_row = grad_output[batch]
        grad_input_row = grad_input[batch]

        for output_index in range(output_size):
            grad_bias[output_index] += go_row[output_index]

        for input_index in range(input_size):
            x_value = x_row[input_index]
            grad_weight_row = grad_weights[input_index]

            if compute_input_gradient:
                weight_row = weights[input_index]
                total = 0.0

                for output_index in range(output_size):
                    grad_value = go_row[output_index]
                    grad_weight_row[output_index] += x_value * grad_value
                    total += grad_value * weight_row[output_index]

                grad_input_row[input_index] = total
            else:
                for output_index in range(output_size):
                    grad_weight_row[output_index] += (
                        x_value * go_row[output_index]
                    )


__all__ = [
    "BACKEND_NAME",
    "NATIVE_AVAILABLE",
    "dense_backward",
    "dense_forward",
    "matmul",
]
