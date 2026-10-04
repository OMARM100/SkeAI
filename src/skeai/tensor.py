"""Minimal tensor operations for SkeAI 0.1.

This module intentionally avoids NumPy/PyTorch. It provides the small set of
numeric operations we need while the project is learning the fundamentals.
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple, Union

Number = Union[int, float]
NestedNumbers = Union[Number, Sequence["NestedNumbers"]]


def _is_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _infer_shape(data: NestedNumbers) -> Tuple[int, ...]:
    if not _is_sequence(data):
        return ()

    seq = list(data)  # type: ignore[arg-type]
    if not seq:
        return (0,)

    first_shape = _infer_shape(seq[0])
    for item in seq[1:]:
        if _infer_shape(item) != first_shape:
            raise ValueError("Tensor data must be rectangular.")

    return (len(seq),) + first_shape


def _to_nested_list(data: NestedNumbers) -> object:
    if not _is_sequence(data):
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            raise TypeError("Tensor values must be int or float.")
        return float(data)

    return [_to_nested_list(item) for item in data]  # type: ignore[arg-type]


def _flatten(data: object) -> List[float]:
    if not isinstance(data, list):
        return [float(data)]

    output: List[float] = []
    for item in data:
        output.extend(_flatten(item))
    return output


def _elementwise(a: object, b: object, operation) -> object:
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            raise ValueError("Tensor shapes are not compatible.")
        return [_elementwise(x, y, operation) for x, y in zip(a, b)]

    if isinstance(a, list) or isinstance(b, list):
        raise ValueError("Tensor shapes are not compatible.")

    return operation(float(a), float(b))


def _map(data: object, operation) -> object:
    if isinstance(data, list):
        return [_map(item, operation) for item in data]
    return operation(float(data))


class Tensor:
    """Small dense tensor backed by nested Python lists."""

    def __init__(self, data: NestedNumbers) -> None:
        normalized = _to_nested_list(data)
        self._data = normalized
        self._shape = _infer_shape(normalized)  # type: ignore[arg-type]

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def size(self) -> int:
        return math.prod(self._shape)

    def to_list(self) -> object:
        """Return a copy of the underlying nested-list data."""
        def copy_nested(value: object) -> object:
            if isinstance(value, list):
                return [copy_nested(item) for item in value]
            return float(value)

        return copy_nested(self._data)

    def flatten(self) -> List[float]:
        return _flatten(self._data)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self._data!r})"

    def __add__(self, other: Union["Tensor", Number]) -> "Tensor":
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must match for addition.")
            data = _elementwise(self._data, other._data, lambda a, b: a + b)
        else:
            data = _map(self._data, lambda value: value + float(other))
        return Tensor(data)  # type: ignore[arg-type]

    def __radd__(self, other: Number) -> "Tensor":
        return self + other

    def __sub__(self, other: Union["Tensor", Number]) -> "Tensor":
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must match for subtraction.")
            data = _elementwise(self._data, other._data, lambda a, b: a - b)
        else:
            data = _map(self._data, lambda value: value - float(other))
        return Tensor(data)  # type: ignore[arg-type]

    def __rsub__(self, other: Number) -> "Tensor":
        return Tensor(_map(self._data, lambda value: float(other) - value))  # type: ignore[arg-type]

    def __mul__(self, other: Union["Tensor", Number]) -> "Tensor":
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must match for multiplication.")
            data = _elementwise(self._data, other._data, lambda a, b: a * b)
        else:
            data = _map(self._data, lambda value: value * float(other))
        return Tensor(data)  # type: ignore[arg-type]

    def __rmul__(self, other: Number) -> "Tensor":
        return self * other

    def __truediv__(self, other: Number) -> "Tensor":
        if other == 0:
            raise ZeroDivisionError("Cannot divide a tensor by zero.")
        return Tensor(_map(self._data, lambda value: value / float(other)))  # type: ignore[arg-type]

    def matmul(self, other: "Tensor") -> "Tensor":
        """Matrix multiplication for rank-2 tensors."""
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul currently requires two rank-2 tensors.")

        rows, inner = self.shape
        other_inner, cols = other.shape

        if inner != other_inner:
            raise ValueError("Incompatible shapes for matrix multiplication.")

        left = self._data
        right = other._data

        result = [
            [
                sum(float(left[i][k]) * float(right[k][j]) for k in range(inner))
                for j in range(cols)
            ]
            for i in range(rows)
        ]

        return Tensor(result)  # type: ignore[arg-type]

    def transpose(self) -> "Tensor":
        if self.ndim != 2:
            raise ValueError("transpose currently requires a rank-2 tensor.")

        rows, cols = self.shape
        data = [
            [self._data[row][col] for row in range(rows)]  # type: ignore[index]
            for col in range(cols)
        ]
        return Tensor(data)  # type: ignore[arg-type]

    def map(self, function) -> "Tensor":
        return Tensor(_map(self._data, function))  # type: ignore[arg-type]


__all__ = ["Tensor"]
