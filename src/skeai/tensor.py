"""Dense tensor abstraction backed by contiguous C++ storage."""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple, Union

from . import engine

Number = Union[int, float]
NestedNumbers = Union[Number, Sequence["NestedNumbers"]]


def _is_sequence(value: object) -> bool:
    return isinstance(value, (list, tuple))


def _to_nested_list(data: NestedNumbers) -> object:
    if not _is_sequence(data):
        if not isinstance(data, (int, float)) or isinstance(data, bool):
            raise TypeError("Tensor values must be int or float.")
        value = float(data)
        if not math.isfinite(value):
            raise ValueError("Tensor values must be finite.")
        return value
    return [_to_nested_list(item) for item in data]  # type: ignore[arg-type]


def _infer_shape(data: object) -> Tuple[int, ...]:
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    first_shape = _infer_shape(data[0])
    for item in data[1:]:
        if _infer_shape(item) != first_shape:
            raise ValueError("Tensor data must be rectangular.")
    return (len(data),) + first_shape


def _flatten(data: object) -> List[float]:
    if not isinstance(data, list):
        return [float(data)]
    values: List[float] = []
    for item in data:
        values.extend(_flatten(item))
    return values


def _unflatten(values: Sequence[float], shape: Tuple[int, ...]) -> object:
    if not shape:
        if len(values) != 1:
            raise ValueError("Scalar tensor storage must contain one value.")
        return float(values[0])

    iterator = iter(values)

    def build(dimension: int) -> object:
        if dimension == len(shape):
            return float(next(iterator))
        return [build(dimension + 1) for _ in range(shape[dimension])]

    return build(0)


class Tensor:
    """Dense tensor with contiguous C++-owned numeric storage."""

    def __init__(self, data: NestedNumbers) -> None:
        normalized = _to_nested_list(data)
        self._shape = _infer_shape(normalized)
        self._storage = engine.storage_from_flat(_flatten(normalized))

    @classmethod
    def _from_storage(cls, storage: object, shape: Tuple[int, ...]) -> "Tensor":
        expected_size = math.prod(shape)
        if int(storage.size) != expected_size:
            raise ValueError("Tensor storage size does not match the requested shape.")
        tensor = cls.__new__(cls)
        tensor._storage = storage
        tensor._shape = shape
        return tensor

    @classmethod
    def zeros(cls, shape: Tuple[int, ...]) -> "Tensor":
        if any(dimension < 0 for dimension in shape):
            raise ValueError("Tensor dimensions cannot be negative.")
        return cls._from_storage(engine.storage_zeros(math.prod(shape)), shape)

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
        return _unflatten(engine.storage_to_flat(self._storage), self._shape)

    def flatten(self) -> List[float]:
        return engine.storage_to_flat(self._storage)

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, data={self.to_list()!r})"

    def __add__(self, other: Union["Tensor", Number]) -> "Tensor":
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must match for addition.")
            storage = engine.add(self._storage, other._storage)
        else:
            storage = engine.scalar_add(self._storage, float(other))
        return Tensor._from_storage(storage, self.shape)

    def __radd__(self, other: Number) -> "Tensor":
        return self + other

    def __sub__(self, other: Union["Tensor", Number]) -> "Tensor":
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must match for subtraction.")
            storage = engine.subtract(self._storage, other._storage)
        else:
            storage = engine.scalar_subtract(self._storage, float(other))
        return Tensor._from_storage(storage, self.shape)

    def __rsub__(self, other: Number) -> "Tensor":
        return Tensor._from_storage(
            engine.scalar_reverse_subtract(self._storage, float(other)),
            self.shape,
        )

    def __mul__(self, other: Union["Tensor", Number]) -> "Tensor":
        if isinstance(other, Tensor):
            if self.shape != other.shape:
                raise ValueError("Tensor shapes must match for multiplication.")
            storage = engine.multiply(self._storage, other._storage)
        else:
            storage = engine.scalar_multiply(self._storage, float(other))
        return Tensor._from_storage(storage, self.shape)

    def __rmul__(self, other: Number) -> "Tensor":
        return self * other

    def __truediv__(self, other: Number) -> "Tensor":
        return Tensor._from_storage(
            engine.scalar_divide(self._storage, float(other)),
            self.shape,
        )

    def matmul(self, other: "Tensor") -> "Tensor":
        if self.ndim != 2 or other.ndim != 2:
            raise ValueError("matmul currently requires two rank-2 tensors.")
        rows, inner = self.shape
        other_inner, cols = other.shape
        if inner != other_inner:
            raise ValueError("Incompatible shapes for matrix multiplication.")
        return Tensor._from_storage(
            engine.matmul(
                self._storage, other._storage,
                rows, inner, other_inner, cols,
            ),
            (rows, cols),
        )

    def transpose(self) -> "Tensor":
        if self.ndim != 2:
            raise ValueError("transpose currently requires a rank-2 tensor.")
        rows, cols = self.shape
        return Tensor._from_storage(
            engine.transpose(self._storage, rows, cols),
            (cols, rows),
        )

    def map(self, function) -> "Tensor":
        return Tensor._from_storage(
            engine.storage_from_flat(
                [function(value) for value in self.flatten()]
            ),
            self.shape,
        )

    def _replace_storage(self, storage: object) -> None:
        if int(storage.size) != self.size:
            raise ValueError("Replacement storage has the wrong size.")
        self._storage = storage


__all__ = ["Tensor"]
