import unittest

from src.skeai.engine import (
    BACKEND_NAME,
    NATIVE_AVAILABLE,
    SCALAR_TYPE,
    matmul_baseline,
    matmul_tiled_16,
    matmul_tiled_32,
    matmul_tiled_64,
    storage_from_flat,
)
from src.skeai.layers import Dense
from src.skeai.tensor import Tensor


@unittest.skipUnless(NATIVE_AVAILABLE, "C++ backend is not built")
class CppEngineTests(unittest.TestCase):
    def test_backend_identity(self) -> None:
        self.assertEqual(BACKEND_NAME, "cpp")
        self.assertEqual(SCALAR_TYPE, "float64")

    def test_tensor_storage_is_contiguous(self) -> None:
        tensor = Tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(tensor._storage.size, 4)
        self.assertEqual(tensor.flatten(), [1.0, 2.0, 3.0, 4.0])

    def test_matrix_multiplication(self) -> None:
        left = Tensor([[1.0, 2.0], [3.0, 4.0]])
        right = Tensor([[5.0, 6.0], [7.0, 8.0]])

        self.assertEqual(
            left.matmul(right).to_list(),
            [[19.0, 22.0], [43.0, 50.0]],
        )

    def test_matmul_shapes_match_reference_and_retained_baseline(self) -> None:
        cases = [
            ([[2.0]], [[-3.0]]),
            (
                [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]],
                [
                    [2.0, 0.5, -1.0, 3.0],
                    [-4.0, 2.0, 1.5, -0.5],
                    [3.0, -2.0, 4.0, 1.0],
                ],
            ),
        ]

        # Irregular shapes exercise dimensions around the future 16/32 tiles.
        for rows, inner, cols in ((17, 19, 21), (31, 33, 17)):
            left = [
                [((row * 7 + col * 3) % 19 - 9) / 7.0 for col in range(inner)]
                for row in range(rows)
            ]
            right = [
                [((row * 5 + col * 11) % 23 - 11) / 9.0 for col in range(cols)]
                for row in range(inner)
            ]
            cases.append((left, right))

        for left_values, right_values in cases:
            with self.subTest(
                left_shape=(len(left_values), len(left_values[0])),
                right_shape=(len(right_values), len(right_values[0])),
            ):
                left = Tensor(left_values)
                right = Tensor(right_values)
                rows = len(left_values)
                inner = len(left_values[0])
                cols = len(right_values[0])

                baseline = matmul_baseline(
                    left._storage,
                    right._storage,
                    rows,
                    inner,
                    len(right_values),
                    cols,
                ).to_list()
                dispatched = left.matmul(right).flatten()
                reference = [
                    sum(
                        left_values[row][k] * right_values[k][col]
                        for k in range(inner)
                    )
                    for row in range(rows)
                    for col in range(cols)
                ]

                self.assertEqual(len(baseline), len(reference))
                self.assertEqual(len(dispatched), len(reference))
                for actual, expected in zip(baseline, reference):
                    self.assertAlmostEqual(actual, expected, places=8)
                for actual, expected in zip(dispatched, reference):
                    self.assertAlmostEqual(actual, expected, places=8)

    def test_tiled_matmul_dispatch_matches_baseline_at_threshold(self) -> None:
        # 100x100x100 reaches the dispatch threshold and exercises the tiled
        # implementation while keeping the test deterministic and reproducible.
        rows = inner = cols = 100
        left_values = [
            [((row * 7 + col * 3) % 19 - 9) / 7.0 for col in range(inner)]
            for row in range(rows)
        ]
        right_values = [
            [((row * 5 + col * 11) % 23 - 11) / 9.0 for col in range(cols)]
            for row in range(inner)
        ]
        left = Tensor(left_values)
        right = Tensor(right_values)

        baseline = matmul_baseline(
            left._storage,
            right._storage,
            rows,
            inner,
            inner,
            cols,
        ).to_list()
        dispatched = left.matmul(right).flatten()

        self.assertEqual(len(dispatched), rows * cols)
        self.assertEqual(dispatched, baseline)

    def test_tiled_kernel_variants_match_baseline(self) -> None:
        # Include irregular dimensions and a workload above the production
        # dispatch threshold; all variants preserve ascending-k accumulation.
        for rows, inner, cols in ((17, 19, 21), (100, 100, 100)):
            left_values = [
                [((row * 7 + col * 3) % 19 - 9) / 7.0 for col in range(inner)]
                for row in range(rows)
            ]
            right_values = [
                [((row * 5 + col * 11) % 23 - 11) / 9.0 for col in range(cols)]
                for row in range(inner)
            ]
            left = storage_from_flat(
                [value for row in left_values for value in row]
            )
            right = storage_from_flat(
                [value for row in right_values for value in row]
            )
            baseline = matmul_baseline(
                left, right, rows, inner, inner, cols
            ).to_list()

            for tile, kernel in (
                (16, matmul_tiled_16),
                (32, matmul_tiled_32),
                (64, matmul_tiled_64),
            ):
                with self.subTest(shape=(rows, inner, cols), tile=tile):
                    actual = kernel(
                        left, right, rows, inner, inner, cols
                    ).to_list()
                    self.assertEqual(actual, baseline)

    def test_matmul_baseline_rejects_invalid_dimensions_and_storage(self) -> None:
        left = storage_from_flat([1.0, 2.0])
        right = storage_from_flat([3.0, 4.0, 5.0, 6.0])

        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            matmul_baseline(left, right, -1, 2, 2, 2)

        with self.assertRaisesRegex(ValueError, "storage sizes"):
            matmul_baseline(left, right, 2, 2, 2, 2)

        with self.assertRaisesRegex(ValueError, "incompatible"):
            matmul_baseline(left, right, 1, 2, 3, 2)

    def test_dense_indexed_forward_and_backward(self) -> None:
        layer = Dense(4, 2, seed=1)
        layer.weights = Tensor(
            [
                [0.5, 1.0],
                [1.5, 2.0],
                [2.5, 3.0],
                [3.5, 4.0],
            ]
        )
        layer.bias = Tensor([0.25, -0.5])
        layer.grad_weights = Tensor.zeros((4, 2))
        layer.grad_bias = Tensor.zeros((2,))

        output = layer.forward_indexed([0, 3, 1, 2], batch_size=2)

        self.assertEqual(
            output.to_list(),
            [[4.25, 4.5], [4.25, 4.5]],
        )

        gradient = layer.backward(
            Tensor([[1.0, 2.0], [3.0, 4.0]]),
            compute_input_gradient=False,
        )

        self.assertIsNone(gradient)
        self.assertEqual(
            layer.grad_weights.to_list(),
            [[1.0, 2.0], [3.0, 4.0], [3.0, 4.0], [1.0, 2.0]],
        )
        self.assertEqual(layer.grad_bias.to_list(), [4.0, 6.0])
    def test_dense_forward_and_backward(self) -> None:
        layer = Dense(2, 2, seed=1)
        layer.weights = Tensor([[0.5, 1.0], [1.5, 2.0]])
        layer.bias = Tensor([0.25, -0.5])
        layer.grad_weights = Tensor.zeros((2, 2))
        layer.grad_bias = Tensor.zeros((2,))

        inputs = Tensor([[1.0, 2.0], [3.0, 4.0]])
        output = layer.forward(inputs)

        self.assertEqual(
            output.to_list(),
            [[3.75, 4.5], [7.75, 10.5]],
        )

        gradient = layer.backward(
            Tensor([[1.0, 2.0], [3.0, 4.0]])
        )

        self.assertEqual(
            layer.grad_weights.to_list(),
            [[10.0, 14.0], [14.0, 20.0]],
        )
        self.assertEqual(
            layer.grad_bias.to_list(),
            [4.0, 6.0],
        )
        self.assertEqual(
            gradient.to_list(),
            [[2.5, 5.5], [5.5, 12.5]],
        )


if __name__ == "__main__":
    unittest.main()
