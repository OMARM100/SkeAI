import unittest

from src.skeai.tensor import Tensor


class TensorTests(unittest.TestCase):
    def test_shape_and_size(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])

        self.assertEqual(tensor.shape, (2, 3))
        self.assertEqual(tensor.ndim, 2)
        self.assertEqual(tensor.size, 6)

    def test_elementwise_operations(self) -> None:
        left = Tensor([[1, 2], [3, 4]])
        right = Tensor([[5, 6], [7, 8]])

        self.assertEqual(
            (left + right).to_list(),
            [[6.0, 8.0], [10.0, 12.0]],
        )
        self.assertEqual(
            (right - left).to_list(),
            [[4.0, 4.0], [4.0, 4.0]],
        )
        self.assertEqual(
            (left * 2).to_list(),
            [[2.0, 4.0], [6.0, 8.0]],
        )

    def test_matrix_multiplication(self) -> None:
        left = Tensor([[1, 2, 3], [4, 5, 6]])
        right = Tensor([[7, 8], [9, 10], [11, 12]])

        self.assertEqual(
            left.matmul(right).to_list(),
            [[58.0, 64.0], [139.0, 154.0]],
        )

    def test_transpose(self) -> None:
        tensor = Tensor([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(
            tensor.transpose().to_list(),
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
        )

    def test_rejects_ragged_data(self) -> None:
        with self.assertRaises(ValueError):
            Tensor([[1, 2], [3]])

    def test_incompatible_matmul(self) -> None:
        with self.assertRaises(ValueError):
            Tensor([[1, 2]]).matmul(Tensor([[1, 2]]))


if __name__ == "__main__":
    unittest.main()
