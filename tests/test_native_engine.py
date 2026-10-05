import unittest

from src.skeai.engine import NATIVE_AVAILABLE, dense_backward, dense_forward, matmul
from src.skeai.tensor import Tensor


@unittest.skipUnless(NATIVE_AVAILABLE, "native backend is not built")
class NativeEngineTests(unittest.TestCase):
    def test_native_matmul(self) -> None:
        result = matmul(
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        )
        self.assertEqual(result, [[19.0, 22.0], [43.0, 50.0]])

    def test_native_dense_forward_and_backward(self) -> None:
        inputs = Tensor([[1.0, 2.0], [3.0, 4.0]])
        weights = Tensor([[0.5, 1.0], [1.5, 2.0]])
        bias = Tensor([0.25, -0.5])
        output = Tensor._from_data(
            [[0.0, 0.0], [0.0, 0.0]],
            (2, 2),
        )

        dense_forward(
            inputs._data,
            weights._data,
            bias._data,
            output._data,
        )

        self.assertEqual(
            output.to_list(),
            [[3.75, 4.5], [7.75, 10.5]],
        )

        grad_output = [[1.0, 2.0], [3.0, 4.0]]
        grad_weights = Tensor._from_data(
            [[0.0, 0.0], [0.0, 0.0]],
            (2, 2),
        )
        grad_bias = Tensor._from_data([0.0, 0.0], (2,))
        grad_input = Tensor._from_data(
            [[0.0, 0.0], [0.0, 0.0]],
            (2, 2),
        )

        dense_backward(
            inputs._data,
            grad_output,
            weights._data,
            grad_weights._data,
            grad_bias._data,
            grad_input._data,
            True,
        )

        self.assertEqual(
            grad_weights.to_list(),
            [[10.0, 14.0], [14.0, 20.0]],
        )
        self.assertEqual(grad_bias.to_list(), [4.0, 6.0])
        self.assertEqual(
            grad_input.to_list(),
            [[2.5, 5.5], [5.5, 12.5]],
        )


if __name__ == "__main__":
    unittest.main()
