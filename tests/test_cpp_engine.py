import unittest

from src.skeai.engine import BACKEND_NAME, NATIVE_AVAILABLE, SCALAR_TYPE
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
            [[2.5, 5.5], [5.5, 11.5]],
        )


if __name__ == "__main__":
    unittest.main()
