import math
import unittest

from src.skeai.layers import Dense, ReLU, Tanh
from src.skeai.tensor import Tensor


class LayerTests(unittest.TestCase):
    def test_dense_forward_shape(self) -> None:
        layer = Dense(3, 2, seed=1)
        output = layer.forward(Tensor([[1, 2, 3], [4, 5, 6]]))

        self.assertEqual(output.shape, (2, 2))

    def test_dense_reused_forward_resets_output(self) -> None:
        layer = Dense(2, 2, seed=1)
        inputs = Tensor([[1.0, 0.0], [0.0, 1.0]])

        first = layer.forward(inputs).to_list()
        second = layer.forward(inputs).to_list()

        self.assertEqual(second, first)

    def test_dense_backward_shapes(self) -> None:
        layer = Dense(3, 2, seed=1)
        inputs = Tensor([[1, 2, 3], [4, 5, 6]])

        output = layer.forward(inputs)
        gradients = layer.backward(
            Tensor([[1, 1], [1, 1]])
        )

        self.assertEqual(output.shape, (2, 2))
        self.assertEqual(gradients.shape, (2, 3))
        self.assertEqual(layer.gradients()["weights"].shape, (3, 2))
        self.assertEqual(layer.gradients()["bias"].shape, (2,))

    def test_relu(self) -> None:
        layer = ReLU()
        output = layer.forward(Tensor([[-2, 0, 3]]))

        self.assertEqual(output.to_list(), [[0.0, 0.0, 3.0]])

        gradient = layer.backward(Tensor([[1, 1, 1]]))
        self.assertEqual(gradient.to_list(), [[0.0, 0.0, 1.0]])

    def test_tanh(self) -> None:
        layer = Tanh()
        output = layer.forward(Tensor([[0.0]]))
        gradient = layer.backward(Tensor([[1.0]]))

        self.assertTrue(math.isclose(output.to_list()[0][0], 0.0))
        self.assertTrue(math.isclose(gradient.to_list()[0][0], 1.0))


if __name__ == "__main__":
    unittest.main()
