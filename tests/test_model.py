import unittest

from src.skeai.layers import Dense, Tanh
from src.skeai.model import Sequential
from src.skeai.tensor import Tensor


class SequentialTests(unittest.TestCase):
    def test_forward_and_backward(self) -> None:
        model = Sequential(
            [
                Dense(2, 3, seed=1),
                Tanh(),
                Dense(3, 2, seed=2),
            ]
        )

        inputs = Tensor([[1.0, 0.0], [0.0, 1.0]])
        outputs = model.forward(inputs)

        self.assertEqual(outputs.shape, (2, 2))

        gradient = Tensor([[1.0, 0.0], [0.0, 1.0]])
        input_gradient = model.backward(gradient)

        self.assertEqual(input_gradient.shape, (2, 2))
        self.assertEqual(len(model.parameters()), 4)
        self.assertEqual(len(model.gradients()), 4)


if __name__ == "__main__":
    unittest.main()
