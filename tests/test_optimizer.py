import unittest

from src.skeai.layers import Dense
from src.skeai.optimizer import SGD
from src.skeai.tensor import Tensor


class OptimizerTests(unittest.TestCase):
    def test_sgd_step_updates_parameters(self) -> None:
        layer = Dense(1, 1, seed=1)
        layer.weights = Tensor([[1.0]])
        layer.bias = Tensor([0.5])
        layer.grad_weights = Tensor([[0.2]])
        layer.grad_bias = Tensor([0.1])

        optimizer = SGD(learning_rate=0.5)
        optimizer.step(layer.parameters(), layer.gradients())

        self.assertAlmostEqual(layer.parameters()["weights"].to_list()[0][0], 0.9)
        self.assertAlmostEqual(layer.parameters()["bias"].to_list()[0], 0.45)

    def test_rejects_invalid_learning_rate(self) -> None:
        with self.assertRaises(ValueError):
            SGD(learning_rate=0.0)


if __name__ == "__main__":
    unittest.main()
