import math
import unittest

from src.skeai.loss import CrossEntropyLoss, MeanSquaredError
from src.skeai.tensor import Tensor


class LossTests(unittest.TestCase):
    def test_mean_squared_error(self) -> None:
        loss = MeanSquaredError()
        predictions = Tensor([[1.0, 3.0]])
        targets = Tensor([[1.0, 1.0]])

        self.assertAlmostEqual(loss.forward(predictions, targets), 2.0)
        self.assertEqual(
            loss.backward(predictions, targets).to_list(),
            [[0.0, 2.0]],
        )

    def test_cross_entropy(self) -> None:
        loss = CrossEntropyLoss()
        logits = Tensor([[2.0, 0.0]])

        value = loss.forward(logits, [0])
        self.assertGreater(value, 0.0)
        self.assertLess(value, 1.0)

        gradient = loss.backward().to_list()[0]
        self.assertAlmostEqual(sum(gradient), 0.0)
        self.assertLess(gradient[0], 0.0)
        self.assertGreater(gradient[1], 0.0)

    def test_cross_entropy_validates_targets(self) -> None:
        loss = CrossEntropyLoss()

        with self.assertRaises(ValueError):
            loss.forward(Tensor([[1.0, 2.0]]), [2])

        with self.assertRaises(ValueError):
            loss.forward(Tensor([[1.0, 2.0]]), [0, 1])


if __name__ == "__main__":
    unittest.main()
