import unittest

from src.skeai.layers import Dense, Tanh
from src.skeai.model import Sequential
from src.skeai.optimizer import SGD
from src.skeai.tensor import Tensor
from src.skeai.trainer import Trainer


class BatchTrainerTests(unittest.TestCase):
    def test_train_batches_reduces_average_loss(self) -> None:
        model = Sequential(
            [
                Dense(2, 4, seed=5),
                Tanh(),
                Dense(4, 2, seed=6),
            ]
        )
        trainer = Trainer(model, SGD(learning_rate=0.2))

        batches = [
            (Tensor([[1.0, 0.0]]), [0]),
            (Tensor([[0.0, 1.0]]), [1]),
        ]

        history = trainer.train_batches(batches, epochs=20)

        self.assertEqual(len(history), 20)
        self.assertLess(history[-1], history[0])


if __name__ == "__main__":
    unittest.main()
