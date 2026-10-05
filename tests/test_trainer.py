import unittest

from src.skeai.layers import Dense, Tanh
from src.skeai.model import Sequential
from src.skeai.optimizer import SGD
from src.skeai.tensor import Tensor
from src.skeai.trainer import Trainer


class TrainerTests(unittest.TestCase):
    def test_training_reduces_loss_on_tiny_dataset(self) -> None:
        # Two simple classes:
        # [1, 0] -> class 0
        # [0, 1] -> class 1
        inputs = Tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        targets = [0, 1]

        model = Sequential(
            [
                Dense(2, 4, seed=7),
                Tanh(),
                Dense(4, 2, seed=8),
            ]
        )
        trainer = Trainer(
            model=model,
            optimizer=SGD(learning_rate=0.25),
        )

        first_loss = trainer.train_step(inputs, targets)

        for _ in range(200):
            trainer.train_step(inputs, targets)

        final_loss = trainer.train_step(inputs, targets)

        self.assertLess(final_loss, first_loss)

        predictions = model.forward(inputs).to_list()
        self.assertGreater(predictions[0][0], predictions[0][1])
        self.assertGreater(predictions[1][1], predictions[1][0])

    def test_evaluation_does_not_update_parameters(self) -> None:
        inputs = Tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )
        targets = [0, 1]

        model = Sequential(
            [
                Dense(2, 4, seed=17),
                Tanh(),
                Dense(4, 2, seed=18),
            ]
        )
        trainer = Trainer(
            model=model,
            optimizer=SGD(learning_rate=0.25),
        )

        before = model.state_dict()
        loss = trainer.evaluate_batches([(inputs, targets)])
        after = model.state_dict()

        self.assertGreater(loss, 0.0)
        self.assertEqual(before, after)

if __name__ == "__main__":
    unittest.main()
