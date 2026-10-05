import unittest

from src.skeai.benchmark import benchmark
from src.skeai.layers import Dense
from src.skeai.loss import CrossEntropyLoss
from src.skeai.model import Sequential
from src.skeai.optimizer import SGD
from src.skeai.tensor import Tensor
from src.skeai.trainer import Trainer


class PerformanceTests(unittest.TestCase):
    def test_benchmark_reports_average_time(self) -> None:
        result = benchmark(
            "tiny",
            lambda: 1 + 1,
            repeats=2,
            warmups=1,
        )

        self.assertEqual(result.name, "tiny")
        self.assertEqual(result.repeats, 2)
        self.assertGreaterEqual(result.elapsed_seconds, 0.0)
        self.assertGreaterEqual(result.average_milliseconds, 0.0)

    def test_parameter_count(self) -> None:
        model = Sequential(
            [
                Dense(4, 3),
                Dense(3, 2),
            ]
        )
        self.assertEqual(model.parameter_count(), (4 * 3 + 3) + (3 * 2 + 2))

    def test_trainer_timing(self) -> None:
        model = Sequential(
            [
                Dense(3, 4),
                Dense(4, 2),
            ]
        )
        trainer = Trainer(
            model=model,
            optimizer=SGD(learning_rate=0.01),
            loss=CrossEntropyLoss(),
            enable_timing=True,
        )

        loss = trainer.train_step(
            Tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            [0, 1],
        )

        self.assertGreater(loss, 0.0)
        for key in (
            "forward_ms",
            "loss_ms",
            "backward_ms",
            "optimizer_ms",
            "total_ms",
        ):
            self.assertIn(key, trainer.last_step_timing)
            self.assertGreaterEqual(trainer.last_step_timing[key], 0.0)


if __name__ == "__main__":
    unittest.main()
