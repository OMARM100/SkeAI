import tempfile
import unittest
from pathlib import Path

from src.skeai.level2.tokenizer import HybridTokenizer
from src.skeai.level2.transformer import TinyTransformerLM, TransformerConfig
from src.skeai.level2.trainer import Level2Trainer
from src.skeai.loss import CrossEntropyLoss
from src.skeai.optimizer import SGD
from src.skeai.tensor import Tensor


class Level2TokenizerTests(unittest.TestCase):
    def test_frequent_units_are_compacted(self) -> None:
        tokenizer = HybridTokenizer()
        tokenizer.fit(
            [
                "مرحبا مرحبا مرحبا",
                "hello hello hello",
                "Python Python",
            ],
            max_units=32,
            min_frequency=2,
        )

        self.assertIn("مرحبا", tokenizer.token_to_id)
        self.assertIn("hello", tokenizer.token_to_id)

        encoded = tokenizer.encode("مرحبا hello")
        self.assertLess(len(encoded), len("مرحبا hello"))

    def test_unseen_text_falls_back_to_characters(self) -> None:
        tokenizer = HybridTokenizer()
        tokenizer.fit(["hello"], max_units=16, min_frequency=2)

        encoded = tokenizer.encode("h")
        self.assertEqual(tokenizer.decode(encoded), "h")

    def test_save_and_load(self) -> None:
        tokenizer = HybridTokenizer()
        tokenizer.fit(["hello hello", "مرحبا مرحبا"])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "level2_tokenizer.json"
            tokenizer.save(path)
            loaded = HybridTokenizer.load(path)

        self.assertEqual(
            loaded.encode("hello مرحبا"),
            tokenizer.encode("hello مرحبا"),
        )


class Level2TransformerTests(unittest.TestCase):
    def make_model(self) -> TinyTransformerLM:
        config = TransformerConfig(
            context_length=16,
            d_model=16,
            n_heads=2,
            feed_forward_size=32,
            n_layers=1,
            max_vocab_size=64,
            seed=7,
        )
        return TinyTransformerLM(vocab_size=32, config=config)

    def test_forward_shape(self) -> None:
        model = self.make_model()
        logits = model.forward([1, 2, 3, 4])
        self.assertEqual(logits.shape, (4, 32))

    def test_next_logits(self) -> None:
        model = self.make_model()
        logits = model.next_logits([1, 2, 3])
        self.assertEqual(len(logits), 32)

    def test_lm_head_gradient_matches_finite_difference(self) -> None:
        model = self.make_model()
        inputs = [1, 2, 3, 4]
        targets = [2, 3, 4, 5]
        loss = CrossEntropyLoss()

        logits = model.forward(inputs)
        loss_value = loss.forward(logits, targets)
        gradients = model.backward(loss.backward())
        analytic = gradients["lm_head"].to_list()[0][0]

        weights = model.lm_head.to_list()
        epsilon = 1e-5

        weights[0][0] += epsilon
        model.lm_head = Tensor(weights)
        plus = loss.forward(model.forward(inputs), targets)

        weights[0][0] -= 2.0 * epsilon
        model.lm_head = Tensor(weights)
        minus = loss.forward(model.forward(inputs), targets)

        weights[0][0] += epsilon
        model.lm_head = Tensor(weights)

        numerical = (plus - minus) / (2.0 * epsilon)
        self.assertAlmostEqual(analytic, numerical, places=3)
        self.assertGreater(loss_value, 0.0)

    def test_attention_training_reduces_loss(self) -> None:
        config = TransformerConfig(
            context_length=6,
            d_model=8,
            n_heads=2,
            feed_forward_size=16,
            n_layers=1,
            max_vocab_size=16,
            seed=5,
        )
        model = TinyTransformerLM(vocab_size=8, config=config)
        trainer = Level2Trainer(
            model=model,
            optimizer=SGD(learning_rate=0.02),
        )

        inputs = [1, 2, 1, 2, 1, 2]
        targets = [2, 1, 2, 1, 2, 1]

        initial = trainer.evaluate(inputs, targets)
        for _ in range(20):
            trainer.train_step(inputs, targets)
        final = trainer.evaluate(inputs, targets)

        self.assertLess(final, initial)

    def test_checkpoint_round_trip(self) -> None:
        model = self.make_model()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "level2.json"
            model.save_checkpoint(path)
            loaded = TinyTransformerLM.load_checkpoint(path)

        self.assertEqual(loaded.parameter_count(), model.parameter_count())
        self.assertEqual(
            loaded.forward([1, 2, 3]).to_list(),
            model.forward([1, 2, 3]).to_list(),
        )


if __name__ == "__main__":
    unittest.main()
