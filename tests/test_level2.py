import tempfile
import unittest
from pathlib import Path

from src.skeai.level2.tokenizer import HybridTokenizer
from src.skeai.level2.transformer import TinyTransformerLM, TransformerConfig
from src.skeai.level2.trainer import Level2Trainer
from src.skeai.loss import CrossEntropyLoss
from src.skeai.optimizer import SGD
from src.skeai.tensor import Tensor
from src.skeai.level2.transformer import _causal_softmax, _softmax_backward


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

    def test_native_causal_softmax(self) -> None:
        values = [
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [3.0, 1.0, -2.0],
        ]
        result = _causal_softmax(values)

        self.assertAlmostEqual(result[0][0], 1.0, places=12)
        self.assertAlmostEqual(result[0][1], 0.0, places=12)
        self.assertAlmostEqual(result[1][0], 0.5, places=12)
        self.assertAlmostEqual(result[1][1], 0.5, places=12)
        self.assertAlmostEqual(result[2][0], 0.8756005950630876, places=10)
        self.assertAlmostEqual(result[2][1], 0.11849965453500957, places=10)
        self.assertAlmostEqual(result[2][2], 0.005899750401902781, places=10)

    def test_native_softmax_backward(self) -> None:
        probabilities = [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ]
        gradients = [
            [2.0, 3.0, 4.0],
            [1.0, 3.0, 5.0],
        ]
        result = _softmax_backward(probabilities, gradients)

        self.assertAlmostEqual(result[0][0], 0.0, places=12)
        self.assertAlmostEqual(result[0][1], 0.0, places=12)
        self.assertAlmostEqual(result[1][0], -0.5, places=12)
        self.assertAlmostEqual(result[1][1], 0.5, places=12)
        self.assertAlmostEqual(result[1][2], 0.0, places=12)

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

    def test_attention_q_gradient_matches_finite_difference(self) -> None:
        config = TransformerConfig(
            context_length=4,
            d_model=8,
            n_heads=2,
            feed_forward_size=12,
            n_layers=1,
            max_vocab_size=12,
            seed=9,
        )
        model = TinyTransformerLM(vocab_size=8, config=config)
        loss = CrossEntropyLoss()

        inputs = [1, 2, 3, 4]
        targets = [2, 3, 4, 1]

        logits = model.forward(inputs)
        loss.forward(logits, targets)
        gradients = model.backward(loss.backward())
        analytic = gradients["blocks.0.wq"].to_list()[0][0]

        weights = model.blocks[0]["wq"].to_list()
        epsilon = 1e-5

        weights[0][0] += epsilon
        model.blocks[0]["wq"] = Tensor(weights)
        plus = loss.forward(model.forward(inputs), targets)

        weights[0][0] -= 2.0 * epsilon
        model.blocks[0]["wq"] = Tensor(weights)
        minus = loss.forward(model.forward(inputs), targets)

        weights[0][0] += epsilon
        model.blocks[0]["wq"] = Tensor(weights)

        numerical = (plus - minus) / (2.0 * epsilon)
        self.assertAlmostEqual(analytic, numerical, places=3)

    def test_native_batch_training_matches_sequential_updates(self) -> None:
        config = TransformerConfig(
            context_length=6,
            d_model=8,
            n_heads=2,
            feed_forward_size=16,
            n_layers=1,
            max_vocab_size=16,
            seed=13,
        )
        inputs = [
            [1, 2, 1, 2, 1, 2],
            [2, 3, 2, 3, 2, 3],
        ]
        targets = [
            [2, 1, 2, 1, 2, 1],
            [3, 2, 3, 2, 3, 2],
        ]

        model_reference = TinyTransformerLM(vocab_size=8, config=config)
        model_single_batch = TinyTransformerLM(vocab_size=8, config=config)
        trainer_reference = Level2Trainer(
            model=model_reference,
            optimizer=SGD(learning_rate=0.01),
        )
        trainer_single_batch = Level2Trainer(
            model=model_single_batch,
            optimizer=SGD(learning_rate=0.01),
        )

        reference_first = trainer_reference.train_step(
            inputs[0],
            targets[0],
        )
        batch_first = trainer_single_batch.train_batch(
            [inputs[0]],
            [targets[0]],
        )
        self.assertAlmostEqual(batch_first, reference_first, places=10)

        for parameter_reference, parameter_batch in zip(
            model_reference.parameters().values(),
            model_single_batch.parameters().values(),
        ):
            self.assertEqual(
                parameter_reference.to_list(),
                parameter_batch.to_list(),
            )

        reference_second = trainer_reference.train_step(
            inputs[1],
            targets[1],
        )
        batch_second = trainer_single_batch.train_batch(
            [inputs[1]],
            [targets[1]],
        )
        self.assertAlmostEqual(batch_second, reference_second, places=10)

        for parameter_reference, parameter_batch in zip(
            model_reference.parameters().values(),
            model_single_batch.parameters().values(),
        ):
            self.assertEqual(
                parameter_reference.to_list(),
                parameter_batch.to_list(),
            )

        model_sequential = TinyTransformerLM(vocab_size=8, config=config)
        model_batch = TinyTransformerLM(vocab_size=8, config=config)
        trainer_sequential = Level2Trainer(
            model=model_sequential,
            optimizer=SGD(learning_rate=0.01),
        )
        trainer_batch = Level2Trainer(
            model=model_batch,
            optimizer=SGD(learning_rate=0.01),
        )

        sequential_loss = (
            trainer_sequential.train_step(inputs[0], targets[0])
            + trainer_sequential.train_step(inputs[1], targets[1])
        ) / 2.0
        batch_loss = trainer_batch.train_batch(inputs, targets)

        self.assertAlmostEqual(batch_loss, sequential_loss, places=10)

        for parameter_sequential, parameter_batch in zip(
            model_sequential.parameters().values(),
            model_batch.parameters().values(),
        ):
            self.assertEqual(
                parameter_sequential.to_list(),
                parameter_batch.to_list(),
            )

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

        
    def test_weighted_training_ignores_zero_weight_targets(self) -> None:
        config = TransformerConfig(
            context_length=6,
            d_model=8,
            n_heads=2,
            feed_forward_size=16,
            n_layers=1,
            max_vocab_size=16,
            seed=23,
        )
        inputs = [1, 2, 1, 2, 1, 2]
        targets_a = [7, 7, 2, 1, 2, 1]
        targets_b = [6, 3, 2, 1, 2, 1]
        weights = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

        model_a = TinyTransformerLM(vocab_size=8, config=config)
        model_b = TinyTransformerLM(vocab_size=8, config=config)
        trainer_a = Level2Trainer(
            model=model_a,
            optimizer=SGD(learning_rate=0.01),
        )
        trainer_b = Level2Trainer(
            model=model_b,
            optimizer=SGD(learning_rate=0.01),
        )

        loss_a = trainer_a.train_batch(
            [inputs],
            [targets_a],
            [weights],
        )
        loss_b = trainer_b.train_batch(
            [inputs],
            [targets_b],
            [weights],
        )

        self.assertAlmostEqual(loss_a, loss_b, places=10)

        for parameter_a, parameter_b in zip(
            model_a.parameters().values(),
            model_b.parameters().values(),
        ):
            self.assertEqual(parameter_a.to_list(), parameter_b.to_list())

    def test_generation_supports_greedy_and_sampling(self) -> None:
        model = self.make_model()
        prompt = [1, 2, 3]

        greedy = model.generate(
            prompt,
            max_new_tokens=5,
            temperature=0.0,
        )
        sampled_a = model.generate(
            prompt,
            max_new_tokens=5,
            temperature=0.9,
            top_k=4,
            seed=123,
        )
        sampled_b = model.generate(
            prompt,
            max_new_tokens=5,
            temperature=0.9,
            top_k=4,
            seed=123,
        )

        self.assertEqual(len(greedy), len(prompt) + 5)
        self.assertEqual(len(sampled_a), len(prompt) + 5)
        self.assertEqual(sampled_a, sampled_b)
        self.assertTrue(all(0 <= token < model.vocab_size for token in greedy))
        self.assertTrue(all(0 <= token < model.vocab_size for token in sampled_a))

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
