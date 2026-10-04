import unittest

from src.skeai.language_model import TinyCharacterLanguageModel
from src.skeai.tokenizer import CharacterTokenizer


class TinyLanguageModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()
        corpus = "hello hello مرحبا مرحبا"
        self.tokenizer.fit([corpus])
        self.model = TinyCharacterLanguageModel(
            self.tokenizer,
            context_length=4,
            hidden_size=8,
        )

    def test_forward_shape(self) -> None:
        logits = self.model.forward(self.tokenizer.encode("hell"))
        self.assertEqual(
            logits.shape,
            (1, self.tokenizer.vocab_size),
        )

    def test_generation_returns_string(self) -> None:
        generated = self.model.generate("hell", max_new_tokens=4, seed=1)
        self.assertIsInstance(generated, str)

    def test_invalid_temperature(self) -> None:
        with self.assertRaises(ValueError):
            self.model.generate("hello", temperature=0.0)


if __name__ == "__main__":
    unittest.main()
