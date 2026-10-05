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

    def test_generation_preserves_prompt_with_unknown_character(self) -> None:
        prompt = "🙂"
        generated = self.model.generate(prompt, max_new_tokens=0)
        self.assertEqual(generated, prompt)

    def test_memorized_response_normalizes_prompt(self) -> None:
        self.model.set_response_memory({"كيف حالك؟": "أنا بخير."})
        self.assertEqual(self.model.respond("  كيف حالك ؟ "), "أنا بخير.")
    def test_fuzzy_memorized_response(self) -> None:
        self.model.set_response_memory({"ما اسمك": "اسمي SkeAI."})
        self.assertEqual(self.model.respond("ما هو اسمك"), "اسمي SkeAI.")

    def test_dynamic_time_response(self) -> None:
        response = self.model.respond("كم الساعة")
        self.assertRegex(response, r"^الساعة الآن \\d{2}:\\d{2}\\.$")

    def test_invalid_generation_controls(self) -> None:
        with self.assertRaises(ValueError):
            self.model.generate("hello", top_k=0)

        with self.assertRaises(ValueError):
            self.model.generate("hello", repetition_penalty=0.9)

        with self.assertRaises(ValueError):
            self.model.generate("hello", no_repeat_ngram_size=-1)

    def test_invalid_temperature(self) -> None:
        with self.assertRaises(ValueError):
            self.model.generate("hello", temperature=0.0)


if __name__ == "__main__":
    unittest.main()
