import tempfile
import unittest
from pathlib import Path

from src.skeai.tokenizer import CharacterTokenizer


class CharacterTokenizerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = CharacterTokenizer()
        self.tokenizer.fit(
            [
                "hello",
                "مرحبا",
                "كيف حالك",
            ]
        )

    def test_contains_both_languages(self) -> None:
        english = self.tokenizer.encode("hello")
        arabic = self.tokenizer.encode("مرحبا")

        self.assertEqual(self.tokenizer.decode(english), "hello")
        self.assertEqual(self.tokenizer.decode(arabic), "مرحبا")

    def test_special_tokens(self) -> None:
        encoded = self.tokenizer.encode("hello", add_bos=True, add_eos=True)

        self.assertEqual(encoded[0], self.tokenizer.bos_id)
        self.assertEqual(encoded[-1], self.tokenizer.eos_id)
        self.assertEqual(self.tokenizer.decode(encoded), "hello")

    def test_unknown_character(self) -> None:
        encoded = self.tokenizer.encode("☃")
        self.assertEqual(encoded, [self.tokenizer.unk_id])

    def test_save_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokenizer.json"

            self.tokenizer.save(path)
            loaded = CharacterTokenizer.load(path)

            self.assertEqual(
                loaded.encode("مرحبا"),
                self.tokenizer.encode("مرحبا"),
            )


if __name__ == "__main__":
    unittest.main()
