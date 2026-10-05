"""Tests for the bundled tiny language-model corpora."""

from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = ROOT / "data" / "samples" / "tiny_corpus.txt"
VALIDATION_PATH = ROOT / "data" / "samples" / "tiny_validation.txt"

ARABIC_LETTERS = set("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
ARABIC_EXTRAS = set("ءأإآؤئـةى")
ARABIC_MARKS = set("ًٌٍَُِّْ")


class CorpusTests(unittest.TestCase):
    def test_training_corpus_has_full_arabic_character_coverage(self) -> None:
        text = TRAINING_PATH.read_text(encoding="utf-8")
        required = ARABIC_LETTERS | ARABIC_EXTRAS | ARABIC_MARKS

        self.assertTrue(required.issubset(set(text)))

        for character in ARABIC_LETTERS:
            self.assertIn(f"ـ{character}ـ", text)

    def test_training_corpus_has_more_than_25000_characters(self) -> None:
        text = TRAINING_PATH.read_text(encoding="utf-8")
        self.assertGreater(len(text), 25_000)

    def test_validation_corpus_uses_known_training_characters(self) -> None:
        training = TRAINING_PATH.read_text(encoding="utf-8")
        validation = VALIDATION_PATH.read_text(encoding="utf-8")

        self.assertTrue(set(validation).issubset(set(training)))


if __name__ == "__main__":
    unittest.main()
