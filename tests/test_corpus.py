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

    def test_training_corpus_has_about_4000_characters(self) -> None:
        text = TRAINING_PATH.read_text(encoding="utf-8")
        self.assertGreaterEqual(len(text), 3900)
        self.assertLessEqual(len(text), 4050)

    def test_training_corpus_has_unique_natural_language_lines(self) -> None:
        lines = [
            line.strip()
            for line in TRAINING_PATH.read_text(encoding="utf-8").splitlines()
            if len(line.strip()) >= 20
        ]
        self.assertGreaterEqual(len(lines), 25)
        self.assertEqual(len(lines), len(set(lines)))

    def test_validation_lines_do_not_repeat_training_lines(self) -> None:
        training_lines = {line.strip() for line in TRAINING_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}
        validation_lines = {line.strip() for line in VALIDATION_PATH.read_text(encoding="utf-8").splitlines() if line.strip()}
        self.assertTrue(validation_lines)
        self.assertTrue(validation_lines.isdisjoint(training_lines))

    def test_validation_corpus_uses_known_training_characters(self) -> None:
        training = TRAINING_PATH.read_text(encoding="utf-8")
        validation = VALIDATION_PATH.read_text(encoding="utf-8")

        self.assertTrue(set(validation).issubset(set(training)))


if __name__ == "__main__":
    unittest.main()
