"""Tests for the bundled tiny language-model corpora."""

from __future__ import annotations

import re
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
TRAINING_PATH = ROOT / "data" / "samples" / "tiny_corpus.txt"
VALIDATION_PATH = ROOT / "data" / "samples" / "tiny_validation.txt"
COVERAGE_PATH = ROOT / "data" / "samples" / "tiny_character_coverage.txt"

ARABIC_LETTERS = set("ابتثجحخدذرزسشصضطظعغفقكلمنهوي")
ARABIC_EXTRAS = set("ءأإآؤئـةى")
ARABIC_MARKS = set("ًٌٍَُِّْ")


def _words(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z\u0600-\u06FF]+", text.lower()))


class CorpusTests(unittest.TestCase):
    def test_training_corpus_has_natural_language_diversity(self) -> None:
        text = TRAINING_PATH.read_text(encoding="utf-8")
        lines = [
            line.strip()
            for line in text.splitlines()
            if len(line.strip()) >= 20
        ]

        self.assertGreaterEqual(len(lines), 45)
        self.assertEqual(len(lines), len(set(lines)))

    def test_training_corpus_has_about_4000_characters(self) -> None:
        text = TRAINING_PATH.read_text(encoding="utf-8")
        self.assertGreaterEqual(len(text), 3900)
        self.assertLessEqual(len(text), 4050)

    def test_character_coverage_file_has_all_arabic_forms(self) -> None:
        text = COVERAGE_PATH.read_text(encoding="utf-8")
        required = ARABIC_LETTERS | ARABIC_EXTRAS | ARABIC_MARKS

        self.assertTrue(required.issubset(set(text)))

        for character in ARABIC_LETTERS:
            self.assertIn(f"ـ{character}ـ", text)

    def test_validation_lines_do_not_repeat_training_lines(self) -> None:
        training_lines = {
            line.strip()
            for line in TRAINING_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        validation_lines = {
            line.strip()
            for line in VALIDATION_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

        self.assertTrue(validation_lines)
        self.assertTrue(validation_lines.isdisjoint(training_lines))

    def test_validation_contains_unseen_words(self) -> None:
        training = TRAINING_PATH.read_text(encoding="utf-8")
        validation = VALIDATION_PATH.read_text(encoding="utf-8")

        unseen = _words(validation) - _words(training)
        self.assertGreaterEqual(len(unseen), 8)

    def test_validation_corpus_uses_known_training_characters(self) -> None:
        training = TRAINING_PATH.read_text(encoding="utf-8")
        validation = VALIDATION_PATH.read_text(encoding="utf-8")

        self.assertTrue(set(validation).issubset(set(training)))


if __name__ == "__main__":
    unittest.main()
