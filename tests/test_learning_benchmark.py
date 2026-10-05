"""Tests for the learning benchmark helpers."""

from __future__ import annotations

import math
import unittest

from tools.benchmark_learning import (
    EvaluationMetrics,
    _is_arabic_character,
    _is_latin_character,
    _logsumexp,
    _target_rank,
    _unseen_word_positions,
)


class LearningBenchmarkTests(unittest.TestCase):
    def test_character_script_detection(self) -> None:
        self.assertTrue(_is_arabic_character("م"))
        self.assertTrue(_is_arabic_character("أ"))
        self.assertTrue(_is_latin_character("A"))
        self.assertTrue(_is_latin_character("z"))
        self.assertFalse(_is_arabic_character("A"))
        self.assertFalse(_is_latin_character("م"))

    def test_target_rank_is_one_based(self) -> None:
        logits = [0.2, 3.0, 1.0, 2.0]
        self.assertEqual(_target_rank(logits, 1), 1)
        self.assertEqual(_target_rank(logits, 3), 2)
        self.assertEqual(_target_rank(logits, 0), 4)

    def test_logsumexp_matches_manual_result(self) -> None:
        values = [0.0, math.log(2.0)]
        self.assertAlmostEqual(_logsumexp(values), math.log(3.0), places=7)

    def test_unseen_word_positions_only_include_new_words(self) -> None:
        text = "hello مرحبا world"
        training_words = {"hello", "world"}
        positions = _unseen_word_positions(text, training_words)

        self.assertIn(text.index("م"), positions)
        self.assertNotIn(text.index("h"), positions)
        self.assertNotIn(text.index("w"), positions)

    def test_metrics_do_not_count_special_tokens(self) -> None:
        metrics = EvaluationMetrics()
        metrics.add("<EOS>", rank=1, nll=0.0)
        self.assertEqual(metrics.examples, 0)

        metrics.add("م", rank=2, nll=1.0)
        metrics.add("A", rank=1, nll=0.5)

        self.assertEqual(metrics.examples, 2)
        self.assertAlmostEqual(metrics.top1_accuracy, 0.5)
        self.assertAlmostEqual(metrics.top3_accuracy, 1.0)
        self.assertEqual(metrics.arabic_examples, 1)
        self.assertEqual(metrics.latin_examples, 1)


if __name__ == "__main__":
    unittest.main()
