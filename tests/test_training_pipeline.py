import unittest

from src.skeai.level2.tokenizer import HybridTokenizer
from training.train_level2 import (
    _fixed_target_window,
    build_dialogue_samples,
    build_response_focused_samples,
    make_samples,
    scheduled_learning_rate,
)


class TrainingPipelineTests(unittest.TestCase):
    def make_tokenizer(self) -> HybridTokenizer:
        tokenizer = HybridTokenizer()
        tokenizer.fit(
            [
                "حالة SkeAI: أنا SkeAI | نوعي: ذكاء اصطناعي | أنا لست إنسانًا | مرحلة التطور: بداية التعلم",
                "المستخدم: مرحبا\nSkeAI: أهلًا بك أنا SkeAI",
                "المستخدم: ما اسمك؟\nSkeAI: اسمي SkeAI",
                "المستخدم: ما هو الذكاء الاصطناعي؟\nSkeAI: هو نظام يتعلم من البيانات",
            ],
            max_units=128,
            min_frequency=1,
        )
        return tokenizer

    def test_response_focused_samples_are_fixed_context_windows(self) -> None:
        tokenizer = self.make_tokenizer()
        conversations = [
            [
                ("مرحبا", "أهلًا بك أنا SkeAI وكيف يمكنني مساعدتك؟"),
                ("ما اسمك؟", "اسمي SkeAI وأنا ذكاء اصطناعي قيد التطوير."),
                ("ما هو الذكاء الاصطناعي؟", "هو نظام يتعلم من البيانات ويستخدمها لتوليد الردود."),
            ]
        ]

        full = build_dialogue_samples(conversations, tokenizer, context_length=8)
        focused = build_response_focused_samples(
            conversations,
            tokenizer,
            context_length=8,
        )

        self.assertTrue(full)
        self.assertTrue(focused)
        self.assertTrue(
            all(
                0 < len(inputs) <= 8
                and len(inputs) == len(targets)
                and len(weights) == len(targets)
                for inputs, targets, weights in focused
            )
        )
        self.assertTrue(
            all(
                any(weight == 1.0 for weight in weights)
                and all(weight in (0.25, 1.0) for weight in weights)
                for _, _, weights in focused
            )
        )

    def test_fixed_target_window_keeps_early_response_context(self) -> None:
        inputs, targets = _fixed_target_window(
            list(range(20)),
            target_index=4,
            context_length=8,
        )
        self.assertEqual(inputs, [0, 1, 2, 3])
        self.assertEqual(targets, [1, 2, 3, 4])

        inputs, targets = _fixed_target_window(
            list(range(20)),
            target_index=15,
            context_length=8,
        )
        self.assertEqual(inputs, list(range(7, 15)))
        self.assertEqual(targets, list(range(8, 16)))

    def test_next_token_samples_are_aligned(self) -> None:
        tokens = list(range(30))
        samples = make_samples(tokens, context_length=8, stride=3)
        self.assertTrue(samples)
        for inputs, targets in samples:
            self.assertEqual(len(inputs), len(targets))
            self.assertEqual(targets[:-1], inputs[1:])

    def test_learning_rate_warms_up_then_decays(self) -> None:
        values = [
            scheduled_learning_rate(
                0.003,
                epoch=epoch,
                total_epochs=6,
                warmup_epochs=2,
                final_scale=0.35,
            )
            for epoch in range(1, 7)
        ]

        self.assertAlmostEqual(values[0], 0.0015, places=12)
        self.assertAlmostEqual(values[1], 0.0030, places=12)
        self.assertGreater(values[1], values[2])
        self.assertGreater(values[2], values[3])
        self.assertGreater(values[3], values[4])
        self.assertGreater(values[4], values[5])
        self.assertAlmostEqual(values[-1], 0.00105, places=12)


if __name__ == "__main__":
    unittest.main()
