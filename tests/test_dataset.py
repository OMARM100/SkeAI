import unittest

from src.skeai.dataset import CharacterLanguageDataset
from src.skeai.tokenizer import CharacterTokenizer


class DatasetTests(unittest.TestCase):
    def test_creates_next_token_batches(self) -> None:
        tokenizer = CharacterTokenizer()
        text = "hello مرحبا"
        tokenizer.fit([text])

        dataset = CharacterLanguageDataset(
            text=text,
            tokenizer=tokenizer,
            context_length=4,
        )

        batches = dataset.all_batches(batch_size=3)

        self.assertGreater(len(dataset), 0)
        self.assertGreater(len(batches), 0)

        inputs, targets = batches[0]
        self.assertEqual(inputs.shape[1], 4 * tokenizer.vocab_size)
        self.assertEqual(inputs.shape[0], len(targets))

    def test_invalid_context(self) -> None:
        tokenizer = CharacterTokenizer()
        tokenizer.fit(["hello"])

        with self.assertRaises(ValueError):
            CharacterLanguageDataset("hi", tokenizer, context_length=4)


if __name__ == "__main__":
    unittest.main()
