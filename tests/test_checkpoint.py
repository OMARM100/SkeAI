import tempfile
import unittest
from pathlib import Path

from src.skeai.language_model import TinyCharacterLanguageModel
from src.skeai.tokenizer import CharacterTokenizer


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_round_trip(self) -> None:
        tokenizer = CharacterTokenizer()
        tokenizer.fit(["hello مرحبا"])

        model = TinyCharacterLanguageModel(
            tokenizer,
            context_length=4,
            hidden_size=8,
            seed=7,
        )

        before = model.forward(tokenizer.encode("hell")).to_list()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.save_checkpoint(path)

            loaded = TinyCharacterLanguageModel.load_checkpoint(path)
            after = loaded.forward(
                loaded.tokenizer.encode("hell")
            ).to_list()

        self.assertEqual(
            loaded.tokenizer.id_to_token,
            tokenizer.id_to_token,
        )
        self.assertEqual(before, after)
        self.assertEqual(
            loaded.context_length,
            model.context_length,
        )
        model.set_response_memory({"مرحبا": "أهلًا بك!"})

        self.assertEqual(
            loaded.hidden_size,
            model.hidden_size,
        )
        self.assertEqual(
            loaded.memorized_response("مرحبا"),
            "أهلًا بك!",
        )


if __name__ == "__main__":
    unittest.main()
