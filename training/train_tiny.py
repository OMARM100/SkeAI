"""Train the first SkeAI character language model.

Run from the repository root:

    python -m training.train_tiny
"""

from __future__ import annotations

from pathlib import Path

from src.skeai.dataset import CharacterLanguageDataset
from src.skeai.language_model import TinyCharacterLanguageModel
from src.skeai.loss import CrossEntropyLoss
from src.skeai.optimizer import SGD
from src.skeai.tokenizer import CharacterTokenizer
from src.skeai.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "samples" / "tiny_corpus.txt"


def main() -> None:
    text = CORPUS_PATH.read_text(encoding="utf-8")

    tokenizer = CharacterTokenizer()
    tokenizer.fit([text])

    dataset = CharacterLanguageDataset(
        text=text,
        tokenizer=tokenizer,
        context_length=8,
    )

    model = TinyCharacterLanguageModel(
        tokenizer=tokenizer,
        context_length=8,
        hidden_size=32,
    )

    trainer = Trainer(
        model=model.network,
        optimizer=SGD(learning_rate=0.05),
        loss=CrossEntropyLoss(),
    )

    batches = dataset.all_batches(batch_size=16)

    def report(epoch: int, batch: int, loss: float) -> None:
        if batch == 1 or epoch == 1:
            print(f"epoch={epoch:03d} batch={batch:02d} loss={loss:.6f}")

    history = trainer.train_batches(
        batches,
        epochs=50,
        callback=report,
    )

    print(f"vocabulary_size={tokenizer.vocab_size}")
    print(f"training_examples={len(dataset)}")
    print(f"initial_loss={history[0]:.6f}")
    print(f"final_loss={history[-1]:.6f}")

    for prompt in ["hello", "مرحبا", "I ", "أنا "]:
        print(f"{prompt!r} -> {model.generate(prompt, max_new_tokens=24)}")


if __name__ == "__main__":
    main()
