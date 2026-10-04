"""Train the first SkeAI character language model.

Run from the repository root:

    python -m training.train_tiny

To continue training from the last checkpoint:

    python -m training.train_tiny --resume
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.skeai.dataset import CharacterLanguageDataset
from src.skeai.language_model import TinyCharacterLanguageModel
from src.skeai.loss import CrossEntropyLoss
from src.skeai.optimizer import SGD
from src.skeai.tokenizer import CharacterTokenizer
from src.skeai.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "samples" / "tiny_corpus.txt"
CHECKPOINT_PATH = ROOT / "models" / "tiny_character_model.json"


def load_model(resume: bool, text: str) -> TinyCharacterLanguageModel:
    if resume:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {CHECKPOINT_PATH}. "
                "Run without --resume first."
            )
        return TinyCharacterLanguageModel.load_checkpoint(CHECKPOINT_PATH)

    tokenizer = CharacterTokenizer()
    tokenizer.fit([text])

    return TinyCharacterLanguageModel(
        tokenizer=tokenizer,
        context_length=8,
        hidden_size=32,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SkeAI tiny language model.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the previous checkpoint before continuing training.",
    )
    args = parser.parse_args()

    text = CORPUS_PATH.read_text(encoding="utf-8")
    model = load_model(args.resume, text)

    dataset = CharacterLanguageDataset(
        text=text,
        tokenizer=model.tokenizer,
        context_length=model.context_length,
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

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(CHECKPOINT_PATH)

    print(f"checkpoint={CHECKPOINT_PATH}")
    print(f"vocabulary_size={model.tokenizer.vocab_size}")
    print(f"training_examples={len(dataset)}")
    print(f"initial_loss={history[0]:.6f}")
    print(f"final_loss={history[-1]:.6f}")

    for prompt in ["hello", "مرحبا", "I ", "أنا "]:
        print(f"{prompt!r} -> {model.generate(prompt, max_new_tokens=24)}")


if __name__ == "__main__":
    main()
