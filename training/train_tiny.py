"""Train the first SkeAI character language model.

Run from the repository root:

    python -m training.train_tiny

To continue training from the last checkpoint:

    python -m training.train_tiny --resume

Use --epochs to control the number of training epochs. The default is 100.
The script reports training and held-out validation loss so we can distinguish
actual generalization from memorization/overfitting.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

from src.skeai.dataset import CharacterLanguageDataset
from src.skeai.language_model import TinyCharacterLanguageModel
from src.skeai.loss import CrossEntropyLoss
from src.skeai.optimizer import SGD
from src.skeai.tokenizer import CharacterTokenizer
from src.skeai.trainer import Trainer


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "data" / "samples" / "tiny_corpus.txt"
VALIDATION_PATH = ROOT / "data" / "samples" / "tiny_validation.txt"
CHECKPOINT_PATH = ROOT / "models" / "tiny_character_model.json"
DEFAULT_EPOCHS = 100
BATCH_SIZE = 16


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
    total_start = perf_counter()

    parser = argparse.ArgumentParser(
        description="Train SkeAI tiny language model."
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Load the previous checkpoint before continuing training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS}).",
    )
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")

    setup_start = perf_counter()
    text = CORPUS_PATH.read_text(encoding="utf-8")
    validation_text = VALIDATION_PATH.read_text(encoding="utf-8")
    model = load_model(args.resume, text)

    unknown_validation_characters = sorted(
        {
            character
            for character in validation_text
            if character not in model.tokenizer.token_to_id
        },
        key=ord,
    )
    if unknown_validation_characters:
        raise ValueError(
            "Validation corpus contains characters outside the training "
            f"vocabulary: {unknown_validation_characters!r}"
        )

    dataset = CharacterLanguageDataset(
        text=text,
        tokenizer=model.tokenizer,
        context_length=model.context_length,
    )
    validation_dataset = CharacterLanguageDataset(
        text=validation_text,
        tokenizer=model.tokenizer,
        context_length=model.context_length,
    )

    trainer = Trainer(
        model=model.network,
        optimizer=SGD(learning_rate=0.05),
        loss=CrossEntropyLoss(),
        enable_timing=True,
    )

    batches = dataset.all_batches(batch_size=BATCH_SIZE)
    validation_batches = validation_dataset.all_batches(batch_size=BATCH_SIZE)
    total_training_steps = len(batches) * args.epochs
    setup_seconds = perf_counter() - setup_start

    print(
        f"training_plan=epochs:{args.epochs} "
        f"batches_per_epoch:{len(batches)} "
        f"total_steps:{total_training_steps} "
        f"batch_size:{BATCH_SIZE} "
        f"validation_examples:{len(validation_dataset)}"
    )

    def report(epoch: int, batch: int, loss: float) -> None:
        if batch == 1 or epoch == 1:
            print(
                f"epoch={epoch:03d} batch={batch:02d} "
                f"loss={loss:.6f}"
            )

    initial_validation_loss = trainer.evaluate_batches(validation_batches)
    validation_history: list[float] = []

    def report_epoch(
        epoch: int,
        loss: float,
        elapsed_seconds: float,
        examples_per_second: float,
    ) -> None:
        validation_loss = trainer.evaluate_batches(validation_batches)
        validation_history.append(validation_loss)
        timing = trainer.last_step_timing

        print(
            f"epoch={epoch:03d} time={elapsed_seconds:.4f}s "
            f"train_loss={loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"gap={validation_loss - loss:+.6f} "
            f"avg_step={trainer.last_epoch_timing['average_step_ms']:.3f}ms "
            f"examples/s={examples_per_second:.2f} "
            f"last_step=[forward:{timing['forward_ms']:.3f}ms "
            f"loss:{timing['loss_ms']:.3f}ms "
            f"backward:{timing['backward_ms']:.3f}ms "
            f"optimizer:{timing['optimizer_ms']:.3f}ms]"
        )

    training_start = perf_counter()
    history = trainer.train_batches(
        batches,
        epochs=args.epochs,
        callback=report,
        epoch_callback=report_epoch,
    )
    training_seconds = perf_counter() - training_start

    checkpoint_start = perf_counter()
    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_checkpoint(CHECKPOINT_PATH)
    checkpoint_seconds = perf_counter() - checkpoint_start

    final_validation_loss = trainer.evaluate_batches(validation_batches)
    total_seconds = perf_counter() - total_start

    print(f"checkpoint={CHECKPOINT_PATH}")
    print(f"vocabulary_size={model.tokenizer.vocab_size}")
    print(f"training_examples={len(dataset)}")
    print(f"validation_examples={len(validation_dataset)}")
    print(f"parameter_count={model.network.parameter_count()}")
    print(f"initial_loss={history[0]:.6f}")
    print(f"final_loss={history[-1]:.6f}")
    print(f"initial_validation_loss={initial_validation_loss:.6f}")
    print(f"final_validation_loss={final_validation_loss:.6f}")
    print(f"best_validation_loss={min(validation_history):.6f}")
    print(
        "best_validation_epoch="
        f"{validation_history.index(min(validation_history)) + 1}"
    )
    print(f"setup_seconds={setup_seconds:.4f}")
    print(f"training_seconds={training_seconds:.4f}")
    print(f"checkpoint_seconds={checkpoint_seconds:.4f}")
    print(f"total_seconds={total_seconds:.4f}")

    for prompt in ["hello", "مرحبا", "I ", "أنا "]:
        print(
            f"{prompt!r} -> "
            f"{model.generate(prompt, max_new_tokens=24)}"
        )


if __name__ == "__main__":
    main()
