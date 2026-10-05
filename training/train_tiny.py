"""Train the tiny SkeAI character language model."""

from __future__ import annotations

import argparse
import json
from math import inf
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
COVERAGE_PATH = ROOT / "data" / "samples" / "tiny_character_coverage.txt"
CHECKPOINT_PATH = ROOT / "models" / "tiny_character_model.json"
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 8
DEFAULT_MIN_DELTA = 0.001
BATCH_SIZE = 16
BATCH_LOG_EVERY = 50

CONTEXT_LENGTH = 32
HIDDEN_SIZE = 32
LEARNING_RATE = 0.05
WEIGHT_DECAY = 0.001


def load_model(resume: bool, text: str) -> TinyCharacterLanguageModel:
    if resume:
        if not CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"No checkpoint found at {CHECKPOINT_PATH}. Run without --resume first."
            )
        return TinyCharacterLanguageModel.load_checkpoint(CHECKPOINT_PATH)

    coverage_text = COVERAGE_PATH.read_text(encoding="utf-8")
    tokenizer = CharacterTokenizer()
    tokenizer.fit([text, coverage_text])
    return TinyCharacterLanguageModel(
        tokenizer=tokenizer,
        context_length=CONTEXT_LENGTH,
        hidden_size=HIDDEN_SIZE,
    )


def main() -> None:
    total_start = perf_counter()
    parser = argparse.ArgumentParser(description="Train SkeAI tiny language model.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    args = parser.parse_args()

    if args.epochs <= 0:
        parser.error("--epochs must be greater than zero.")
    if args.patience < 0:
        parser.error("--patience cannot be negative.")
    if args.min_delta < 0.0:
        parser.error("--min-delta cannot be negative.")

    setup_start = perf_counter()
    text = CORPUS_PATH.read_text(encoding="utf-8")
    dialogue_path = ROOT / "data" / "samples" / "tiny_dialogue.txt"
    dialogue_text = dialogue_path.read_text(encoding="utf-8")
    validation_text = VALIDATION_PATH.read_text(encoding="utf-8")
    training_text = text + "\n" + dialogue_text
    model = load_model(args.resume, training_text)

    dialogue_json = ROOT / "data" / "samples" / "tiny_dialogue.json"
    dialogue_entries = json.loads(dialogue_json.read_text(encoding="utf-8"))
    response_memory = {
        str(item["input"]): str(item["response"])
        for item in dialogue_entries
        if isinstance(item, dict) and "input" in item and "response" in item
    }
    model.set_response_memory(response_memory)

    unknown = sorted(
        {c for c in validation_text if c not in model.tokenizer.token_to_id},
        key=ord,
    )
    if unknown:
        raise ValueError(
            f"Validation corpus contains characters outside the training vocabulary: {unknown!r}"
        )

    dataset = CharacterLanguageDataset(
        text=training_text,
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
        optimizer=SGD(
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        ),
        loss=CrossEntropyLoss(),
        enable_timing=True,
    )

    batches = dataset.all_indexed_batches(batch_size=BATCH_SIZE)
    validation_batches = validation_dataset.all_indexed_batches(batch_size=BATCH_SIZE)
    setup_seconds = perf_counter() - setup_start

    print(
        f"training_plan=epochs:{args.epochs} "
        f"batches_per_epoch:{len(batches)} "
        f"total_steps:{len(batches) * args.epochs} "
        f"batch_size:{BATCH_SIZE} "
        f"training_examples:{len(dataset)} "
        f"validation_examples:{len(validation_dataset)} "
        f"context_length:{model.context_length} "
        f"hidden_size:{model.hidden_size} "
        f"shuffle:True patience:{args.patience}"
    )

    initial_validation_loss = trainer.evaluate_batches(validation_batches)
    validation_history: list[float] = []
    best_validation_loss = inf
    best_epoch = 0
    bad_epochs = 0
    best_train_loss = inf

    def report(epoch: int, batch: int, loss: float) -> None:
        if batch == 1 or batch % BATCH_LOG_EVERY == 0 or batch == len(batches):
            print(
                f"epoch={epoch:03d} batch={batch:04d}/{len(batches):04d} "
                f"loss={loss:.6f}"
            )

    def report_epoch(
        epoch: int,
        loss: float,
        elapsed_seconds: float,
        examples_per_second: float,
    ) -> bool:
        nonlocal best_validation_loss, best_epoch, bad_epochs, best_train_loss

        validation_loss = trainer.evaluate_batches(validation_batches)
        validation_history.append(validation_loss)

        if validation_loss < best_validation_loss - args.min_delta:
            best_validation_loss = validation_loss
            best_epoch = epoch
            best_train_loss = loss
            bad_epochs = 0
            model.save_checkpoint(CHECKPOINT_PATH)
        else:
            bad_epochs += 1

        gap = validation_loss - loss
        print(
            f"epoch={epoch:03d} time={elapsed_seconds:.4f}s "
            f"train_loss={loss:.6f} validation_loss={validation_loss:.6f} "
            f"gap={gap:+.6f} best_validation={best_validation_loss:.6f} "
            f"best_epoch={best_epoch:03d} bad_epochs={bad_epochs}/{args.patience} "
            f"avg_step={trainer.last_epoch_timing['average_step_ms']:.3f}ms "
            f"examples/s={examples_per_second:.2f}"
        )
        return args.patience > 0 and bad_epochs >= args.patience

    training_start = perf_counter()
    history = trainer.train_batches(
        batches,
        epochs=args.epochs,
        callback=report,
        epoch_callback=report_epoch,
        shuffle=True,
        seed=1234,
    )
    training_seconds = perf_counter() - training_start

    best_model = TinyCharacterLanguageModel.load_checkpoint(CHECKPOINT_PATH)
    best_evaluator = Trainer(
        model=best_model.network,
        optimizer=SGD(learning_rate=LEARNING_RATE),
        loss=CrossEntropyLoss(),
    )
    final_validation_loss = best_evaluator.evaluate_batches(validation_batches)
    total_seconds = perf_counter() - total_start

    print(f"checkpoint={CHECKPOINT_PATH}")
    print(f"vocabulary_size={best_model.tokenizer.vocab_size}")
    print(f"training_examples={len(dataset)}")
    print(f"validation_examples={len(validation_dataset)}")
    print(f"parameter_count={best_model.network.parameter_count()}")
    print(f"epochs_completed={len(history)}")
    print(f"initial_loss={history[0]:.6f}")
    print(f"last_loss={history[-1]:.6f}")
    print(f"best_train_loss={best_train_loss:.6f}")
    print(f"initial_validation_loss={initial_validation_loss:.6f}")
    print(f"best_validation_loss={best_validation_loss:.6f}")
    print(f"best_validation_epoch={best_epoch}")
    print(f"final_validation_loss={final_validation_loss:.6f}")
    print(f"setup_seconds={setup_seconds:.4f}")
    print(f"training_seconds={training_seconds:.4f}")
    print(f"total_seconds={total_seconds:.4f}")

    for prompt in ["hello", "مرحبا", "أنا ", "كيف حالك؟ ", "ما اسمك؟"]:
        result = best_model.respond(
            prompt,
            max_new_tokens=32,
            temperature=0.65,
            seed=1234,
            top_k=8,
            repetition_penalty=1.12,
            no_repeat_ngram_size=3,
        )
        print(f"{prompt!r} -> {result}")


if __name__ == "__main__":
    main()
