"""Experimental Level 2 Transformer trainer for SkeAI."""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from src.skeai.level2.tokenizer import HybridTokenizer
from src.skeai.level2.transformer import TinyTransformerLM, TransformerConfig
from src.skeai.level2.trainer import Level2Trainer
from src.skeai.optimizer import SGD


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CORPUS = ROOT / "data" / "samples" / "tiny_corpus.txt"
DIALOGUE_CORPUS = ROOT / "data" / "samples" / "tiny_dialogue.txt"
VALIDATION_CORPUS = ROOT / "data" / "samples" / "tiny_validation.txt"
DEFAULT_CHECKPOINT = ROOT / "models" / "level2_transformer.json"


def make_samples(
    token_ids: list[int],
    context_length: int,
    *,
    stride: int,
) -> list[tuple[list[int], list[int]]]:
    if len(token_ids) <= context_length:
        return []

    samples: list[tuple[list[int], list[int]]] = []
    for start in range(0, len(token_ids) - context_length, stride):
        inputs = token_ids[start:start + context_length]
        targets = token_ids[start + 1:start + context_length + 1]
        if len(targets) != context_length:
            break
        samples.append((inputs, targets))
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SkeAI Level 2 Transformer.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-train-steps", type=int, default=500)
    parser.add_argument("--max-validation-steps", type=int, default=50)
    parser.add_argument("--context", type=int, default=64)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if args.epochs <= 0 or args.max_train_steps <= 0:
        raise ValueError("epochs and max-train-steps must be positive.")

    train_text = (
        TRAIN_CORPUS.read_text(encoding="utf-8")
        + "\n"
        + DIALOGUE_CORPUS.read_text(encoding="utf-8")
    )
    validation_text = VALIDATION_CORPUS.read_text(encoding="utf-8")

    tokenizer = HybridTokenizer()
    tokenizer.fit([train_text], max_units=args.vocab, min_frequency=2)

    train_tokens = tokenizer.encode(train_text, add_bos=True, add_eos=True)
    validation_tokens = tokenizer.encode(
        validation_text,
        add_bos=True,
        add_eos=True,
    )

    train_samples = make_samples(
        train_tokens,
        args.context,
        stride=max(1, args.context // 2),
    )
    validation_samples = make_samples(
        validation_tokens,
        args.context,
        stride=args.context,
    )

    if not train_samples or not validation_samples:
        raise ValueError("Corpus is too short for the selected context.")

    config = TransformerConfig(
        context_length=args.context,
        d_model=args.d_model,
        n_heads=args.heads,
        feed_forward_size=args.ff,
        n_layers=args.layers,
        max_vocab_size=args.vocab,
        seed=args.seed,
    )
    model = TinyTransformerLM(tokenizer.vocab_size, config)
    trainer = Level2Trainer(
        model=model,
        optimizer=SGD(learning_rate=args.learning_rate),
    )

    rng = random.Random(args.seed)
    best_validation = float("inf")
    started = time.perf_counter()
    completed_steps = 0

    print("=== SkeAI Level 2 Training ===")
    print(f"vocabulary_size={tokenizer.vocab_size}")
    print(f"train_samples={len(train_samples)}")
    print(f"validation_samples={len(validation_samples)}")
    print(f"parameter_count={model.parameter_count()}")
    print(f"context_length={args.context}")

    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train_samples)))
        rng.shuffle(order)

        train_loss_total = 0.0
        steps_this_epoch = 0
        epoch_start = time.perf_counter()

        for index in order:
            inputs, targets = train_samples[index]
            train_loss_total += trainer.train_step(inputs, targets)
            steps_this_epoch += 1
            completed_steps += 1

            if steps_this_epoch >= args.max_train_steps:
                break

        validation_loss_total = 0.0
        validation_count = 0
        for index in range(min(args.max_validation_steps, len(validation_samples))):
            inputs, targets = validation_samples[index]
            validation_loss_total += trainer.evaluate(inputs, targets)
            validation_count += 1

        train_loss = train_loss_total / max(steps_this_epoch, 1)
        validation_loss = validation_loss_total / max(validation_count, 1)
        elapsed = time.perf_counter() - epoch_start

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"steps={steps_this_epoch} "
            f"epoch_seconds={elapsed:.3f}"
        )

        if validation_loss < best_validation:
            best_validation = validation_loss
            model.save_checkpoint(args.checkpoint)
            tokenizer.save(args.checkpoint.with_name("level2_tokenizer.json"))
            print(f"saved_best={args.checkpoint}")

    total_seconds = time.perf_counter() - started
    print(f"completed_steps={completed_steps}")
    print(f"best_validation_loss={best_validation:.6f}")
    print(f"total_seconds={total_seconds:.3f}")


if __name__ == "__main__":
    main()
