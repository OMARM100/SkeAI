"""Production-style training entry point for SkeAI Level 2 Transformer."""

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
DEFAULT_TOKENIZER = ROOT / "models" / "level2_tokenizer.json"


def make_samples(
    token_ids: list[int],
    context_length: int,
    *,
    stride: int,
) -> list[tuple[list[int], list[int]]]:
    if context_length <= 0 or stride <= 0:
        raise ValueError("context_length and stride must be positive.")
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


def build_tokenizer(
    train_text: str,
    *,
    vocab_size: int,
    resume: bool,
    tokenizer_path: Path,
) -> HybridTokenizer:
    if resume:
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                f"Resume tokenizer not found: {tokenizer_path}"
            )
        return HybridTokenizer.load(tokenizer_path)

    tokenizer = HybridTokenizer()
    tokenizer.fit([train_text], max_units=vocab_size, min_frequency=2)
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SkeAI Level 2 Transformer."
    )
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
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.epochs <= 0 or args.max_train_steps <= 0:
        raise ValueError("epochs and max-train-steps must be positive.")
    if args.max_validation_steps <= 0:
        raise ValueError("max-validation-steps must be positive.")
    if args.patience < 0:
        raise ValueError("patience cannot be negative.")
    if args.min_delta < 0.0:
        raise ValueError("min-delta cannot be negative.")

    tokenizer_path = args.checkpoint.with_name("level2_tokenizer.json")

    train_text = (
        TRAIN_CORPUS.read_text(encoding="utf-8")
        + "\n"
        + DIALOGUE_CORPUS.read_text(encoding="utf-8")
    )
    validation_text = VALIDATION_CORPUS.read_text(encoding="utf-8")

    tokenizer = build_tokenizer(
        train_text,
        vocab_size=args.vocab,
        resume=args.resume,
        tokenizer_path=tokenizer_path,
    )

    train_tokens = tokenizer.encode(
        train_text,
        add_bos=True,
        add_eos=True,
    )
    validation_tokens = tokenizer.encode(
        validation_text,
        add_bos=True,
        add_eos=True,
    )

    if args.resume:
        if not args.checkpoint.exists():
            raise FileNotFoundError(
                f"Resume checkpoint not found: {args.checkpoint}"
            )
        model = TinyTransformerLM.load_checkpoint(args.checkpoint)
        if model.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                "Checkpoint and tokenizer vocabulary sizes do not match."
            )
    else:
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

    context_length = model.config.context_length
    train_samples = make_samples(
        train_tokens,
        context_length,
        stride=max(1, context_length // 2),
    )
    validation_samples = make_samples(
        validation_tokens,
        context_length,
        stride=context_length,
    )

    if not train_samples or not validation_samples:
        raise ValueError("Corpus is too short for the selected context.")

    trainer = Level2Trainer(
        model=model,
        optimizer=SGD(learning_rate=args.learning_rate),
    )

    rng = random.Random(args.seed)
    best_validation = float("inf")
    epochs_without_improvement = 0
    started = time.perf_counter()
    completed_steps = 0

    print("=== SkeAI Level 2 Training ===")
    print(f"resume={args.resume}")
    print(f"vocabulary_size={tokenizer.vocab_size}")
    print(f"train_samples={len(train_samples)}")
    print(f"validation_samples={len(validation_samples)}")
    print(f"parameter_count={model.parameter_count()}")
    print(f"context_length={context_length}")
    print(f"d_model={model.config.d_model}")
    print(f"heads={model.config.n_heads}")
    print(f"layers={model.config.n_layers}")
    print(f"feed_forward={model.config.feed_forward_size}")
    print(f"learning_rate={args.learning_rate}")
    print(f"patience={args.patience}")
    print("attention_backend=cpp")
    print("matrix_backend=cpp")

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
        for index in range(
            min(args.max_validation_steps, len(validation_samples))
        ):
            inputs, targets = validation_samples[index]
            validation_loss_total += trainer.evaluate(inputs, targets)
            validation_count += 1

        train_loss = train_loss_total / max(steps_this_epoch, 1)
        validation_loss = validation_loss_total / max(validation_count, 1)
        epoch_seconds = time.perf_counter() - epoch_start
        steps_per_second = steps_this_epoch / max(epoch_seconds, 1e-9)

        improved = validation_loss < (best_validation - args.min_delta)
        if improved:
            best_validation = validation_loss
            epochs_without_improvement = 0
            model.save_checkpoint(args.checkpoint)
            tokenizer.save(tokenizer_path)
            saved = True
        else:
            epochs_without_improvement += 1
            saved = False

        print(
            f"epoch={epoch} "
            f"train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"steps={steps_this_epoch} "
            f"epoch_seconds={epoch_seconds:.3f} "
            f"steps_per_second={steps_per_second:.3f} "
            f"saved_best={saved}"
        )

        if (
            args.patience > 0
            and epochs_without_improvement >= args.patience
        ):
            print("early_stopping=true")
            break

    total_seconds = time.perf_counter() - started
    print(f"completed_steps={completed_steps}")
    print(f"best_validation_loss={best_validation:.6f}")
    print(f"checkpoint={args.checkpoint}")
    print(f"tokenizer={tokenizer_path}")
    print(f"total_seconds={total_seconds:.3f}")


if __name__ == "__main__":
    main()
