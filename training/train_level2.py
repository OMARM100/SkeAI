"""Production-style training entry point for SkeAI Level 2 Transformer."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

from src.skeai.level2.tokenizer import HybridTokenizer
from src.skeai.level2.transformer import TinyTransformerLM, TransformerConfig
from src.skeai.level2.trainer import Level2Trainer
from src.skeai.optimizer import SGD


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CORPUS = ROOT / "data" / "samples" / "tiny_corpus.txt"
DIALOGUE_JSON = ROOT / "data" / "samples" / "tiny_dialogue.json"
VALIDATION_CORPUS = ROOT / "data" / "samples" / "tiny_validation.txt"
DEFAULT_CHECKPOINT = ROOT / "models" / "level2_transformer.json"
DEFAULT_TOKENIZER = ROOT / "models" / "level2_tokenizer.json"
DEFAULT_METADATA = ROOT / "models" / "level2_training_metadata.json"

USER_LABEL = "المستخدم:"
ASSISTANT_LABEL = "SkeAI:"


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


def load_dialogue_pairs(path: Path) -> list[list[tuple[str, str]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Dialogue corpus must contain a JSON list.")

    conversations: list[list[tuple[str, str]]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        turns = item.get("turns")
        if isinstance(turns, list):
            conversation: list[tuple[str, str]] = []
            pending_user: str | None = None
            for turn in turns:
                if not isinstance(turn, dict):
                    continue
                role = str(turn.get("role", "")).lower()
                text = turn.get("text")
                if not isinstance(text, str):
                    continue
                text = " ".join(text.strip().split())
                if not text:
                    continue
                if role == "user":
                    pending_user = text
                elif role == "assistant" and pending_user is not None:
                    conversation.append((pending_user, text))
                    pending_user = None
            if conversation:
                conversations.append(conversation)
            continue

        user = item.get("input")
        response = item.get("response")
        if isinstance(user, str) and isinstance(response, str):
            user = " ".join(user.strip().split())
            response = " ".join(response.strip().split())
            if user and response:
                conversations.append([(user, response)])

    if not conversations:
        raise ValueError("Dialogue corpus does not contain valid conversations.")
    return conversations


TRAINING_SELF_CONTEXT = (
    "حالة SkeAI: أنا SkeAI | نوعي: ذكاء اصطناعي | أنا لست إنسانًا | "
    "مرحلة التطور: بداية التعلم"
)

def format_dialogue(
    conversation: list[tuple[str, str]],
    *,
    include_self_context: bool = True,
) -> str:
    # Keep the neural training prompt aligned with the runtime chat prompt.
    parts = [TRAINING_SELF_CONTEXT] if include_self_context else []
    parts.extend(
        f"{USER_LABEL} {user}\n{ASSISTANT_LABEL} {response}"
        for user, response in conversation
    )
    return "\n".join(parts)


def build_dialogue_samples(
    conversations: list[list[tuple[str, str]]],
    tokenizer: HybridTokenizer,
    context_length: int,
) -> list[tuple[list[int], list[int]]]:
    samples: list[tuple[list[int], list[int]]] = []
    for conversation in conversations:
        text = format_dialogue(conversation, include_self_context=True)
        tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
        samples.extend(
            make_samples(
                tokens,
                context_length,
                stride=max(1, context_length // 3),
            )
        )
    return samples


def _fixed_target_window(
    token_ids: list[int],
    target_index: int,
    context_length: int,
) -> tuple[list[int], list[int]] | None:
    # A response-focused example keeps the same fixed context size required by
    # the fused native trainer, while choosing the final target around an
    # assistant response token rather than an arbitrary corpus offset.
    if context_length <= 0:
        raise ValueError("context_length must be positive.")
    if target_index < context_length or target_index >= len(token_ids):
        return None

    start = target_index - context_length
    inputs = token_ids[start:target_index]
    targets = token_ids[start + 1:target_index + 1]
    if len(inputs) != context_length or len(targets) != context_length:
        return None
    return inputs, targets


def build_response_focused_samples(
    conversations: list[list[tuple[str, str]]],
    tokenizer: HybridTokenizer,
    context_length: int,
) -> list[tuple[list[int], list[int]]]:
    # The normal dialogue windows train the whole conversation stream. These
    # additional windows deliberately end on assistant response tokens, so
    # useful answer tokens receive more learning signal without changing the
    # native trainer interface or introducing a second Python training path.
    samples: list[tuple[list[int], list[int]]] = []

    for conversation in conversations:
        dialogue_lines = [TRAINING_SELF_CONTEXT]

        for user, response in conversation:
            prompt_text = "\\n".join(
                [
                    *dialogue_lines,
                    f"{USER_LABEL} {user}",
                    f"{ASSISTANT_LABEL} ",
                ]
            )
            prefix_tokens = tokenizer.encode(
                prompt_text,
                add_bos=False,
                add_eos=False,
            )
            response_tokens = tokenizer.encode(
                response,
                add_bos=False,
                add_eos=False,
            )
            if not response_tokens:
                continue

            sequence = [
                tokenizer.bos_id,
                *prefix_tokens,
                *response_tokens,
                tokenizer.eos_id,
            ]
            response_start = 1 + len(prefix_tokens)
            response_last = response_start + len(response_tokens) - 1

            candidate_indices = {
                response_start,
                response_start + min(2, len(response_tokens) - 1),
                response_start + min(5, len(response_tokens) - 1),
                response_last,
            }

            for target_index in sorted(candidate_indices):
                sample = _fixed_target_window(
                    sequence,
                    target_index,
                    context_length,
                )
                if sample is not None:
                    samples.append(sample)

            dialogue_lines.extend(
                [
                    f"{USER_LABEL} {user}",
                    f"{ASSISTANT_LABEL} {response}",
                ]
            )

    return samples


def scheduled_learning_rate(
    base_learning_rate: float,
    *,
    epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    final_scale: float,
) -> float:
    if base_learning_rate <= 0.0:
        raise ValueError("base_learning_rate must be positive.")
    if epoch <= 0 or total_epochs <= 0:
        raise ValueError("epoch and total_epochs must be positive.")
    if warmup_epochs < 0:
        raise ValueError("warmup_epochs cannot be negative.")
    if not 0.0 < final_scale <= 1.0:
        raise ValueError("final_scale must be in the interval (0, 1].")

    if warmup_epochs > 0 and epoch <= warmup_epochs:
        return base_learning_rate * (epoch / warmup_epochs)

    decay_span = max(1, total_epochs - warmup_epochs)
    progress = min(
        1.0,
        max(0.0, (epoch - warmup_epochs) / decay_span),
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    scale = final_scale + (1.0 - final_scale) * cosine
    return base_learning_rate * scale


def build_tokenizer(
    train_text: str,
    dialogue_texts: list[str],
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
    tokenizer.fit(
        [train_text, *dialogue_texts],
        max_units=vocab_size,
        min_frequency=2,
    )
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train SkeAI Level 2 Transformer."
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--dialogue-repeat", type=int, default=12)
    parser.add_argument("--response-focus-repeat", type=int, default=8)
    parser.add_argument("--context", type=int, default=48)
    parser.add_argument("--vocab", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--ff", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--final-learning-rate-scale", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-delta", type=float, default=0.0005)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.dialogue_repeat <= 0:
        raise ValueError("dialogue-repeat must be positive.")
    if args.response_focus_repeat <= 0:
        raise ValueError("response-focus-repeat must be positive.")
    if args.warmup_epochs < 0:
        raise ValueError("warmup-epochs cannot be negative.")
    if not 0.0 < args.final_learning_rate_scale <= 1.0:
        raise ValueError("final-learning-rate-scale must be in the interval (0, 1].")
    if args.patience < 0:
        raise ValueError("patience cannot be negative.")
    if args.min_delta < 0.0:
        raise ValueError("min-delta cannot be negative.")

    tokenizer_path = args.checkpoint.with_name("level2_tokenizer.json")
    metadata_path = args.checkpoint.with_name("level2_training_metadata.json")

    train_text = TRAIN_CORPUS.read_text(encoding="utf-8")
    validation_text = VALIDATION_CORPUS.read_text(encoding="utf-8")
    all_dialogue_conversations = load_dialogue_pairs(DIALOGUE_JSON)

    # Hold out complete conversations. This prevents the validation metric from
    # rewarding memorization of the exact dialogue pairs used for training.
    split_rng = random.Random(args.seed)
    shuffled_dialogues = list(all_dialogue_conversations)
    split_rng.shuffle(shuffled_dialogues)
    validation_dialogue_count = max(1, len(shuffled_dialogues) // 5)
    validation_dialogues = shuffled_dialogues[:validation_dialogue_count]
    dialogue_conversations = shuffled_dialogues[validation_dialogue_count:]
    if not dialogue_conversations:
        raise ValueError("Dialogue corpus must contain at least two conversations.")

    dialogue_texts = [
        format_dialogue(conversation, include_self_context=True)
        for conversation in dialogue_conversations
    ]

    tokenizer = build_tokenizer(
        train_text,
        dialogue_texts,
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

    language_samples = make_samples(
        train_tokens,
        context_length,
        stride=max(1, context_length // 2),
    )
    dialogue_samples = build_dialogue_samples(
        dialogue_conversations,
        tokenizer,
        context_length,
    )
    response_focused_samples = build_response_focused_samples(
        dialogue_conversations,
        tokenizer,
        context_length,
    )
    validation_samples = make_samples(
        validation_tokens,
        context_length,
        stride=context_length,
    )
    dialogue_validation_samples = build_dialogue_samples(
        validation_dialogues,
        tokenizer,
        context_length,
    )

    if (
        not language_samples
        or not dialogue_samples
        or not validation_samples
        or not dialogue_validation_samples
    ):
        raise ValueError("One of the training or validation corpora is too short.")

    # Dialogue examples are deliberately oversampled. This is the first
    # conversation milestone, so the model must spend meaningful training
    # capacity learning turn boundaries, identity, and short answers.
    language_training_steps = len(language_samples)
    dialogue_training_steps = len(dialogue_samples) * args.dialogue_repeat
    response_focused_training_steps = (
        len(response_focused_samples) * args.response_focus_repeat
    )
    training_pool = (
        language_samples
        + (dialogue_samples * args.dialogue_repeat)
        + (response_focused_samples * args.response_focus_repeat)
    )

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
    print(f"language_samples={len(language_samples)}")
    print(f"dialogue_conversations={len(dialogue_conversations)}")
    print(f"dialogue_validation_conversations={len(validation_dialogues)}")
    print(f"dialogue_pairs={sum(len(c) for c in dialogue_conversations)}")
    print(f"dialogue_samples={len(dialogue_samples)}")
    print(f"response_focused_samples={len(response_focused_samples)}")
    print(f"dialogue_validation_samples={len(dialogue_validation_samples)}")
    print(f"dialogue_repeat={args.dialogue_repeat}")
    print(f"response_focus_repeat={args.response_focus_repeat}")
    print(f"language_training_steps={language_training_steps}")
    print(f"dialogue_training_steps={dialogue_training_steps}")
    print(f"response_focused_training_steps={response_focused_training_steps}")
    print(f"training_pool={len(training_pool)}")
    print(f"validation_samples={len(validation_samples)}")
    print(f"parameter_count={model.parameter_count()}")
    print(f"context_length={context_length}")
    print(f"d_model={model.config.d_model}")
    print(f"heads={model.config.n_heads}")
    print(f"layers={model.config.n_layers}")
    print(f"feed_forward={model.config.feed_forward_size}")
    print(f"learning_rate={args.learning_rate}")
    print(f"warmup_epochs={args.warmup_epochs}")
    print(f"final_learning_rate_scale={args.final_learning_rate_scale}")
    print(f"patience={args.patience}")
    print("training_backend=cpp_batch_fused")
    print("attention_backend=cpp")
    print("matrix_backend=cpp")

    for epoch in range(1, args.epochs + 1):
        epoch_learning_rate = scheduled_learning_rate(
            args.learning_rate,
            epoch=epoch,
            total_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            final_scale=args.final_learning_rate_scale,
        )
        trainer.optimizer.learning_rate = epoch_learning_rate

        # Always train on the complete training pool.
        # The corpus is the source of truth; no subset of samples is selected.
        selected = list(training_pool)
        rng.shuffle(selected)

        if not selected:
            raise RuntimeError("No training samples available for epoch.")

        steps_this_epoch = len(selected)
        epoch_start = time.perf_counter()
        train_loss = trainer.train_batch(
            [sample[0] for sample in selected],
            [sample[1] for sample in selected],
        )
        completed_steps += steps_this_epoch

        language_validation_total = 0.0
        language_validation_count = 0
        # Evaluate the complete validation corpus as well.
        for inputs, targets in validation_samples:
            language_validation_total += trainer.evaluate(inputs, targets)
            language_validation_count += 1

        dialogue_validation_total = 0.0
        dialogue_validation_count = 0
        for inputs, targets in dialogue_validation_samples:
            dialogue_validation_total += trainer.evaluate(inputs, targets)
            dialogue_validation_count += 1

        language_validation_loss = (
            language_validation_total / max(language_validation_count, 1)
        )
        dialogue_validation_loss = (
            dialogue_validation_total / max(dialogue_validation_count, 1)
        )
        # Equal weighting keeps generic language quality from hiding a
        # regression in actual conversation behavior.
        validation_loss = (
            language_validation_loss + dialogue_validation_loss
        ) / 2.0
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
            f"epoch_learning_rate={epoch_learning_rate:.8f} "
            f"train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"language_validation_loss={language_validation_loss:.6f} "
            f"dialogue_validation_loss={dialogue_validation_loss:.6f} "
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
    metadata: dict[str, Any] = {
        "version": 2,
        "model": "level2_transformer",
        "training_backend": "cpp_batch_fused",
        "language_samples": len(language_samples),
        "dialogue_conversations": len(dialogue_conversations),
        "dialogue_validation_conversations": len(validation_dialogues),
        "dialogue_pairs": sum(len(c) for c in dialogue_conversations),
        "dialogue_samples": len(dialogue_samples),
        "response_focused_samples": len(response_focused_samples),
        "dialogue_validation_samples": len(dialogue_validation_samples),
        "language_validation_loss": language_validation_loss,
        "dialogue_validation_loss": dialogue_validation_loss,
        "dialogue_repeat": args.dialogue_repeat,
        "response_focus_repeat": args.response_focus_repeat,
        "language_training_steps": language_training_steps,
        "dialogue_training_steps": dialogue_training_steps,
        "response_focused_training_steps": response_focused_training_steps,
        "warmup_epochs": args.warmup_epochs,
        "final_learning_rate_scale": args.final_learning_rate_scale,
        "training_pool": len(training_pool),
        "completed_steps": completed_steps,
        "best_validation_loss": best_validation,
        "parameter_count": model.parameter_count(),
        "context_length": model.config.context_length,
        "vocabulary_size": tokenizer.vocab_size,
        "total_seconds": total_seconds,
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"completed_steps={completed_steps}")
    print(f"best_validation_loss={best_validation:.6f}")
    print(f"checkpoint={args.checkpoint}")
    print(f"tokenizer={tokenizer_path}")
    print(f"metadata={metadata_path}")
    print(f"total_seconds={total_seconds:.3f}")


if __name__ == "__main__":
    main()
