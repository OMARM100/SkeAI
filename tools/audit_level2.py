"""Validate Level 2 token/sequence/context alignment before training."""

from __future__ import annotations

from pathlib import Path

from src.skeai.level2.tokenizer import HybridTokenizer
from src.skeai.level2.transformer import TransformerConfig
from training.train_level2 import (
    build_dialogue_samples,
    build_response_focused_samples,
    load_dialogue_pairs,
    make_samples,
)


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CORPUS = ROOT / "data" / "samples" / "tiny_corpus.txt"
DIALOGUE_JSON = ROOT / "data" / "samples" / "tiny_dialogue.json"


def _assert_next_token_alignment(
    samples: list[tuple[list[int], list[int]]],
    label: str,
) -> None:
    if not samples:
        raise AssertionError(f"{label}: no samples were generated.")

    for index, (inputs, targets) in enumerate(samples):
        if not inputs or len(inputs) != len(targets):
            raise AssertionError(
                f"{label}: invalid lengths at sample {index}."
            )
        if targets[:-1] != inputs[1:]:
            raise AssertionError(
                f"{label}: input/target shift is broken at sample {index}."
            )


def main() -> None:
    train_text = TRAIN_CORPUS.read_text(encoding="utf-8")
    conversations = load_dialogue_pairs(DIALOGUE_JSON)
    dialogue_texts = []

    for conversation in conversations:
        dialogue_texts.append(
            "\n".join(
                [
                    (
                        "حالة SkeAI: أنا SkeAI | نوعي: ذكاء اصطناعي | "
                        "أنا لست إنسانًا | مرحلة التطور: بداية التعلم"
                    ),
                    *[
                        f"{role}: {text}"
                        for user, response in conversation
                        for role, text in (
                            ("المستخدم", user),
                            ("SkeAI", response),
                        )
                    ],
                ]
            )
        )

    tokenizer = HybridTokenizer()
    tokenizer.fit(
        [train_text, *dialogue_texts],
        max_units=2048,
        min_frequency=2,
    )

    config = TransformerConfig()
    if (
        config.context_length != 128
        or config.d_model != 128
        or config.n_heads != 4
        or config.feed_forward_size != 512
        or config.n_layers != 4
        or config.max_vocab_size != 2048
    ):
        raise AssertionError("Level 2 default configuration is not medium.")

    train_tokens = tokenizer.encode(train_text, add_bos=True, add_eos=True)
    language_samples = make_samples(
        train_tokens,
        config.context_length,
        stride=max(1, config.context_length // 2),
    )
    dialogue_samples = build_dialogue_samples(
        conversations,
        tokenizer,
        config.context_length,
    )
    response_samples = build_response_focused_samples(
        conversations,
        tokenizer,
        config.context_length,
    )

    _assert_next_token_alignment(language_samples, "language")
    _assert_next_token_alignment(dialogue_samples, "dialogue")

    if not response_samples:
        raise AssertionError("response-focused: no samples were generated.")

    for index, (inputs, targets, weights) in enumerate(response_samples):
        if not inputs or len(inputs) != len(targets) != len(weights):
            raise AssertionError(
                f"response-focused: invalid lengths at sample {index}."
            )
        if targets[:-1] != inputs[1:]:
            raise AssertionError(
                f"response-focused: broken next-token shift at sample {index}."
            )
        if any(weight not in (0.25, 1.0) for weight in weights):
            raise AssertionError(
                f"response-focused: unexpected target weight at sample {index}."
            )
        if not any(weight == 1.0 for weight in weights):
            raise AssertionError(
                f"response-focused: no response target weight at sample {index}."
            )

    medium_parameters = (
        tokenizer.vocab_size * config.d_model
        + config.context_length * config.d_model
        + config.d_model * tokenizer.vocab_size
        + config.n_layers
        * (
            4 * config.d_model * config.d_model
            + 2 * config.d_model * config.feed_forward_size
        )
    )

    print("=== SkeAI Level 2 Pipeline Audit ===")
    print(f"vocabulary_size={tokenizer.vocab_size}")
    print(f"language_tokens={len(train_tokens)}")
    print(f"language_samples={len(language_samples)}")
    print(f"dialogue_conversations={len(conversations)}")
    print(f"dialogue_samples={len(dialogue_samples)}")
    print(f"response_focused_samples={len(response_samples)}")
    print(f"context_length={config.context_length}")
    print(f"d_model={config.d_model}")
    print(f"heads={config.n_heads}")
    print(f"layers={config.n_layers}")
    print(f"feed_forward={config.feed_forward_size}")
    print(f"parameter_count={medium_parameters}")
    print("next_token_alignment=OK")
    print("response_weight_alignment=OK")
    print("medium_configuration=OK")
    print("PIPELINE_AUDIT=OK")


if __name__ == "__main__":
    main()
