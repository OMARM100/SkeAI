"""Benchmark SkeAI text generation across fixed decoding configurations.

This benchmark does not train the model or modify the checkpoint. It compares
generation settings on the same checkpoint, prompts, and random seed.

The numeric score measures generation diversity/repetition only. It is not a
semantic quality score, so the generated text is printed for manual inspection.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.skeai.language_model import TinyCharacterLanguageModel

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "tiny_character_model.json"
DEFAULT_SEED = 1234
DEFAULT_MAX_NEW_TOKENS = 32

PROMPTS = [
    "hello",
    "مرحبا",
    "أنا ",
    "كيف حالك؟ ",
    "الذكاء الاصطناعي ",
    "SkeAI ",
]


@dataclass(frozen=True)
class GenerationConfig:
    name: str
    temperature: float
    top_k: int | None
    repetition_penalty: float
    no_repeat_ngram_size: int


CONFIGS = [
    GenerationConfig(
        name="greedy",
        temperature=1.0,
        top_k=None,
        repetition_penalty=1.12,
        no_repeat_ngram_size=3,
    ),
    GenerationConfig(
        name="focused",
        temperature=0.65,
        top_k=5,
        repetition_penalty=1.12,
        no_repeat_ngram_size=3,
    ),
    GenerationConfig(
        name="balanced",
        temperature=0.75,
        top_k=8,
        repetition_penalty=1.12,
        no_repeat_ngram_size=3,
    ),
    GenerationConfig(
        name="baseline",
        temperature=0.85,
        top_k=8,
        repetition_penalty=1.12,
        no_repeat_ngram_size=3,
    ),
    GenerationConfig(
        name="creative",
        temperature=0.95,
        top_k=12,
        repetition_penalty=1.12,
        no_repeat_ngram_size=3,
    ),
    GenerationConfig(
        name="wide",
        temperature=1.10,
        top_k=16,
        repetition_penalty=1.12,
        no_repeat_ngram_size=3,
    ),
]


def _ngram_repetition_rate(text: str, n: int) -> float:
    if len(text) < n:
        return 0.0

    ngrams = [
        text[index:index + n]
        for index in range(len(text) - n + 1)
    ]
    if not ngrams:
        return 0.0

    return 1.0 - (len(set(ngrams)) / len(ngrams))


def _script_ratio(text: str) -> float:
    if not text:
        return 0.0

    arabic_count = 0
    meaningful_count = 0

    for character in text:
        category = ord(character)
        if character.isspace():
            continue

        meaningful_count += 1
        if (
            0x0600 <= category <= 0x06FF
            or 0x0750 <= category <= 0x077F
            or 0x08A0 <= category <= 0x08FF
            or 0xFB50 <= category <= 0xFDFF
            or 0xFE70 <= category <= 0xFEFF
        ):
            arabic_count += 1

    if meaningful_count == 0:
        return 0.0

    return arabic_count / meaningful_count


def _continuation(text: str, prompt: str) -> str:
    if not text.startswith(prompt):
        raise ValueError(
            f"Generated text does not preserve prompt prefix: {prompt!r}"
        )

    return text[len(prompt):]


def _health_score(
    unique_ratio: float,
    bigram_repeat_rate: float,
    trigram_repeat_rate: float,
) -> float:
    """Heuristic anti-repetition score from 0 to 100.

    This score measures diversity and repetition only. It does not measure
    grammaticality, factuality, or semantic coherence.
    """
    diversity_term = unique_ratio
    bigram_term = 1.0 - bigram_repeat_rate
    trigram_term = 1.0 - trigram_repeat_rate

    score = 100.0 * (
        0.50 * diversity_term
        + 0.25 * bigram_term
        + 0.25 * trigram_term
    )
    return max(0.0, min(100.0, score))


def _metrics(text: str) -> dict[str, float]:
    length = len(text)
    unique_ratio = (len(set(text)) / length) if length else 0.0
    bigram_repeat_rate = _ngram_repetition_rate(text, 2)
    trigram_repeat_rate = _ngram_repetition_rate(text, 3)
    return {
        "chars": float(length),
        "unique_ratio": unique_ratio,
        "bigram_repeat": bigram_repeat_rate,
        "trigram_repeat": trigram_repeat_rate,
        "arabic_ratio": _script_ratio(text),
        "health_score": _health_score(
            unique_ratio,
            bigram_repeat_rate,
            trigram_repeat_rate,
        ),
    }


def _average_metrics(metrics: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(metrics)
    if not rows:
        return {
            "chars": 0.0,
            "unique_ratio": 0.0,
            "bigram_repeat": 0.0,
            "trigram_repeat": 0.0,
            "arabic_ratio": 0.0,
            "health_score": 0.0,
        }

    keys = rows[0].keys()
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in keys
    }


def _format_ratio(value: float) -> str:
    return f"{value * 100:6.2f}%"


def _print_config_header(config: GenerationConfig) -> None:
    top_k = "all" if config.top_k is None else str(config.top_k)
    mode = "argmax" if config.temperature == 1.0 else "sampling"
    print(
        f"\n[{config.name}] "
        f"temperature={config.temperature:.2f} "
        f"top_k={top_k} "
        f"repetition_penalty={config.repetition_penalty:.2f} "
        f"no_repeat_ngram={config.no_repeat_ngram_size} "
        f"mode={mode}"
    )
    print("-" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SkeAI generation decoding settings."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint path (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=f"Maximum generated characters (default: {DEFAULT_MAX_NEW_TOKENS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducible sampling (default: {DEFAULT_SEED})",
    )
    args = parser.parse_args()

    if args.tokens <= 0:
        parser.error("--tokens must be greater than zero.")

    if not args.checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Run training first so models/tiny_character_model.json exists."
        )

    model = TinyCharacterLanguageModel.load_checkpoint(args.checkpoint)

    print("SkeAI Generation Benchmark")
    print(f"checkpoint={args.checkpoint}")
    print(f"vocabulary_size={model.tokenizer.vocab_size}")
    print(f"context_length={model.context_length}")
    print(f"hidden_size={model.hidden_size}")
    print(f"prompts={len(PROMPTS)}")
    print(f"max_new_tokens={args.tokens}")
    print(f"seed={args.seed}")
    print()
    print(
        "Note: health_score is only an anti-repetition/diversity heuristic; "
        "inspect the generated text for coherence."
    )

    summary_rows: list[tuple[str, dict[str, float]]] = []

    for config in CONFIGS:
        _print_config_header(config)
        config_metrics: list[dict[str, float]] = []

        for prompt in PROMPTS:
            result = model.generate(
                prompt,
                max_new_tokens=args.tokens,
                temperature=config.temperature,
                seed=args.seed,
                top_k=config.top_k,
                repetition_penalty=config.repetition_penalty,
                no_repeat_ngram_size=config.no_repeat_ngram_size,
            )
            continuation = _continuation(result, prompt)
            metrics = _metrics(continuation)
            config_metrics.append(metrics)

            print(f"prompt={prompt!r}")
            print(f"output={result}")
            print(
                "metrics="
                f"chars={int(metrics['chars'])} "
                f"unique={_format_ratio(metrics['unique_ratio'])} "
                f"bigram_repeat={_format_ratio(metrics['bigram_repeat'])} "
                f"trigram_repeat={_format_ratio(metrics['trigram_repeat'])} "
                f"arabic={_format_ratio(metrics['arabic_ratio'])} "
                f"health={metrics['health_score']:6.2f}"
            )
            print()

        average = _average_metrics(config_metrics)
        summary_rows.append((config.name, average))

        print(
            "AVERAGE "
            f"health={average['health_score']:6.2f} "
            f"unique={_format_ratio(average['unique_ratio'])} "
            f"bigram_repeat={_format_ratio(average['bigram_repeat'])} "
            f"trigram_repeat={_format_ratio(average['trigram_repeat'])} "
            f"arabic={_format_ratio(average['arabic_ratio'])}"
        )

    ranked = sorted(
        summary_rows,
        key=lambda item: item[1]["health_score"],
        reverse=True,
    )

    print("\n" + "=" * 88)
    print("SUMMARY — ranked by anti-repetition health score")
    print("=" * 88)
    print(
        f"{'config':<10} {'health':>8} {'unique':>9} "
        f"{'bi-repeat':>10} {'tri-repeat':>11} {'arabic':>9}"
    )

    for name, metrics in ranked:
        print(
            f"{name:<10} "
            f"{metrics['health_score']:8.2f} "
            f"{_format_ratio(metrics['unique_ratio']):>9} "
            f"{_format_ratio(metrics['bigram_repeat']):>10} "
            f"{_format_ratio(metrics['trigram_repeat']):>11} "
            f"{_format_ratio(metrics['arabic_ratio']):>9}"
        )


if __name__ == "__main__":
    main()
