"""Benchmark SkeAI next-character learning and generalization.

This benchmark does not train the model or modify the checkpoint. It evaluates
the saved model on the training and validation corpora using next-character
prediction metrics.

The most important comparison is the train/validation gap:
a much stronger training score than validation indicates memorization or
overfitting; closer scores are a better sign of generalization.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from src.skeai.dataset import CharacterLanguageDataset
from src.skeai.language_model import TinyCharacterLanguageModel
from src.skeai.tokenizer import SPECIAL_TOKENS

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "tiny_character_model.json"
DEFAULT_TRAINING = ROOT / "data" / "samples" / "tiny_corpus.txt"
DEFAULT_VALIDATION = ROOT / "data" / "samples" / "tiny_validation.txt"
DEFAULT_BATCH_SIZE = 128

_WORD_PATTERN = re.compile(r"[A-Za-z\u0600-\u06FF]+")


@dataclass
class EvaluationMetrics:
    examples: int = 0
    top1_correct: int = 0
    top3_correct: int = 0
    top5_correct: int = 0
    nll_sum: float = 0.0

    arabic_examples: int = 0
    arabic_top1_correct: int = 0
    arabic_top3_correct: int = 0
    arabic_top5_correct: int = 0

    latin_examples: int = 0
    latin_top1_correct: int = 0
    latin_top3_correct: int = 0
    latin_top5_correct: int = 0

    unseen_word_examples: int = 0
    unseen_word_top1_correct: int = 0
    unseen_word_top3_correct: int = 0
    unseen_word_top5_correct: int = 0

    def add(
        self,
        target_token: str,
        rank: int,
        nll: float,
        *,
        unseen_word: bool = False,
    ) -> None:
        if target_token in SPECIAL_TOKENS:
            return

        self.examples += 1
        self.nll_sum += nll

        if rank <= 1:
            self.top1_correct += 1
        if rank <= 3:
            self.top3_correct += 1
        if rank <= 5:
            self.top5_correct += 1

        if _is_arabic_character(target_token):
            self.arabic_examples += 1
            if rank <= 1:
                self.arabic_top1_correct += 1
            if rank <= 3:
                self.arabic_top3_correct += 1
            if rank <= 5:
                self.arabic_top5_correct += 1
        elif _is_latin_character(target_token):
            self.latin_examples += 1
            if rank <= 1:
                self.latin_top1_correct += 1
            if rank <= 3:
                self.latin_top3_correct += 1
            if rank <= 5:
                self.latin_top5_correct += 1

        if unseen_word:
            self.unseen_word_examples += 1
            if rank <= 1:
                self.unseen_word_top1_correct += 1
            if rank <= 3:
                self.unseen_word_top3_correct += 1
            if rank <= 5:
                self.unseen_word_top5_correct += 1

    @property
    def top1_accuracy(self) -> float:
        return _ratio(self.top1_correct, self.examples)

    @property
    def top3_accuracy(self) -> float:
        return _ratio(self.top3_correct, self.examples)

    @property
    def top5_accuracy(self) -> float:
        return _ratio(self.top5_correct, self.examples)

    @property
    def mean_nll(self) -> float:
        return self.nll_sum / self.examples if self.examples else math.inf

    @property
    def perplexity(self) -> float:
        if not self.examples:
            return math.inf
        return math.exp(min(self.mean_nll, 50.0))

    @property
    def arabic_top1_accuracy(self) -> float:
        return _ratio(self.arabic_top1_correct, self.arabic_examples)

    @property
    def arabic_top3_accuracy(self) -> float:
        return _ratio(self.arabic_top3_correct, self.arabic_examples)

    @property
    def arabic_top5_accuracy(self) -> float:
        return _ratio(self.arabic_top5_correct, self.arabic_examples)

    @property
    def latin_top1_accuracy(self) -> float:
        return _ratio(self.latin_top1_correct, self.latin_examples)

    @property
    def unseen_word_top1_accuracy(self) -> float:
        return _ratio(
            self.unseen_word_top1_correct,
            self.unseen_word_examples,
        )

    @property
    def unseen_word_top3_accuracy(self) -> float:
        return _ratio(
            self.unseen_word_top3_correct,
            self.unseen_word_examples,
        )

    @property
    def unseen_word_top5_accuracy(self) -> float:
        return _ratio(
            self.unseen_word_top5_correct,
            self.unseen_word_examples,
        )


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _is_arabic_character(character: str) -> bool:
    if not character:
        return False

    codepoint = ord(character[0])
    return (
        0x0600 <= codepoint <= 0x06FF
        or 0x0750 <= codepoint <= 0x077F
        or 0x08A0 <= codepoint <= 0x08FF
        or 0xFB50 <= codepoint <= 0xFDFF
        or 0xFE70 <= codepoint <= 0xFEFF
    )


def _is_latin_character(character: str) -> bool:
    return bool(character) and character[0].isascii() and character[0].isalpha()


def _logsumexp(values: Sequence[float]) -> float:
    maximum = max(values)
    return maximum + math.log(
        sum(math.exp(value - maximum) for value in values)
    )


def _target_rank(logits: Sequence[float], target_id: int) -> int:
    target_value = logits[target_id]
    return 1 + sum(value > target_value for value in logits)


def _target_nll(logits: Sequence[float], target_id: int) -> float:
    return _logsumexp(logits) - logits[target_id]


def _unseen_word_positions(
    text: str,
    training_words: set[str],
) -> set[int]:
    positions: set[int] = set()

    for match in _WORD_PATTERN.finditer(text):
        word = match.group(0).lower()
        if word in training_words:
            continue
        positions.update(range(match.start(), match.end()))

    return positions


def _training_words(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in _WORD_PATTERN.finditer(text)
    }


def evaluate_text(
    model: TinyCharacterLanguageModel,
    text: str,
    *,
    batch_size: int,
    unseen_word_positions: set[int] | None = None,
) -> EvaluationMetrics:
    dataset = CharacterLanguageDataset(
        text=text,
        tokenizer=model.tokenizer,
        context_length=model.context_length,
    )
    metrics = EvaluationMetrics()

    sample_start = 0
    for inputs, targets in dataset.batches(batch_size=batch_size):
        logits_rows = model.network.forward(inputs).to_list()

        for row_index, (logits, target_id) in enumerate(
            zip(logits_rows, targets)
        ):
            target_token = model.tokenizer.id_to_token[target_id]
            rank = _target_rank(logits, target_id)
            nll = _target_nll(logits, target_id)

            text_position = (
                sample_start
                + row_index
                + model.context_length
                - 1
            )
            unseen_word = (
                unseen_word_positions is not None
                and target_token not in SPECIAL_TOKENS
                and text_position in unseen_word_positions
            )

            metrics.add(
                target_token,
                rank,
                nll,
                unseen_word=unseen_word,
            )

        sample_start += len(targets)

    return metrics


def _format_percent(value: float) -> str:
    return f"{value * 100:6.2f}%"


def _print_metrics(
    label: str,
    metrics: EvaluationMetrics,
    vocab_size: int,
) -> None:
    print(f"\n[{label}]")
    print(f"examples={metrics.examples}")
    print(
        f"top1_accuracy={_format_percent(metrics.top1_accuracy)} "
        f"top3_accuracy={_format_percent(metrics.top3_accuracy)} "
        f"top5_accuracy={_format_percent(metrics.top5_accuracy)}"
    )
    print(f"mean_nll={metrics.mean_nll:.6f}")
    print(f"perplexity={metrics.perplexity:.4f}")
    print(
        f"arabic_examples={metrics.arabic_examples} "
        f"arabic_top1={_format_percent(metrics.arabic_top1_accuracy)} "
        f"arabic_top3={_format_percent(metrics.arabic_top3_accuracy)} "
        f"arabic_top5={_format_percent(metrics.arabic_top5_accuracy)}"
    )
    print(
        f"latin_examples={metrics.latin_examples} "
        f"latin_top1={_format_percent(metrics.latin_top1_accuracy)}"
    )
    if metrics.unseen_word_examples:
        print(
            f"unseen_word_examples={metrics.unseen_word_examples} "
            f"unseen_word_top1={_format_percent(metrics.unseen_word_top1_accuracy)} "
            f"unseen_word_top3={_format_percent(metrics.unseen_word_top3_accuracy)} "
            f"unseen_word_top5={_format_percent(metrics.unseen_word_top5_accuracy)}"
        )
    random_top1 = 1.0 / vocab_size
    random_top3 = min(3, vocab_size) / vocab_size
    random_top5 = min(5, vocab_size) / vocab_size
    print(
        "random_reference="
        f"top1={_format_percent(random_top1)} "
        f"top3={_format_percent(random_top3)} "
        f"top5={_format_percent(random_top5)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark SkeAI next-character learning."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )
    parser.add_argument(
        "--training",
        type=Path,
        default=DEFAULT_TRAINING,
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=DEFAULT_VALIDATION,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero.")

    for path in (args.checkpoint, args.training, args.validation):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    model = TinyCharacterLanguageModel.load_checkpoint(args.checkpoint)
    training_text = args.training.read_text(encoding="utf-8")
    validation_text = args.validation.read_text(encoding="utf-8")

    unknown = sorted(
        {character for character in validation_text
         if character not in model.tokenizer.token_to_id},
        key=ord,
    )
    if unknown:
        raise ValueError(
            "Validation contains characters outside the checkpoint vocabulary: "
            f"{unknown!r}"
        )

    unseen_positions = _unseen_word_positions(
        validation_text,
        _training_words(training_text),
    )

    print("SkeAI Learning Benchmark")
    print(f"checkpoint={args.checkpoint}")
    print(f"training={args.training}")
    print(f"validation={args.validation}")
    print(f"vocabulary_size={model.tokenizer.vocab_size}")
    print(f"context_length={model.context_length}")
    print(f"hidden_size={model.hidden_size}")
    print(f"batch_size={args.batch_size}")
    print(
        "This benchmark evaluates next-character prediction. "
        "It does not prove semantic understanding."
    )

    training_metrics = evaluate_text(
        model,
        training_text,
        batch_size=args.batch_size,
    )
    validation_metrics = evaluate_text(
        model,
        validation_text,
        batch_size=args.batch_size,
        unseen_word_positions=unseen_positions,
    )

    _print_metrics(
        "TRAINING",
        training_metrics,
        model.tokenizer.vocab_size,
    )
    _print_metrics(
        "VALIDATION",
        validation_metrics,
        model.tokenizer.vocab_size,
    )

    print("\n[GENERALIZATION GAP]")
    print(
        "top1_gap="
        f"{_format_percent(training_metrics.top1_accuracy - validation_metrics.top1_accuracy)}"
    )
    print(
        "top3_gap="
        f"{_format_percent(training_metrics.top3_accuracy - validation_metrics.top3_accuracy)}"
    )
    print(
        "top5_gap="
        f"{_format_percent(training_metrics.top5_accuracy - validation_metrics.top5_accuracy)}"
    )
    print(
        "nll_gap="
        f"{training_metrics.mean_nll - validation_metrics.mean_nll:+.6f}"
    )
    print(
        "perplexity_ratio="
        f"{validation_metrics.perplexity / training_metrics.perplexity:.3f}x"
    )

    print("\n[INTERPRETATION GUIDE]")
    print(
        "1) High training accuracy + much lower validation accuracy = "
        "strong memorization/overfitting signal."
    )
    print(
        "2) Similar training and validation accuracy = better generalization signal."
    )
    print(
        "3) Unseen-word metrics show whether known characters combine correctly "
        "inside words the model did not see as complete words."
    )
    print(
        "4) Next-character accuracy is a learning diagnostic, not a proof of "
        "semantic understanding."
    )


if __name__ == "__main__":
    main()
