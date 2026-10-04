"""Tiny character-level language dataset for SkeAI 0.1."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from .tensor import Tensor
from .tokenizer import CharacterTokenizer


class CharacterLanguageDataset:
    """Create fixed-length next-character training examples."""

    def __init__(
        self,
        text: str,
        tokenizer: CharacterTokenizer,
        context_length: int = 8,
    ) -> None:
        if not isinstance(text, str) or not text:
            raise ValueError("text must be a non-empty string.")
        if context_length <= 0:
            raise ValueError("context_length must be greater than zero.")

        self.tokenizer = tokenizer
        self.context_length = context_length
        self.tokens = tokenizer.encode(text, add_bos=True, add_eos=True)

        if len(self.tokens) <= context_length:
            raise ValueError("Text is too short for the selected context length.")

    def __len__(self) -> int:
        return len(self.tokens) - self.context_length

    def _one_hot_context(self, context: Sequence[int]) -> List[float]:
        vocab_size = self.tokenizer.vocab_size
        values = [0.0] * (len(context) * vocab_size)

        for position, token_id in enumerate(context):
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(f"Token ID out of range: {token_id}")
            values[position * vocab_size + token_id] = 1.0

        return values

    def batches(self, batch_size: int) -> Tuple[Tensor, List[int]]:
        """Return the full dataset as contiguous mini-batches one at a time."""
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")

        for start in range(0, len(self), batch_size):
            inputs: List[List[float]] = []
            targets: List[int] = []

            stop = min(start + batch_size, len(self))

            for index in range(start, stop):
                context = self.tokens[index:index + self.context_length]
                target = self.tokens[index + self.context_length]

                inputs.append(self._one_hot_context(context))
                targets.append(target)

            yield Tensor(inputs), targets

    def all_batches(self, batch_size: int) -> List[Tuple[Tensor, List[int]]]:
        return list(self.batches(batch_size))


__all__ = ["CharacterLanguageDataset"]
