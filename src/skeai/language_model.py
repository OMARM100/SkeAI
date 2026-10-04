"""A tiny character-level language model for SkeAI 0.1.

This is intentionally not a Transformer. It is a small fixed-context neural
language model used to prove the full learning/generation pipeline first.
"""

from __future__ import annotations

import random
from typing import List

from .layers import Dense, Tanh
from .model import Sequential
from .tensor import Tensor
from .tokenizer import CharacterTokenizer


class TinyCharacterLanguageModel:
    def __init__(
        self,
        tokenizer: CharacterTokenizer,
        context_length: int = 8,
        hidden_size: int = 32,
        seed: int = 123,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be greater than zero.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be greater than zero.")

        self.tokenizer = tokenizer
        self.context_length = context_length

        input_size = context_length * tokenizer.vocab_size

        self.network = Sequential(
            [
                Dense(input_size, hidden_size, seed=seed),
                Tanh(),
                Dense(hidden_size, tokenizer.vocab_size, seed=seed + 1),
            ]
        )

    def encode_context(self, token_ids: List[int]) -> Tensor:
        context = list(token_ids[-self.context_length:])

        if len(context) < self.context_length:
            context = [self.tokenizer.bos_id] * (
                self.context_length - len(context)
            ) + context

        values = [0.0] * (self.context_length * self.tokenizer.vocab_size)

        for position, token_id in enumerate(context):
            if token_id < 0 or token_id >= self.tokenizer.vocab_size:
                raise ValueError(f"Token ID out of range: {token_id}")

            values[
                position * self.tokenizer.vocab_size + token_id
            ] = 1.0

        return Tensor([values])

    def forward(self, token_ids: List[int]) -> Tensor:
        return self.network.forward(self.encode_context(token_ids))

    @staticmethod
    def _argmax(values: List[float]) -> int:
        return max(range(len(values)), key=values.__getitem__)

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        seed: int | None = None,
    ) -> str:
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")
        if temperature <= 0.0:
            raise ValueError("temperature must be greater than zero.")

        token_ids = self.tokenizer.encode(prompt)
        rng = random.Random(seed)

        for _ in range(max_new_tokens):
            logits = self.forward(token_ids).to_list()[0]

            if temperature == 1.0:
                next_id = self._argmax(logits)
            else:
                scaled = [value / temperature for value in logits]
                maximum = max(scaled)
                probabilities = [
                    pow(2.718281828, value - maximum)
                    for value in scaled
                ]
                total = sum(probabilities)
                probabilities = [value / total for value in probabilities]

                next_id = rng.choices(
                    range(len(probabilities)),
                    weights=probabilities,
                    k=1,
                )[0]

            token_ids.append(next_id)

            if next_id == self.tokenizer.eos_id:
                break

        return self.tokenizer.decode(token_ids)


__all__ = ["TinyCharacterLanguageModel"]
