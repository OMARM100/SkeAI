"""A tiny character-level language model for SkeAI 0.1."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, List

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
        self.hidden_size = hidden_size

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

    def save_checkpoint(self, path: str | Path) -> None:
        """Save tokenizer and model state as a portable JSON checkpoint."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "format": "skeai-checkpoint",
            "version": 1,
            "model": {
                "type": "tiny-character-language-model",
                "context_length": self.context_length,
                "hidden_size": self.hidden_size,
                "tokenizer": {
                    "type": "character",
                    "vocabulary": self.tokenizer.id_to_token,
                },
                "state": self.network.state_dict(),
            },
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

    @classmethod
    def load_checkpoint(
        cls,
        path: str | Path,
    ) -> "TinyCharacterLanguageModel":
        input_path = Path(path)

        with input_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if payload.get("format") != "skeai-checkpoint":
            raise ValueError("Unsupported SkeAI checkpoint format.")
        if payload.get("version") != 1:
            raise ValueError("Unsupported SkeAI checkpoint version.")

        model_data = payload.get("model", {})
        if model_data.get("type") != "tiny-character-language-model":
            raise ValueError("Unsupported SkeAI model type.")

        tokenizer_data = model_data.get("tokenizer", {})
        vocabulary = tokenizer_data.get("vocabulary")
        if not isinstance(vocabulary, list):
            raise ValueError("Checkpoint vocabulary is invalid.")

        tokenizer = CharacterTokenizer(vocabulary)

        model = cls(
            tokenizer=tokenizer,
            context_length=int(model_data["context_length"]),
            hidden_size=int(model_data["hidden_size"]),
            seed=123,
        )

        state = model_data.get("state")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state is invalid.")

        model.network.load_state_dict(state)
        return model


__all__ = ["TinyCharacterLanguageModel"]
