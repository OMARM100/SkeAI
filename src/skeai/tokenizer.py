"""A small character-level tokenizer implemented from scratch.

SkeAI 0.1 intentionally uses Unicode characters as tokens. This keeps the
first tokenizer understandable, language-independent, and usable on small
datasets without requiring an external tokenizer library.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]


class CharacterTokenizer:
    """Build a vocabulary from text and encode/decode Unicode characters."""

    def __init__(self, vocabulary: Sequence[str] | None = None) -> None:
        if vocabulary is None:
            vocabulary = SPECIAL_TOKENS

        vocabulary = list(vocabulary)

        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Vocabulary contains duplicate tokens.")

        missing = [token for token in SPECIAL_TOKENS if token not in vocabulary]
        if missing:
            raise ValueError(
                f"Vocabulary is missing required special tokens: {missing}"
            )

        self.id_to_token = vocabulary
        self.token_to_id = {
            token: index for index, token in enumerate(self.id_to_token)
        }

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    def fit(self, texts: Iterable[str]) -> None:
        """Extend the vocabulary using characters found in texts."""
        discovered = set()

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("All training samples must be strings.")
            discovered.update(text)

        new_tokens = sorted(
            discovered.difference(self.token_to_id),
            key=ord,
        )

        for token in new_tokens:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")

        tokens: List[int] = []

        if add_bos:
            tokens.append(self.bos_id)

        tokens.extend(
            self.token_to_id.get(character, self.unk_id)
            for character in text
        )

        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:
        characters: List[str] = []

        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("Token IDs must be integers.")

            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID out of range: {token_id}")

            token = self.id_to_token[token_id]

            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue

            characters.append(token)

        return "".join(characters)

    def save(self, path: str | Path) -> None:
        """Save the vocabulary as UTF-8 JSON."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": 1,
            "type": "character",
            "vocabulary": self.id_to_token,
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "CharacterTokenizer":
        input_path = Path(path)

        with input_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if payload.get("type") != "character":
            raise ValueError("Unsupported tokenizer type.")

        vocabulary = payload.get("vocabulary")
        if not isinstance(vocabulary, list):
            raise ValueError("Tokenizer vocabulary must be a list.")

        return cls(vocabulary)


__all__ = [
    "CharacterTokenizer",
    "PAD_TOKEN",
    "UNK_TOKEN",
    "BOS_TOKEN",
    "EOS_TOKEN",
]
