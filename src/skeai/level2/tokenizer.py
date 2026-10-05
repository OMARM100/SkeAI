"""Hybrid token/character tokenizer for SkeAI Level 2.

Frequent text units become tokens while unseen units fall back to characters.
This keeps sequences shorter than pure character-level tokenization without
requiring an external tokenizer library.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Sequence


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]

_UNIT_RE = re.compile(r"\s+|[A-Za-z0-9_]+|[\u0600-\u06FF]+|[^\sA-Za-z0-9_\u0600-\u06FF]")


class HybridTokenizer:
    """Vocabulary of frequent units plus character fallback tokens."""

    def __init__(self, vocabulary: Sequence[str] | None = None) -> None:
        vocabulary = list(vocabulary or SPECIAL_TOKENS)
        if len(set(vocabulary)) != len(vocabulary):
            raise ValueError("Vocabulary contains duplicate tokens.")
        missing = [token for token in SPECIAL_TOKENS if token not in vocabulary]
        if missing:
            raise ValueError(f"Vocabulary is missing required special tokens: {missing}")

        self.id_to_token = vocabulary
        self.token_to_id = {token: i for i, token in enumerate(vocabulary)}

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

    @staticmethod
    def split_units(text: str) -> List[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        return _UNIT_RE.findall(text)

    def fit(
        self,
        texts: Iterable[str],
        *,
        max_units: int = 512,
        min_frequency: int = 2,
    ) -> None:
        if max_units <= 0:
            raise ValueError("max_units must be positive.")
        if min_frequency <= 0:
            raise ValueError("min_frequency must be positive.")

        unit_counts: Counter[str] = Counter()
        character_counts: Counter[str] = Counter()

        for text in texts:
            if not isinstance(text, str):
                raise TypeError("All training samples must be strings.")
            unit_counts.update(self.split_units(text))
            character_counts.update(text)

        # Prefer frequent multi-character units. The previous implementation
        # filled the vocabulary with single characters first, which left very
        # little room for words/phrases. That made Arabic dialogue sequences
        # unnecessarily long and caused the 48-token context to lose most of
        # the conversational structure.
        unit_candidates = [
            token
            for token, count in unit_counts.items()
            if count >= min_frequency and len(token) > 1
        ]
        unit_candidates.sort(key=lambda token: (-unit_counts[token], -len(token), token))

        character_candidates = sorted(
            character_counts,
            key=lambda token: (-character_counts[token], ord(token)),
        )

        seen = set(self.id_to_token)
        added = 0
        available_slots = max_units - len(self.id_to_token)
        if available_slots < 0:
            raise ValueError(
                "max_units must be at least the number of existing vocabulary tokens."
            )

        # Keep a bounded character reserve so unseen words still decompose
        # into characters, while preserving most of the vocabulary for useful
        # frequent words and punctuation patterns.
        character_reserve = min(192, available_slots)
        for token in character_candidates:
            if token in seen:
                continue
            if added >= character_reserve:
                break
            self.id_to_token.append(token)
            self.token_to_id[token] = len(self.id_to_token) - 1
            seen.add(token)
            added += 1

        for token in unit_candidates:
            if token in seen:
                continue
            if added >= available_slots:
                break
            self.id_to_token.append(token)
            self.token_to_id[token] = len(self.id_to_token) - 1
            seen.add(token)
            added += 1

        # Fill any remaining slots with lower-frequency single-character
        # units that were not part of the reserve.
        if added < available_slots:
            for token in sorted(character_counts, key=ord):
                if token in seen:
                    continue
                if added >= available_slots:
                    break
                self.id_to_token.append(token)
                self.token_to_id[token] = len(self.id_to_token) - 1
                seen.add(token)
                added += 1

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> List[int]:
        units = self.split_units(text)
        output: List[int] = []

        if add_bos:
            output.append(self.bos_id)

        for unit in units:
            token_id = self.token_to_id.get(unit)
            if token_id is not None:
                output.append(token_id)
                continue

            for character in unit:
                output.append(self.token_to_id.get(character, self.unk_id))

        if add_eos:
            output.append(self.eos_id)

        return output

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool = True) -> str:
        parts: List[str] = []
        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("Token IDs must be integers.")
            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(f"Token ID out of range: {token_id}")

            token = self.id_to_token[token_id]
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            parts.append(token)
        return "".join(parts)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "version": 1,
                    "type": "hybrid",
                    "vocabulary": self.id_to_token,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "HybridTokenizer":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("type") != "hybrid":
            raise ValueError("Unsupported tokenizer type.")
        vocabulary = payload.get("vocabulary")
        if not isinstance(vocabulary, list):
            raise ValueError("Tokenizer vocabulary must be a list.")
        return cls(vocabulary)


__all__ = ["HybridTokenizer", "PAD_TOKEN", "UNK_TOKEN", "BOS_TOKEN", "EOS_TOKEN"]
