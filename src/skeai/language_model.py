"""A tiny character-level language model for SkeAI."""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from datetime import datetime
from difflib import SequenceMatcher
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
        self.response_memory: dict[str, str] = {}

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

    def encode_context_indices(self, token_ids: List[int]) -> List[int]:
        context = list(token_ids[-self.context_length:])

        if len(context) < self.context_length:
            context = [self.tokenizer.bos_id] * (
                self.context_length - len(context)
            ) + context

        indices: List[int] = []
        vocab_size = self.tokenizer.vocab_size

        for position, token_id in enumerate(context):
            if token_id < 0 or token_id >= vocab_size:
                raise ValueError(f"Token ID out of range: {token_id}")
            indices.append(position * vocab_size + token_id)

        return indices

    def forward(self, token_ids: List[int]) -> Tensor:
        return self.network.forward_indexed(
            self.encode_context_indices(token_ids),
            batch_size=1,
        )
    @staticmethod
    def normalize_prompt(text: str) -> str:
        """Normalize short prompts for deterministic memory lookup."""
        normalized = unicodedata.normalize("NFKC", text).strip().casefold()
        normalized = re.sub(r"[\s\u200f\u200e]+", " ", normalized)
        normalized = re.sub(r"[!؟?.,،؛;:]+", " ", normalized)
        return normalized.strip()

    def set_response_memory(self, entries: dict[str, str]) -> None:
        if not isinstance(entries, dict):
            raise TypeError("response memory must be a dictionary")
        self.response_memory = {
            self.normalize_prompt(str(prompt)): str(response).strip()
            for prompt, response in entries.items()
            if str(prompt).strip() and str(response).strip()
        }

    def memorized_response(self, prompt: str) -> str | None:
        key = self.normalize_prompt(prompt)
        if not key:
            return None

        exact = self.response_memory.get(key)
        if exact is not None:
            return exact

        # Small fuzzy lookup for natural variations such as:
        # "ما اسمك" vs "ما هو اسمك"
        # "كيف حالك" vs "كيف حالك ؟"
        if len(key) < 4 or not self.response_memory:
            return None

        key_tokens = set(key.split())
        best_response: str | None = None
        best_score = 0.0

        for candidate, response in self.response_memory.items():
            candidate_tokens = set(candidate.split())
            if not candidate_tokens:
                continue

            character_score = SequenceMatcher(
                None,
                key,
                candidate,
            ).ratio()

            union = key_tokens | candidate_tokens
            overlap = key_tokens & candidate_tokens
            token_score = len(overlap) / len(union) if union else 0.0

            score = 0.72 * character_score + 0.28 * token_score
            if score > best_score:
                best_score = score
                best_response = response

        if best_score >= 0.82:
            return best_response
        return None

    def _dynamic_response(self, prompt: str) -> str | None:
        """Handle deterministic facts that should never become stale."""
        key = self.normalize_prompt(prompt)

        time_queries = {
            "كم الساعة",
            "كم الساعه",
            "الساعة كام",
            "الساعه كام",
            "الوقت كام",
            "الوقت الآن",
            "الوقت الان",
        }
        if key in time_queries:
            return f"الساعة الآن {datetime.now().strftime('%H:%M')}."

        date_queries = {
            "ما التاريخ اليوم",
            "ما تاريخ اليوم",
            "تاريخ اليوم",
            "التاريخ اليوم",
        }
        if key in date_queries:
            return f"تاريخ اليوم هو {datetime.now().strftime('%Y-%m-%d')}."

        return None

    def respond(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.65,
        seed: int | None = 1234,
    ) -> str:
        """Return a grounded response when available, otherwise generate."""
        dynamic = self._dynamic_response(prompt)
        if dynamic is not None:
            return dynamic

        memorized = self.memorized_response(prompt)
        if memorized is not None:
            return memorized
        generated = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            seed=seed,
            top_k=6,
            repetition_penalty=1.15,
            no_repeat_ngram_size=3,
        )
        if generated.startswith(prompt):
            return generated[len(prompt):].strip()
        return generated.strip()
    @staticmethod
    def _argmax(values: List[float], candidates: List[int] | None = None) -> int:
        if candidates is None:
            return max(range(len(values)), key=values.__getitem__)
        return max(candidates, key=values.__getitem__)

    @staticmethod
    def _would_repeat_ngram(
        token_ids: List[int],
        candidate_id: int,
        ngram_size: int,
    ) -> bool:
        if ngram_size <= 1:
            return False

        proposed = token_ids + [candidate_id]
        if len(proposed) < ngram_size:
            return False

        target = tuple(proposed[-ngram_size:])
        for start in range(len(proposed) - ngram_size):
            if tuple(proposed[start:start + ngram_size]) == target:
                return True

        return False

    def _adjust_logits(
        self,
        logits: List[float],
        token_ids: List[int],
        repetition_penalty: float,
    ) -> List[float]:
        adjusted = list(logits)

        if repetition_penalty == 1.0:
            return adjusted

        recent_tokens = set(token_ids[-self.context_length:])
        special_ids = {
            self.tokenizer.pad_id,
            self.tokenizer.bos_id,
            self.tokenizer.eos_id,
            self.tokenizer.unk_id,
        }

        for token_id in recent_tokens:
            if token_id in special_ids:
                continue
            if adjusted[token_id] >= 0.0:
                adjusted[token_id] /= repetition_penalty
            else:
                adjusted[token_id] *= repetition_penalty

        return adjusted

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        seed: int | None = None,
        top_k: int | None = 8,
        repetition_penalty: float = 1.1,
        no_repeat_ngram_size: int = 3,
    ) -> str:
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")
        if temperature <= 0.0:
            raise ValueError("temperature must be greater than zero.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be greater than zero or None.")
        if repetition_penalty < 1.0:
            raise ValueError("repetition_penalty must be at least one.")
        if no_repeat_ngram_size < 0:
            raise ValueError("no_repeat_ngram_size cannot be negative.")

        token_ids = self.tokenizer.encode(prompt)
        rng = random.Random(seed)
        generated_ids: List[int] = []

        for _ in range(max_new_tokens):
            logits = self.forward(token_ids).to_list()[0]
            adjusted = self._adjust_logits(
                logits,
                token_ids,
                repetition_penalty,
            )

            blocked = {
                token_id
                for token_id in range(self.tokenizer.vocab_size)
                if self._would_repeat_ngram(
                    token_ids,
                    token_id,
                    no_repeat_ngram_size,
                )
            }

            blocked.update({
                self.tokenizer.pad_id,
                self.tokenizer.bos_id,
                self.tokenizer.unk_id,
            })

            allowed = [
                token_id
                for token_id in range(self.tokenizer.vocab_size)
                if token_id not in blocked
            ]
            if not allowed:
                allowed = list(range(self.tokenizer.vocab_size))

            if temperature == 1.0:
                next_id = self._argmax(adjusted, allowed)
            else:
                if top_k is None:
                    candidates = allowed
                else:
                    candidates = sorted(
                        allowed,
                        key=adjusted.__getitem__,
                        reverse=True,
                    )[: min(top_k, len(allowed))]

                scaled = [adjusted[token_id] / temperature for token_id in candidates]
                maximum = max(scaled)
                probabilities = [
                    math.exp(value - maximum)
                    for value in scaled
                ]
                total = sum(probabilities)
                probabilities = [value / total for value in probabilities]

                next_id = rng.choices(
                    candidates,
                    weights=probabilities,
                    k=1,
                )[0]

            token_ids.append(next_id)
            generated_ids.append(next_id)

            if next_id == self.tokenizer.eos_id:
                break

        return prompt + self.tokenizer.decode(generated_ids)

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
                "response_memory": self.response_memory,
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

        memory = model_data.get("response_memory", {})
        if isinstance(memory, dict):
            model.set_response_memory(memory)

        state = model_data.get("state")
        if not isinstance(state, dict):
            raise ValueError("Checkpoint state is invalid.")

        model.network.load_state_dict(state)
        return model


__all__ = ["TinyCharacterLanguageModel"]
