"""Conversation, persistent memory, and chat orchestration for SkeAI Level 2."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .generation import generate_text
from .tokenizer import HybridTokenizer
from .transformer import TinyTransformerLM


USER_LABEL = "المستخدم:"
ASSISTANT_LABEL = "SkeAI:"
MAX_STORED_TURNS = 12


@dataclass
class ConversationMemory:
    """Small persistent episodic memory kept outside model weights."""

    turns: list[dict[str, str]] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append({"user": user, "assistant": assistant})
        if len(self.turns) > MAX_STORED_TURNS:
            del self.turns[:-MAX_STORED_TURNS]

    def remember_from_user(self, message: str) -> None:
        patterns = (
            (r"^(?:اسمي|انا اسمي|أنا اسمي)\s+(.+)$", "user_name"),
            (r"^(?:my name is)\s+(.+)$", "user_name"),
        )
        normalized = " ".join(message.strip().split())
        for pattern, key in patterns:
            match = re.match(pattern, normalized, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" .!؟?,")
                if value:
                    self.facts[key] = value
                return

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "type": "skeai_conversation_memory",
            "facts": self.facts,
            "turns": self.turns,
        }
        output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ConversationMemory":
        source = Path(path)
        if not source.exists():
            return cls()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("type") != "skeai_conversation_memory":
            raise ValueError("Unsupported conversation memory file.")
        facts = payload.get("facts", {})
        turns = payload.get("turns", [])
        if not isinstance(facts, dict) or not isinstance(turns, list):
            raise ValueError("Invalid conversation memory format.")
        clean_turns = [
            {"user": str(turn["user"]), "assistant": str(turn["assistant"])}
            for turn in turns
            if isinstance(turn, dict)
            and "user" in turn
            and "assistant" in turn
        ]
        return cls(
            facts={str(key): str(value) for key, value in facts.items()},
            turns=clean_turns[-MAX_STORED_TURNS:],
        )


class SkeAIConversation:
    """Stateful chat facade over the trained Level 2 model."""

    def __init__(
        self,
        model: TinyTransformerLM,
        tokenizer: HybridTokenizer,
        *,
        memory_path: str | Path | None = None,
        max_history_turns: int = 3,
    ) -> None:
        if max_history_turns <= 0:
            raise ValueError("max_history_turns must be positive.")
        self.model = model
        self.tokenizer = tokenizer
        self.memory_path = Path(memory_path) if memory_path is not None else None
        self.max_history_turns = max_history_turns
        self.memory = (
            ConversationMemory.load(self.memory_path)
            if self.memory_path is not None
            else ConversationMemory()
        )

    def _identity_context(self) -> str:
        name = self.memory.facts.get("user_name")
        if name:
            return (
                "هوية النظام: أنت SkeAI، ذكاء اصطناعي صغير قيد التطوير "
                "من مشروع مستقل مبني من الصفر. أنت لست إنسانًا. "
                f"اسم المستخدم الذي تتذكره هو {name}.\n"
            )
        return (
            "هوية النظام: أنت SkeAI، ذكاء اصطناعي صغير قيد التطوير "
            "من مشروع مستقل مبني من الصفر. أنت لست إنسانًا.\n"
        )

    def _build_prompt(self, message: str) -> str:
        turns = self.memory.turns[-self.max_history_turns:]
        parts = [self._identity_context()]
        for turn in turns:
            parts.append(f"{USER_LABEL} {turn['user']}\n{ASSISTANT_LABEL} {turn['assistant']}\n")
        parts.append(f"{USER_LABEL} {message}\n{ASSISTANT_LABEL}")
        return "".join(parts)

    def _trim_to_context(self, prompt: str) -> str:
        token_ids = self.tokenizer.encode(prompt)
        limit = self.model.config.context_length
        if len(token_ids) <= limit:
            return prompt

        # Keep the current user turn and the most recent history. Character
        # slicing is intentionally conservative; the tokenizer can re-encode it.
        marker = f"{USER_LABEL} "
        chunks = prompt.split(marker)
        current = chunks[-1]
        current_tokens = self.tokenizer.encode(current)
        if len(current_tokens) >= limit:
            return self.tokenizer.decode(
                current_tokens[-limit:],
                skip_special_tokens=False,
            )

        kept = current
        for previous in reversed(chunks[:-1]):
            candidate = marker + previous + kept
            if len(self.tokenizer.encode(candidate)) > limit:
                break
            kept = candidate
        return kept

    def chat(
        self,
        message: str,
        *,
        max_new_tokens: int = 48,
        temperature: float = 0.35,
        top_k: int = 8,
        seed: int | None = 1234,
    ) -> str:
        if not isinstance(message, str):
            raise TypeError("message must be a string.")
        message = message.strip()
        if not message:
            raise ValueError("message cannot be empty.")

        self.memory.remember_from_user(message)
        prompt = self._trim_to_context(self._build_prompt(message))
        response = generate_text(
            self.model,
            self.tokenizer,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
        ).strip()

        # A language model can occasionally reproduce the next role label.
        for label in (USER_LABEL, ASSISTANT_LABEL):
            if label in response:
                response = response.split(label, 1)[0].rstrip()
        if not response:
            response = "أحتاج إلى تدريب إضافي لأتمكن من الرد بشكل أفضل."

        self.memory.add_turn(message, response)
        if self.memory_path is not None:
            self.memory.save(self.memory_path)
        return response

    def reset(self, *, clear_persistent_memory: bool = False) -> None:
        self.memory.turns.clear()
        if clear_persistent_memory:
            self.memory.facts.clear()
        if self.memory_path is not None:
            self.memory.save(self.memory_path)


__all__ = ["ConversationMemory", "SkeAIConversation"]
