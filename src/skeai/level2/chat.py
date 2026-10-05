"""Conversation state, developmental memory, self-model, and learning for SkeAI Level 2."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from .generation import generate_text
from .tokenizer import HybridTokenizer
from .transformer import TinyTransformerLM


USER_LABEL = "المستخدم:"
ASSISTANT_LABEL = "SkeAI:"
SYSTEM_LABEL = "حالة SkeAI:"
MAX_STORED_TURNS = 20
MAX_EXPERIENCES = 50


@dataclass
class SkeAIState:
    """Persistent non-neural state: identity, user model, and learned experiences."""

    self_model: dict[str, str] = field(default_factory=lambda: {
        "name": "SkeAI",
        "entity_type": "ذكاء اصطناعي",
        "human": "false",
        "development_stage": "بداية التعلم",
        "origin": "مشروع ذكاء اصطناعي مستقل مبني من الصفر",
    })
    user_facts: dict[str, str] = field(default_factory=dict)
    conversation_topic: str = ""
    current_goal: str = ""
    experiences: list[dict[str, str]] = field(default_factory=list)

    def observe(self, event: str, lesson: str = "") -> None:
        event = " ".join(str(event).strip().split())
        lesson = " ".join(str(lesson).strip().split())
        if not event:
            return
        item = {"event": event}
        if lesson:
            item["lesson"] = lesson
        self.experiences.append(item)
        if len(self.experiences) > MAX_EXPERIENCES:
            del self.experiences[:-MAX_EXPERIENCES]

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "type": "skeai_state",
            "self_model": self.self_model,
            "user_facts": self.user_facts,
            "conversation_topic": self.conversation_topic,
            "current_goal": self.current_goal,
            "experiences": self.experiences,
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SkeAIState":
        source = Path(path)
        if not source.exists():
            return cls()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("type") not in {"skeai_state", "skeai_conversation_memory"}:
            raise ValueError("Unsupported SkeAI state file.")
        if payload.get("type") == "skeai_conversation_memory":
            state = cls()
            state.user_facts = {
                str(k): str(v) for k, v in payload.get("facts", {}).items()
            }
            return state
        return cls(
            self_model={str(k): str(v) for k, v in payload.get("self_model", {}).items()},
            user_facts={str(k): str(v) for k, v in payload.get("user_facts", {}).items()},
            conversation_topic=str(payload.get("conversation_topic", "")),
            current_goal=str(payload.get("current_goal", "")),
            experiences=[
                {str(k): str(v) for k, v in item.items()}
                for item in payload.get("experiences", [])
                if isinstance(item, dict)
            ][-MAX_EXPERIENCES:],
        )


@dataclass
class ConversationMemory:
    """Persistent episodic conversation plus structured state."""

    turns: list[dict[str, str]] = field(default_factory=list)
    facts: dict[str, str] = field(default_factory=dict)
    state: SkeAIState = field(default_factory=SkeAIState)

    @property
    def user_facts(self) -> dict[str, str]:
        return self.state.user_facts

    def add_turn(self, user: str, assistant: str) -> None:
        self.turns.append({"user": user, "assistant": assistant})
        if len(self.turns) > MAX_STORED_TURNS:
            del self.turns[:-MAX_STORED_TURNS]
        self.state.observe(f"محادثة: {user}", f"الرد: {assistant}")

    def remember_from_user(self, message: str) -> None:
        normalized = " ".join(message.strip().split())
        patterns = (
            (r"^(?:اسمي|انا اسمي|أنا اسمي)\s+(.+)$", "user_name"),
            (r"^(?:my name is)\s+(.+)$", "user_name"),
        )
        for pattern, key in patterns:
            match = re.match(pattern, normalized, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" .!؟?,")
                if value:
                    self.facts[key] = value
                    self.state.user_facts[key] = value
                break

        if re.search(r"(?:مجرد|فقط)\s+(?:اسم مستخدم|يوزر|username)", normalized, re.IGNORECASE):
            self.state.user_facts["username_is_real_name"] = "false"

        if re.search(r"(?:اسمي الحقيقي|ده اسمي الحقيقي|هذا اسمي الحقيقي)", normalized, re.IGNORECASE):
            self.state.user_facts["username_is_real_name"] = "true"

        if normalized.startswith(("أنا أحب ", "انا احب ", "I like ")):
            value = re.sub(r"^(?:أنا أحب|انا احب|I like)\s+", "", normalized, flags=re.IGNORECASE)
            if value:
                self.state.user_facts["likes"] = value

        if normalized.startswith(("أكره ", "انا اكره ", "أنا أكره ", "I hate ")):
            value = re.sub(r"^(?:أكره|انا اكره|أنا أكره|I hate)\s+", "", normalized, flags=re.IGNORECASE)
            if value:
                self.state.user_facts["dislikes"] = value

        self.state.observe(f"المستخدم قال: {normalized}")

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "type": "skeai_state",
            "facts": self.facts,
            "turns": self.turns,
            "state": {
                "self_model": self.state.self_model,
                "user_facts": self.state.user_facts,
                "conversation_topic": self.state.conversation_topic,
                "current_goal": self.state.current_goal,
                "experiences": self.state.experiences,
            },
        }
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConversationMemory":
        source = Path(path)
        if not source.exists():
            return cls()
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("type") == "skeai_conversation_memory":
            state = SkeAIState()
            facts = payload.get("facts", {})
            turns = payload.get("turns", [])
            state.user_facts = {str(k): str(v) for k, v in facts.items()} if isinstance(facts, dict) else {}
            clean_turns = [
                {"user": str(t["user"]), "assistant": str(t["assistant"])}
                for t in turns
                if isinstance(t, dict) and "user" in t and "assistant" in t
            ]
            return cls(facts=state.user_facts.copy(), turns=clean_turns[-MAX_STORED_TURNS:], state=state)

        if payload.get("type") != "skeai_state":
            raise ValueError("Unsupported conversation memory file.")

        state_payload = payload.get("state", {})
        state = SkeAIState(
            self_model={str(k): str(v) for k, v in state_payload.get("self_model", {}).items()} or SkeAIState().self_model,
            user_facts={str(k): str(v) for k, v in state_payload.get("user_facts", {}).items()},
            conversation_topic=str(state_payload.get("conversation_topic", "")),
            current_goal=str(state_payload.get("current_goal", "")),
            experiences=[
                {str(k): str(v) for k, v in item.items()}
                for item in state_payload.get("experiences", [])
                if isinstance(item, dict)
            ][-MAX_EXPERIENCES:],
        )
        facts = payload.get("facts", state.user_facts)
        turns = payload.get("turns", [])
        return cls(
            facts={str(k): str(v) for k, v in facts.items()} if isinstance(facts, dict) else state.user_facts.copy(),
            turns=[
                {"user": str(t["user"]), "assistant": str(t["assistant"])}
                for t in turns
                if isinstance(t, dict) and "user" in t and "assistant" in t
            ][-MAX_STORED_TURNS:],
            state=state,
        )


class SkeAIConversation:
    """Conversation agent facade: self-model + memory + context + generation."""

    def __init__(
        self,
        model: TinyTransformerLM,
        tokenizer: HybridTokenizer,
        *,
        memory_path: str | Path | None = None,
        max_history_turns: int = 5,
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

    def _self_context(self) -> str:
        model = self.memory.state.self_model
        user = self.memory.state.user_facts
        lines = [
            f"أنا: {model.get('name', 'SkeAI')}",
            f"نوعي: {model.get('entity_type', 'ذكاء اصطناعي')}",
            "أنا لست إنسانًا.",
            f"مرحلة التطور: {model.get('development_stage', 'بداية التعلم')}",
            f"المشروع: {model.get('origin', 'مشروع مستقل')}",
        ]
        if user.get("user_name"):
            label = "اسم المستخدم"
            if user.get("username_is_real_name") == "false":
                label = "اسم المستخدم المدخل في النظام (ليس بالضرورة الاسم الحقيقي)"
            lines.append(f"{label}: {user['user_name']}")
        if user.get("likes"):
            lines.append(f"المستخدم يحب: {user['likes']}")
        if user.get("dislikes"):
            lines.append(f"المستخدم لا يحب: {user['dislikes']}")
        if self.memory.state.conversation_topic:
            lines.append(f"موضوع الحوار الحالي: {self.memory.state.conversation_topic}")
        return SYSTEM_LABEL + " " + " | ".join(lines) + "\n"

    def _update_dialogue_state(self, message: str) -> None:
        lowered = message.lower()
        if any(word in lowered for word in ("اسمك", "من أنت", "من انت", "who are you", "your name")):
            self.memory.state.conversation_topic = "هوية SkeAI"
        elif any(word in lowered for word in ("مساعدة", "ساعدني", "help")):
            self.memory.state.conversation_topic = "مساعدة المستخدم"
        elif any(word in lowered for word in ("ما رأيك", "رأيك", "opinion")):
            self.memory.state.conversation_topic = "رأي أو نقاش"
        elif message.endswith("?") or message.endswith("؟"):
            self.memory.state.conversation_topic = "سؤال المستخدم"

    def _build_prompt(self, message: str) -> str:
        turns = self.memory.turns[-self.max_history_turns:]
        parts = [self._self_context()]
        for turn in turns:
            parts.append(f"{USER_LABEL} {turn['user']}\n{ASSISTANT_LABEL} {turn['assistant']}\n")
        parts.append(f"{USER_LABEL} {message}\n{ASSISTANT_LABEL}")
        return "".join(parts)

    def _trim_to_context(self, prompt: str) -> str:
        """Trim dialogue history while always preserving the self-model."""
        limit = self.model.config.context_length
        system_context = self._self_context()
        system_tokens = self.tokenizer.encode(system_context)

        # The self-model is mandatory context. If it alone exceeds the model
        # context, keep its beginning rather than dropping identity entirely.
        if len(system_tokens) >= limit:
            return self.tokenizer.decode(
                system_tokens[:limit],
                skip_special_tokens=False,
            )

        if not prompt.startswith(system_context):
            # Defensive fallback for callers providing a custom prompt.
            token_ids = self.tokenizer.encode(prompt)
            if len(token_ids) <= limit:
                return prompt
            return self.tokenizer.decode(
                token_ids[-limit:],
                skip_special_tokens=False,
            )

        body = prompt[len(system_context):]
        marker = f"{USER_LABEL} "
        raw_chunks = [chunk for chunk in body.split(marker) if chunk]

        if not raw_chunks:
            return system_context

        # The final chunk is always the current user turn followed by SkeAI:.
        current = marker + raw_chunks[-1]
        current_tokens = self.tokenizer.encode(current)

        # Identity + current turn have priority over older dialogue.
        if len(system_tokens) + len(current_tokens) > limit:
            available = max(1, limit - len(system_tokens))
            current_ids = self.tokenizer.encode(current)
            return system_context + self.tokenizer.decode(
                current_ids[-available:],
                skip_special_tokens=False,
            )

        kept_reversed: list[str] = []
        used = len(system_tokens) + len(current_tokens)

        # Add the newest previous turns first. Old turns are discarded first.
        for previous in reversed(raw_chunks[:-1]):
            candidate = marker + previous
            candidate_tokens = self.tokenizer.encode(candidate)

            if used + len(candidate_tokens) > limit:
                break

            kept_reversed.append(candidate)
            used += len(candidate_tokens)

        kept_reversed.reverse()

        return system_context + "".join(kept_reversed) + current

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
        message = " ".join(message.strip().split())
        if not message:
            raise ValueError("message cannot be empty.")

        self.memory.remember_from_user(message)
        self._update_dialogue_state(message)
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

        for label in (USER_LABEL, ASSISTANT_LABEL, SYSTEM_LABEL):
            if label in response:
                response = response.split(label, 1)[0].rstrip()
        if not response:
            response = "لا أملك ردًا جيدًا بعد. سأتعلم من هذه المحادثة."

        self.memory.add_turn(message, response)
        if self.memory_path is not None:
            self.memory.save(self.memory_path)
        return response

    def reset(self, *, clear_persistent_memory: bool = False) -> None:
        self.memory.turns.clear()
        self.memory.state.conversation_topic = ""
        self.memory.state.current_goal = ""
        if clear_persistent_memory:
            self.memory.facts.clear()
            self.memory.state.user_facts.clear()
            self.memory.state.experiences.clear()
        if self.memory_path is not None:
            self.memory.save(self.memory_path)


__all__ = ["SkeAIState", "ConversationMemory", "SkeAIConversation"]
