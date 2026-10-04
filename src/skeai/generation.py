"""Text generation helpers for SkeAI."""

from __future__ import annotations

from .language_model import TinyCharacterLanguageModel


def generate_text(
    model: TinyCharacterLanguageModel,
    prompt: str,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    seed: int | None = None,
) -> str:
    return model.generate(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )


__all__ = ["generate_text"]
