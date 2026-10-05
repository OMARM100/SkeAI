"""Text generation helpers for SkeAI Level 2."""

from __future__ import annotations

from .tokenizer import HybridTokenizer
from .transformer import TinyTransformerLM


def generate_text(
    model: TinyTransformerLM,
    tokenizer: HybridTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.85,
    top_k: int = 0,
    seed: int | None = 1234,
) -> str:
    """Generate a continuation and return only the newly generated text."""
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")
    if not prompt:
        raise ValueError("prompt cannot be empty.")

    encoded = tokenizer.encode(prompt)
    generated = model.generate(
        encoded,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        eos_token_id=tokenizer.eos_id,
    )
    continuation = generated[len(encoded):]
    return tokenizer.decode(continuation)


__all__ = ["generate_text"]
