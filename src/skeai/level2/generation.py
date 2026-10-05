""""Text generation helpers for SkeAI Level 2."""

from __future__ import annotations

import math
import random

from .tokenizer import HybridTokenizer
from .transformer import TinyTransformerLM


def _blocked_by_ngram(candidate: int, generated: list[int], n: int) -> bool:
    if n <= 0 or len(generated) < n - 1:
        return False
    prefix = generated[-(n - 1):]
    for index in range(len(generated) - n + 1):
        if generated[index:index + n - 1] == prefix and generated[index + n - 1] == candidate:
            return True
    return False


def generate_text(
    model: TinyTransformerLM,
    tokenizer: HybridTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 32,
    temperature: float = 0.85,
    top_k: int = 0,
    seed: int | None = 1234,
    repetition_penalty: float = 1.0,
    no_repeat_ngram_size: int = 0,
) -> str:
    """Generate a continuation and return only the newly generated text.

    The decoding controls are intentionally kept outside the Transformer so
    training remains unchanged while inference can avoid degenerate repetition.
    """
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string.")
    if not prompt:
        raise ValueError("prompt cannot be empty.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative.")
    if temperature < 0.0:
        raise ValueError("temperature cannot be negative.")
    if top_k < 0 or top_k > tokenizer.vocab_size:
        raise ValueError("top_k must be between 0 and the vocabulary size")
    if repetition_penalty < 1.0:
        raise ValueError("repetition_penalty must be at least 1.0")
    if no_repeat_ngram_size < 0:
        raise ValueError("no_repeat_ngram_size cannot be negative")

    encoded = tokenizer.encode(prompt)
    generated = list(encoded)
    rng = random.Random(seed)
    banned = {tokenizer.pad_id, tokenizer.bos_id}

    for _ in range(max_new_tokens):
        logits = model.next_logits(generated)
        candidates = [index for index in range(tokenizer.vocab_size) if index not in banned]

        if repetition_penalty > 1.0:
            for token_id in set(generated):
                if token_id in banned:
                    continue
                if logits[token_id] > 0.0:
                    logits[token_id] /= repetition_penalty
                else:
                    logits[token_id] *= repetition_penalty

        if no_repeat_ngram_size > 1:
            candidates = [
                token_id
                for token_id in candidates
                if not _blocked_by_ngram(token_id, generated, no_repeat_ngram_size)
            ] or candidates

        if temperature == 0.0:
            next_token = max(candidates, key=lambda index: logits[index])
        else:
            scale = 1.0 / temperature
            if top_k:
                candidates = sorted(
                    candidates,
                    key=lambda index: logits[index],
                    reverse=True,
                )[:min(top_k, len(candidates))]

            maximum = max(logits[index] * scale for index in candidates)
            weights = [
                math.exp(logits[index] * scale - maximum)
                for index in candidates
            ]
            next_token = rng.choices(candidates, weights=weights, k=1)[0]

        generated.append(next_token)
        if next_token == tokenizer.eos_id:
            break

    continuation = generated[len(encoded):]
    return tokenizer.decode(continuation)


__all__ = ["generate_text"]
