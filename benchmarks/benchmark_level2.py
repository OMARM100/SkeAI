"""Smoke benchmark for SkeAI Level 2."""

from __future__ import annotations

from time import perf_counter
from pathlib import Path

from src.skeai.level2.tokenizer import HybridTokenizer
from src.skeai.level2.transformer import TinyTransformerLM


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "samples" / "tiny_corpus.txt"
DIALOGUE = ROOT / "data" / "samples" / "tiny_dialogue.txt"


def main() -> None:
    corpus = CORPUS.read_text(encoding="utf-8")
    dialogue = DIALOGUE.read_text(encoding="utf-8")
    text = corpus + "\n" + dialogue

    tokenizer = HybridTokenizer()
    tokenizer.fit([text], max_units=512, min_frequency=2)

    sample = "ما هو اسمك؟"
    token_ids = tokenizer.encode(sample)
    model = TinyTransformerLM(tokenizer.vocab_size)

    start = perf_counter()
    logits = model.forward(token_ids)
    elapsed = perf_counter() - start

    print("=== SkeAI Level 2 Benchmark ===")
    print(f"vocabulary_size={tokenizer.vocab_size}")
    print(f"character_count={len(sample)}")
    print(f"token_count={len(token_ids)}")
    print(f"compression_ratio={len(sample) / max(len(token_ids), 1):.2f}x")
    print(f"parameter_count={model.parameter_count()}")
    print(f"forward_shape={logits.shape}")
    print(f"forward_ms={elapsed * 1000.0:.3f}")
    print("backend=tensor/cpp matmul + Python attention orchestration")


if __name__ == "__main__":
    main()
