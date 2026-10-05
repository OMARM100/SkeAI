"""End-to-end benchmark for SkeAI Level 2."""
from __future__ import annotations

import argparse
from time import perf_counter
from pathlib import Path

from src.skeai.level2.tokenizer import HybridTokenizer
from src.skeai.level2.transformer import TinyTransformerLM
from src.skeai.level2.trainer import Level2Trainer
from src.skeai.loss import CrossEntropyLoss
from src.skeai.optimizer import SGD


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "samples" / "tiny_corpus.txt"
DIALOGUE = ROOT / "data" / "samples" / "tiny_dialogue.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SkeAI Level 2.")
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    corpus = CORPUS.read_text(encoding="utf-8")
    dialogue = DIALOGUE.read_text(encoding="utf-8")
    text = corpus + "\n" + dialogue

    tokenizer = HybridTokenizer()
    tokenizer.fit([text], max_units=512, min_frequency=2)

    token_ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    sequence_length = min(32, len(token_ids) - 1)
    inputs = token_ids[:sequence_length]
    targets = token_ids[1:sequence_length + 1]

    model = TinyTransformerLM(tokenizer.vocab_size)
    trainer = Level2Trainer(
        model=model,
        optimizer=SGD(learning_rate=0.003),
    )
    loss = CrossEntropyLoss()

    # Warm up the native extension and allocator before timing.
    logits = model.forward(inputs)
    loss.forward(logits, targets)
    model.backward(loss.backward())

    forward_start = perf_counter()
    for _ in range(args.steps):
        model.forward(inputs)
    forward_ms = (perf_counter() - forward_start) * 1000.0 / args.steps

    backward_start = perf_counter()
    for _ in range(args.steps):
        logits = model.forward(inputs)
        loss.forward(logits, targets)
        model.backward(loss.backward())
    backward_ms = (perf_counter() - backward_start) * 1000.0 / args.steps

    train_start = perf_counter()
    for _ in range(args.steps):
        trainer.train_step(inputs, targets)
    train_step_ms = (perf_counter() - train_start) * 1000.0 / args.steps

    print("=== SkeAI Level 2 Benchmark ===")
    print(f"vocabulary_size={tokenizer.vocab_size}")
    print(f"sequence_length={sequence_length}")
    print(f"parameter_count={model.parameter_count()}")
    print(f"forward_ms={forward_ms:.3f}")
    print(f"forward_backward_ms={backward_ms:.3f}")
    print(f"train_step_ms={train_step_ms:.3f}")
    print("attention_backend=cpp")
    print("matrix_backend=cpp")
    print("softmax_backend=cpp")


if __name__ == "__main__":
    main()
