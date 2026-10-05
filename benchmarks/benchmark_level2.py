"""End-to-end benchmark for SkeAI Level 2."""
from __future__ import annotations

import argparse
from time import perf_counter
from pathlib import Path

from src.skeai.level2.tokenizer import HybridTokenizer
from src.skeai.level2.transformer import TinyTransformerLM
from src.skeai.level2.trainer import Level2Trainer
from src.skeai.optimizer import SGD


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "data" / "samples" / "tiny_corpus.txt"
DIALOGUE = ROOT / "data" / "samples" / "tiny_dialogue.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SkeAI Level 2.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if args.steps <= 0:
        raise ValueError("--steps must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    text = (
        CORPUS.read_text(encoding="utf-8")
        + "\n"
        + DIALOGUE.read_text(encoding="utf-8")
    )

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

    warmup_batch = [inputs] * min(2, args.batch_size)
    warmup_targets = [targets] * min(2, args.batch_size)
    trainer.train_batch(warmup_batch, warmup_targets)

    forward_start = perf_counter()
    for _ in range(args.steps):
        model.forward(inputs)
    forward_ms = (perf_counter() - forward_start) * 1000.0 / args.steps

    model = TinyTransformerLM(tokenizer.vocab_size)
    trainer = Level2Trainer(
        model=model,
        optimizer=SGD(learning_rate=0.003),
    )

    single_start = perf_counter()
    for _ in range(args.steps):
        trainer.train_step(inputs, targets)
    train_step_ms = (perf_counter() - single_start) * 1000.0 / args.steps

    model = TinyTransformerLM(tokenizer.vocab_size)
    trainer = Level2Trainer(
        model=model,
        optimizer=SGD(learning_rate=0.003),
    )
    batch_inputs = [inputs] * args.batch_size
    batch_targets = [targets] * args.batch_size

    batch_start = perf_counter()
    for _ in range(args.steps):
        trainer.train_batch(batch_inputs, batch_targets)
    batch_total_ms = (perf_counter() - batch_start) * 1000.0
    batch_per_step_ms = batch_total_ms / args.steps
    batch_per_example_ms = batch_per_step_ms / args.batch_size
    batch_examples_per_second = (
        args.steps * args.batch_size
        / max(batch_total_ms / 1000.0, 1e-12)
    )

    print("=== SkeAI Level 2 Benchmark ===")
    print(f"vocabulary_size={tokenizer.vocab_size}")
    print(f"sequence_length={sequence_length}")
    print(f"parameter_count={model.parameter_count()}")
    print(f"forward_ms={forward_ms:.3f}")
    print(f"train_step_ms={train_step_ms:.3f}")
    print(f"batch_size={args.batch_size}")
    print(f"train_batch_total_ms={batch_total_ms:.3f}")
    print(f"train_batch_ms_per_step={batch_per_step_ms:.3f}")
    print(f"train_batch_ms_per_example={batch_per_example_ms:.3f}")
    print(f"train_batch_examples_per_second={batch_examples_per_second:.2f}")
    print("training_backend=cpp_batch_fused")
    print("attention_backend=cpp")
    print("matrix_backend=cpp")
    print("softmax_backend=cpp")


if __name__ == "__main__":
    main()
