"""Benchmark SkeAI's from-scratch numerical engine.

Run from the repository root:

    python -m benchmarks.benchmark_engine

The benchmark is dependency-free so the same test can run on a phone through
Termux or on a desktop computer.
"""

from __future__ import annotations

import platform

from src.skeai.benchmark import benchmark
from src.skeai.dataset import CharacterLanguageDataset
from src.skeai.language_model import TinyCharacterLanguageModel
from src.skeai.layers import Dense
from src.skeai.loss import CrossEntropyLoss
from src.skeai.optimizer import SGD
from src.skeai.tensor import Tensor
from src.skeai.tokenizer import CharacterTokenizer
from src.skeai.trainer import Trainer


def make_matrix(rows: int, cols: int, scale: float = 0.01) -> Tensor:
    return Tensor(
        [
            [((row * cols + col) % 17 - 8) * scale for col in range(cols)]
            for row in range(rows)
        ]
    )


def main() -> None:
    print("=== SkeAI Engine Benchmark ===")
    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")

    left = make_matrix(16, 256)
    right = make_matrix(256, 64)

    matmul_result = benchmark(
        "tensor.matmul",
        lambda: left.matmul(right),
        repeats=5,
        warmups=1,
    )
    matmul_ops = 2 * 16 * 256 * 64
    matmul_mflops = (
        matmul_ops / matmul_result.average_seconds / 1_000_000.0
    )

    print(
        f"matmul=16x256 @ 256x64 "
        f"avg={matmul_result.average_milliseconds:.3f}ms "
        f"throughput={matmul_mflops:.3f}M-FLOP/s"
    )

    dense = Dense(256, 64, seed=7)
    dense_input = left
    dense_gradient = make_matrix(16, 64, scale=0.001)

    forward_result = benchmark(
        "dense.forward",
        lambda: dense.forward(dense_input),
        repeats=5,
        warmups=1,
    )

    dense.forward(dense_input)
    backward_result = benchmark(
        "dense.backward",
        lambda: dense.backward(dense_gradient),
        repeats=5,
        warmups=1,
    )

    print(f"dense_forward={forward_result.average_milliseconds:.3f}ms")
    print(f"dense_backward={backward_result.average_milliseconds:.3f}ms")

    text = (
        "SkeAI says hello. "
        "SkeAI says مرحبا. "
        "hello there. "
        "مرحبا بك. "
        "I am learning. "
        "أنا أتعلم."
    )
    tokenizer = CharacterTokenizer()
    tokenizer.fit([text])
    model = TinyCharacterLanguageModel(
        tokenizer=tokenizer,
        context_length=8,
        hidden_size=32,
    )

    dataset = CharacterLanguageDataset(
        text=text,
        tokenizer=tokenizer,
        context_length=8,
    )
    inputs, targets = dataset.all_batches(batch_size=16)[0]

    trainer = Trainer(
        model=model.network,
        optimizer=SGD(learning_rate=0.05),
        loss=CrossEntropyLoss(),
        enable_timing=True,
    )

    step_result = benchmark(
        "trainer.train_step",
        lambda: trainer.train_step(inputs, targets),
        repeats=3,
        warmups=1,
    )

    print(
        f"train_step=batch16 "
        f"avg={step_result.average_milliseconds:.3f}ms "
        f"throughput={16 / step_result.average_seconds:.2f}examples/s"
    )
    print(f"train_step_forward={trainer.last_step_timing['forward_ms']:.3f}ms")
    print(f"train_step_loss={trainer.last_step_timing['loss_ms']:.3f}ms")
    print(f"train_step_backward={trainer.last_step_timing['backward_ms']:.3f}ms")
    print(f"train_step_optimizer={trainer.last_step_timing['optimizer_ms']:.3f}ms")
    print(f"model_parameters={model.network.parameter_count()}")


if __name__ == "__main__":
    main()
