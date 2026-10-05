"""Benchmark SkeAI's from-scratch C++ numerical engine."""
from __future__ import annotations

import platform
from time import perf_counter

from src.skeai.benchmark import benchmark
from src.skeai.dataset import CharacterLanguageDataset
from src.skeai.engine import BACKEND_NAME, SCALAR_TYPE
from src.skeai.language_model import TinyCharacterLanguageModel
from src.skeai.layers import Dense
from src.skeai.loss import CrossEntropyLoss
from src.skeai.optimizer import SGD
from src.skeai.tensor import Tensor
from src.skeai.tokenizer import CharacterTokenizer
from src.skeai.trainer import Trainer


def make_matrix(rows: int, cols: int, scale: float = 0.01) -> Tensor:
    return Tensor([
        [((row * cols + col) % 17 - 8) * scale for col in range(cols)]
        for row in range(rows)
    ])


def profile_model_layers(
    model: TinyCharacterLanguageModel,
    inputs: Tensor,
    targets: list[int],
    repeats: int = 5,
):
    forward_totals = {
        f"layer{index}_{type(layer).__name__}_forward_ms": 0.0
        for index, layer in enumerate(model.network.layers)
    }
    backward_totals = {
        f"layer{index}_{type(layer).__name__}_backward_ms": 0.0
        for index, layer in enumerate(model.network.layers)
    }
    loss_forward_total = 0.0
    loss_backward_total = 0.0

    for _ in range(repeats):
        current = inputs

        for index, layer in enumerate(model.network.layers):
            start = perf_counter()
            current = layer.forward(current)
            forward_totals[
                f"layer{index}_{type(layer).__name__}_forward_ms"
            ] += (perf_counter() - start) * 1000.0

        loss = CrossEntropyLoss()
        start = perf_counter()
        loss.forward(current, targets)
        loss_forward_total += (perf_counter() - start) * 1000.0

        start = perf_counter()
        gradient = loss.backward()
        loss_backward_total += (perf_counter() - start) * 1000.0

        for index in range(len(model.network.layers) - 1, -1, -1):
            layer = model.network.layers[index]
            start = perf_counter()

            if isinstance(layer, Dense):
                gradient = layer.backward(
                    gradient,
                    compute_input_gradient=(index != 0),
                )
            else:
                gradient = layer.backward(gradient)

            backward_totals[
                f"layer{index}_{type(layer).__name__}_backward_ms"
            ] += (perf_counter() - start) * 1000.0

    return (
        {key: value / repeats for key, value in forward_totals.items()},
        {key: value / repeats for key, value in backward_totals.items()},
        loss_forward_total / repeats,
        loss_backward_total / repeats,
    )


def main() -> None:
    print("=== SkeAI C++ Engine Benchmark ===")
    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"engine_backend={BACKEND_NAME}")
    print(f"engine_scalar={SCALAR_TYPE}")

    left = make_matrix(16, 256)
    right = make_matrix(256, 64)

    result = benchmark(
        "tensor.matmul",
        lambda: left.matmul(right),
        repeats=5,
        warmups=1,
    )
    matmul_ops = 2 * 16 * 256 * 64
    mflops = matmul_ops / result.average_seconds / 1_000_000.0

    print(
        f"matmul=16x256 @ 256x64 "
        f"avg={result.average_milliseconds:.3f}ms "
        f"throughput={mflops:.3f}M-FLOP/s"
    )

    dense = Dense(256, 64, seed=7)
    dense_input = left
    dense_gradient = make_matrix(16, 64, scale=0.001)

    forward = benchmark(
        "dense.forward",
        lambda: dense.forward(dense_input),
        repeats=5,
        warmups=1,
    )

    dense.forward(dense_input)
    backward = benchmark(
        "dense.backward",
        lambda: dense.backward(dense_gradient),
        repeats=5,
        warmups=1,
    )

    print(f"dense_forward={forward.average_milliseconds:.3f}ms")
    print(f"dense_backward={backward.average_milliseconds:.3f}ms")

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

    warmups = 1
    repeats = 5

    for _ in range(warmups):
        trainer.train_step(inputs, targets)

    totals = {
        "forward_ms": 0.0,
        "loss_ms": 0.0,
        "backward_ms": 0.0,
        "optimizer_ms": 0.0,
    }

    start = perf_counter()

    for _ in range(repeats):
        trainer.train_step(inputs, targets)
        for key in totals:
            totals[key] += trainer.last_step_timing[key]

    elapsed = perf_counter() - start
    average_step = elapsed / repeats

    print(
        f"train_step=batch16 "
        f"avg={average_step * 1000.0:.3f}ms "
        f"throughput={16 / average_step:.2f}examples/s"
    )
    print(f"train_step_forward={totals['forward_ms'] / repeats:.3f}ms")
    print(f"train_step_loss={totals['loss_ms'] / repeats:.3f}ms")
    print(f"train_step_backward={totals['backward_ms'] / repeats:.3f}ms")
    print(f"train_step_optimizer={totals['optimizer_ms'] / repeats:.3f}ms")

    forward_profile, backward_profile, loss_forward, loss_backward = (
        profile_model_layers(model, inputs, targets, repeats=5)
    )

    print("=== Layer Profile ===")
    for key, value in forward_profile.items():
        print(f"{key}={value:.3f}ms")
    print(f"loss_forward_profile={loss_forward:.3f}ms")

    for key, value in backward_profile.items():
        print(f"{key}={value:.3f}ms")
    print(f"loss_backward_profile={loss_backward:.3f}ms")

    print(
        f"profile_forward_layers_total="
        f"{sum(forward_profile.values()):.3f}ms"
    )
    print(
        f"profile_backward_layers_total="
        f"{sum(backward_profile.values()):.3f}ms"
    )
    print("native_kernel_profile=covered_by_cpp_engine")
    print(f"model_parameters={model.network.parameter_count()}")


if __name__ == "__main__":
    main()
