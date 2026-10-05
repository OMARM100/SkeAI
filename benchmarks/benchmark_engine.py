"""Benchmark SkeAI's from-scratch numerical engine.

Run from the repository root:

    python -m benchmarks.benchmark_engine

The benchmark is dependency-free so the same test can run on a phone through
Termux or on a desktop computer.
"""

from __future__ import annotations

import platform

from time import perf_counter

from src.skeai.benchmark import benchmark
from src.skeai.dataset import CharacterLanguageDataset
from src.skeai.engine import BACKEND_NAME
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


def profile_model_layers(
    model: TinyCharacterLanguageModel,
    inputs: Tensor,
    targets: Tensor,
    repeats: int = 5,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    """Profile each model layer on the real training batch.

    Returns average forward times, average backward times, loss-forward time,
    and loss-backward time. The backward pass mirrors Trainer: the first Dense
    layer skips input-gradient arithmetic.
    """
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
            elapsed = (perf_counter() - start) * 1000.0
            forward_totals[
                f"layer{index}_{type(layer).__name__}_forward_ms"
            ] += elapsed

        start = perf_counter()
        loss = CrossEntropyLoss()
        loss.forward(current, targets)
        loss_forward_total += (perf_counter() - start) * 1000.0

        start = perf_counter()
        gradient = loss.backward()
        loss_backward_total += (perf_counter() - start) * 1000.0

        for index in range(len(model.network.layers) - 1, -1, -1):
            layer = model.network.layers[index]
            start = perf_counter()

            if index == 0 and isinstance(layer, Dense):
                gradient = layer.backward(
                    gradient,
                    compute_input_gradient=False,
                )
            else:
                gradient = layer.backward(gradient)

            elapsed = (perf_counter() - start) * 1000.0
            backward_totals[
                f"layer{index}_{type(layer).__name__}_backward_ms"
            ] += elapsed

    forward_averages = {
        key: value / repeats for key, value in forward_totals.items()
    }
    backward_averages = {
        key: value / repeats for key, value in backward_totals.items()
    }

    return (
        forward_averages,
        backward_averages,
        loss_forward_total / repeats,
        loss_backward_total / repeats,
    )


def profile_dense_parts(
    dense: Dense,
    inputs: Tensor,
    grad_output: Tensor,
    repeats: int = 5,
) -> dict[str, float]:
    """Profile the internal arithmetic sections of a Dense layer."""
    dense.forward(inputs)

    output_data = dense._cached_output._data  # type: ignore[union-attr,attr-defined]
    input_data = inputs._data  # type: ignore[attr-defined]
    weight_data = dense.weights._data  # type: ignore[attr-defined]
    bias_data = dense.bias._data  # type: ignore[attr-defined]
    grad_output_data = grad_output._data  # type: ignore[attr-defined]
    grad_weight_data = dense.grad_weights._data  # type: ignore[attr-defined]
    grad_bias_data = dense.grad_bias._data  # type: ignore[attr-defined]

    batch_size, input_size = inputs.shape
    output_size = dense.weights.shape[1]

    totals = {
        "dense1_forward_reset_ms": 0.0,
        "dense1_forward_matmul_ms": 0.0,
        "dense1_forward_bias_ms": 0.0,
        "dense1_backward_clear_grad_w_ms": 0.0,
        "dense1_backward_clear_grad_b_ms": 0.0,
        "dense1_backward_grad_b_ms": 0.0,
        "dense1_backward_grad_w_ms": 0.0,
    }

    for _ in range(repeats):
        start = perf_counter()
        for batch in range(batch_size):
            result_row = output_data[batch]
            for output_index in range(output_size):
                result_row[output_index] = 0.0
        totals["dense1_forward_reset_ms"] += (perf_counter() - start) * 1000.0

        start = perf_counter()
        for batch in range(batch_size):
            x_row = input_data[batch]
            result_row = output_data[batch]
            for input_index in range(input_size):
                value = x_row[input_index]
                weight_row = weight_data[input_index]
                for output_index in range(output_size):
                    result_row[output_index] += value * weight_row[output_index]
        totals["dense1_forward_matmul_ms"] += (perf_counter() - start) * 1000.0

        start = perf_counter()
        for batch in range(batch_size):
            result_row = output_data[batch]
            for output_index in range(output_size):
                result_row[output_index] += bias_data[output_index]
        totals["dense1_forward_bias_ms"] += (perf_counter() - start) * 1000.0

        start = perf_counter()
        for input_index in range(input_size):
            grad_weight_row = grad_weight_data[input_index]
            for output_index in range(output_size):
                grad_weight_row[output_index] = 0.0
        totals["dense1_backward_clear_grad_w_ms"] += (
            (perf_counter() - start) * 1000.0
        )

        start = perf_counter()
        for output_index in range(output_size):
            grad_bias_data[output_index] = 0.0
        totals["dense1_backward_clear_grad_b_ms"] += (
            (perf_counter() - start) * 1000.0
        )

        start = perf_counter()
        for batch in range(batch_size):
            go_row = grad_output_data[batch]
            for output_index in range(output_size):
                grad_bias_data[output_index] += go_row[output_index]
        totals["dense1_backward_grad_b_ms"] += (
            (perf_counter() - start) * 1000.0
        )

        for input_index in range(input_size):
            grad_weight_row = grad_weight_data[input_index]
            for output_index in range(output_size):
                grad_weight_row[output_index] = 0.0

        start = perf_counter()
        input_range = range(input_size)
        output_range = range(output_size)
        for batch in range(batch_size):
            x_row = input_data[batch]
            go_row = grad_output_data[batch]
            for input_index in input_range:
                x_value = x_row[input_index]
                grad_weight_row = grad_weight_data[input_index]
                for output_index in output_range:
                    grad_weight_row[output_index] += x_value * go_row[output_index]
        totals["dense1_backward_grad_w_ms"] += (
            (perf_counter() - start) * 1000.0
        )

    return {key: value / repeats for key, value in totals.items()}


def main() -> None:
    print("=== SkeAI Engine Benchmark ===")
    print(f"python={platform.python_version()}")
    print(f"platform={platform.platform()}")
    print(f"engine_backend={BACKEND_NAME}")

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

    warmups = 1
    repeats = 5

    for _ in range(warmups):
        trainer.train_step(inputs, targets)

    stage_totals = {
        "forward_ms": 0.0,
        "loss_ms": 0.0,
        "backward_ms": 0.0,
        "optimizer_ms": 0.0,
        "total_ms": 0.0,
    }

    total_start = perf_counter()
    for _ in range(repeats):
        trainer.train_step(inputs, targets)
        for key in stage_totals:
            stage_totals[key] += trainer.last_step_timing[key]
    total_elapsed = perf_counter() - total_start

    average_step_seconds = total_elapsed / repeats

    print(
        f"train_step=batch16 "
        f"avg={average_step_seconds * 1000.0:.3f}ms "
        f"throughput={16 / average_step_seconds:.2f}examples/s"
    )
    print(
        f"train_step_forward="
        f"{stage_totals['forward_ms'] / repeats:.3f}ms"
    )
    print(
        f"train_step_loss="
        f"{stage_totals['loss_ms'] / repeats:.3f}ms"
    )
    print(
        f"train_step_backward="
        f"{stage_totals['backward_ms'] / repeats:.3f}ms"
    )
    print(
        f"train_step_optimizer="
        f"{stage_totals['optimizer_ms'] / repeats:.3f}ms"
    )

    print("=== Layer Profile ===")
    profile_model = TinyCharacterLanguageModel(
        tokenizer=tokenizer,
        context_length=8,
        hidden_size=32,
    )

    profile_forward, profile_backward, loss_forward, loss_backward = (
        profile_model_layers(
            profile_model,
            inputs,
            targets,
            repeats=5,
        )
    )

    for key, value in profile_forward.items():
        print(f"{key}={value:.3f}ms")

    print(f"loss_forward_profile={loss_forward:.3f}ms")

    for key, value in profile_backward.items():
        print(f"{key}={value:.3f}ms")

    print(f"loss_backward_profile={loss_backward:.3f}ms")

    forward_total = sum(profile_forward.values())
    backward_total = sum(profile_backward.values())

    print(f"profile_forward_layers_total={forward_total:.3f}ms")
    print(f"profile_backward_layers_total={backward_total:.3f}ms")

    dense1 = profile_model.network.layers[0]
    dense1_grad_output = make_matrix(
        inputs.shape[0],
        dense1.weights.shape[1],
        scale=0.001,
    )
    if BACKEND_NAME == "python":
        dense1_parts = profile_dense_parts(
            dense1,
            inputs,
            dense1_grad_output,
            repeats=5,
        )

        print("=== Dense #1 Python Internal Profile ===")
        for key, value in dense1_parts.items():
            print(f"{key}={value:.3f}ms")
    else:
        print("dense1_internal_profile=covered_by_native_kernels")

    print(f"model_parameters={model.network.parameter_count()}")


if __name__ == "__main__":
    main()
