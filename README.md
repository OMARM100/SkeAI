# SkeAI

SkeAI is an experimental artificial intelligence project built from scratch.

The first goal is intentionally small: build and understand a lightweight Arabic/English language-learning model without starting from a pretrained model.

## Project goals

- Build the core components ourselves.
- Start with a very small dataset.
- Support Arabic and English text.
- Train initially on CPU so development can happen on a phone.
- Keep the architecture lightweight and understandable.
- Measure performance instead of guessing where the bottlenecks are.
- Gradually add native/GPU acceleration later without hiding the core implementation.
- Eventually integrate a lightweight runtime into SkeOS for kernel testing and diagnostics.

## Current stage

**SkeAI 0.2 — Performance engine foundation**

The project now includes:

- A character-level Arabic/English tokenizer.
- A from-scratch tensor engine.
- Dense, ReLU, and Tanh layers.
- Cross-entropy and mean-squared-error losses.
- SGD optimization with in-place parameter updates.
- A sequential neural network.
- A tiny fixed-context character language model.
- A bilingual toy corpus.
- Text generation.
- JSON model checkpoints with resume training.
- Training and engine timing telemetry.
- A dependency-free numerical benchmark.
- Automated Python tests through GitHub Actions.

The model is intentionally small. It is a learning and engineering milestone, not a general-purpose assistant.

## Run training

From the repository root:

```bash
python -m training.train_tiny
```

The training command now reports:

- Setup time.
- Per-epoch elapsed time.
- Average batch-step time.
- Examples per second.
- Last-step timing for forward, loss, backward, and optimizer stages.
- Total training time.
- Checkpoint write time.
- Total command time.

The model is saved to:

```text
models/tiny_character_model.json
```

To continue training from the saved checkpoint:

```bash
python -m training.train_tiny --resume
```

## Benchmark the engine

Run the dependency-free benchmark:

```bash
python -m benchmarks.benchmark_engine
```

It measures:

- Tensor matrix multiplication.
- Dense forward.
- Dense backward.
- A real SkeAI training step.
- Approximate matrix-operation throughput.
- Model parameter count.
- Timing of the main training stages.

Run it on the phone and later on a desktop using the same command. The results give us a repeatable baseline before adding larger models.

## Run tests

Locally:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same test suite on every push to `main` and every pull request.

## Current architecture

```text
Text
  ↓
Tokenizer
  ↓
Tokens
  ↓
Context Builder
  ↓
Tensor Engine
  ↓
Neural Network
  ↓
Logits
  ↓
Loss
  ↓
Backpropagation
  ↓
Optimizer
  ↓
Updated Weights
  ↓
Checkpoint / Runtime
  ↓
Text Generation
```

## Long-term runtime architecture

The long-term goal is larger than a text generator. We want SkeAI to become a small, modular runtime that can learn and execute multiple classes of tasks.

```text
                 ┌─────────────────────────────┐
                 │          SkeAI Core          │
                 │ Tensor + NN + Training       │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │       Model / Memory        │
                 │ language + learned state    │
                 └──────────────┬──────────────┘
                                │
                 ┌──────────────▼──────────────┐
                 │     Runtime / Orchestrator  │
                 │ tasks + tools + scheduling   │
                 └───────┬─────────┬───────────┘
                         │         │
                ┌────────▼───┐ ┌──▼───────────┐
                │ PC / OS    │ │ Voice / I/O │
                │ tools      │ │ input/output │
                └────────────┘ └──────────────┘
                         │
                ┌────────▼─────────┐
                │      SkeOS       │
                │ kernel support   │
                │ diagnostics      │
                └──────────────────┘
```

This architecture is a future direction. The current release does not yet control a computer, perform autonomous multi-step tasks, or provide real-time voice conversation.

## Performance direction

The performance engine is being improved before the model is made larger.

Current hot-path improvements include:

1. Direct matrix accumulation in `Tensor.matmul`.
2. One-pass Dense backward computation for `grad_W`, `grad_b`, and `grad_X`.
3. In-place SGD parameter updates.
4. Reusing the cross-entropy gradient instead of rebuilding it during backward.
5. Runtime timing telemetry for the training stages.

The next performance goal is to reduce allocation and Python-loop overhead further while keeping the implementation understandable. After that, we can consider larger contexts, better datasets, a small Transformer, and optimized/native backends.

## Development roadmap

1. Project foundation
2. Tokenizer
3. Tensor and numerical operations
4. Basic neural network layers
5. Loss and optimizer
6. Training loop
7. Tiny character language model
8. Checkpointing and resume training
9. Performance telemetry and engine benchmark
10. Tensor/gradient performance optimization
11. Better Arabic/English dataset
12. Improved tokenization
13. Memory-efficient batching
14. Small Transformer
15. Native/GPU acceleration
16. Task/runtime system
17. Safe computer/OS tool interface
18. Voice input/output
19. SkeOS integration
20. Kernel diagnostics and development assistant

## Philosophy

SkeAI is intended to be a learning project first. A small working model is more valuable at this stage than a large model whose internals are hidden behind a framework.

The core engine should remain modular so the same model and runtime concepts can later be implemented in optimized native code for desktop systems and eventually SkeOS.

## Status

Early development — performance-engine foundation.
