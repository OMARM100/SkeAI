# SkeAI

SkeAI is an experimental artificial intelligence project built from scratch.

The first goal is intentionally small: build and understand a lightweight Arabic/English language-learning model without starting from a pretrained model.

## Project goals

- Build the core components ourselves.
- Start with a very small dataset.
- Support Arabic and English text.
- Train initially on CPU so development can happen on a phone.
- Keep the architecture lightweight and understandable.
- Gradually add GPU acceleration and native backends later.
- Eventually integrate a lightweight inference/runtime into SkeOS for kernel testing and diagnostics.

## Current stage

**SkeAI 0.1 — Tiny character language model**

The project now includes:

- A character-level Arabic/English tokenizer.
- A from-scratch tensor engine.
- Dense, ReLU, and Tanh layers.
- Cross-entropy and mean-squared-error losses.
- SGD optimization.
- A sequential neural network.
- A tiny fixed-context character language model.
- A bilingual toy corpus.
- Text generation.
- JSON model checkpoints with resume training.
- Automated Python tests through GitHub Actions.

This model is intentionally small. It is a learning and engineering milestone, not a general-purpose assistant.

## Run training

From the repository root:

```bash
python -m training.train_tiny
```

The model is saved to:

```text
models/tiny_character_model.json
```

To continue training from the saved checkpoint:

```bash
python -m training.train_tiny --resume
```

To run the tests locally:

```bash
python -m unittest discover -s tests -v
```

## Planned architecture

```
Text
  ↓
Tokenizer
  ↓
Tokens
  ↓
Context Builder
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
Text Generation
```

## Development roadmap

1. Project foundation
2. Tokenizer
3. Tensor and numerical operations
4. Basic neural network layers
5. Loss and optimizer
6. Training loop
7. Tiny character language model
8. Checkpointing and resume training
9. Better Arabic/English dataset
10. Improved tokenization
11. Small Transformer
12. Mobile optimization
13. SkeOS integration

## Philosophy

SkeAI is intended to be a learning project first. A small working model is more valuable at this stage than a large model whose internals are hidden behind a framework.

The project should remain modular so the same model concepts can later be implemented in native code for SkeOS.

## Status

Early development.
