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

**SkeAI 0.1 — Foundation**

At this stage the repository contains the project structure only. The neural network, tokenizer, training loop, and language model will be implemented incrementally.

## Planned architecture

```
Text
  ↓
Tokenizer
  ↓
Tokens
  ↓
Embeddings
  ↓
Neural Network
  ↓
Prediction
  ↓
Text generation
```

## Development roadmap

1. Project foundation
2. Tokenizer
3. Tensor and numerical operations
4. Basic neural network layers
5. Loss and optimizer
6. Training loop
7. Text generation
8. Arabic/English language experiments
9. Small Transformer
10. Mobile optimization
11. SkeOS integration

## Philosophy

SkeAI is intended to be a learning project first. A small working model is more valuable at this stage than a large model whose internals are hidden behind a framework.

The project should remain modular so the same model concepts can later be implemented in native code for SkeOS.

## Status

Early development.
