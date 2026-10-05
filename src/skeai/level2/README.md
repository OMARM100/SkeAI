# SkeAI Level 2

Level 2 is SkeAI's first token-level causal Transformer milestone. It remains
from scratch and runs on the existing contiguous Tensor/C++ numerical engine.

## Architecture

```
text
  ↓
HybridTokenizer
  ↓
Token IDs + character fallback
  ↓
Token Embedding + learned Position Embedding
  ↓
Pre-LN Transformer Block
  ├─ Multi-Head Causal Self-Attention
  │    ├─ Q/K/V projections
  │    ├─ scaled QKᵀ
  │    ├─ causal softmax
  │    └─ attention × V
  ├─ residual connection
  ├─ Feed-Forward + ReLU
  └─ residual connection
  ↓
Final LayerNorm
  ↓
Language Model Head
  ↓
Next-token logits
```

The default mobile-oriented model is intentionally small:

- context: 64
- model width: 32
- heads: 2
- feed-forward width: 64
- layers: 2
- maximum vocabulary: 512

## Native acceleration

Level 2 keeps Python responsible for model orchestration and caching while the
heavy numerical path runs through the C++ engine:

- matrix multiplication
- causal softmax
- softmax backward
- Tensor storage
- cross-entropy
- SGD

This is important because the complete training step is the primary performance
signal, not an isolated kernel benchmark.

## Training

Build the native engine first:

    python -m tools.build_cpp

Run the full test suite:

    python -m unittest discover -s tests -v

Run the Level 2 benchmark:

    python -m benchmarks.benchmark_level2 --steps 5

Run a short smoke training session:

    python -m training.train_level2 --epochs 1 --max-train-steps 5 --max-validation-steps 5 --context 32

Run the normal Level 2 experiment:

    python -m training.train_level2

Resume from the best checkpoint:

    python -m training.train_level2 --resume

Training writes:

- `models/level2_transformer.json`
- `models/level2_tokenizer.json`

The trainer also reports validation loss, epoch time, steps/second, best
checkpoint status, and early-stopping state.

## Validation

The Level 2 test coverage includes:

- hybrid tokenizer compaction
- character fallback for unseen units
- tokenizer serialization
- forward output shape
- next-token logits
- LM-head finite-difference gradient checking
- attention Q finite-difference gradient checking
- native causal softmax correctness
- native softmax backward correctness
- end-to-end attention learning
- checkpoint round trips

## Current scope

This milestone is a complete small Transformer training path, not a production
LLM runtime. It does not yet include large-scale datasets, batched sequence
training, KV-cache inference, mixed precision, or a retrieval/web knowledge
layer. Those are separate scalability layers and do not need to change the
core Level 2 architecture.

Level 1 remains unchanged.
