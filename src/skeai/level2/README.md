# SkeAI Level 2

Level 2 is an experimental token-level causal Transformer running on top of
the existing from-scratch Tensor/C++ backend.

## Architecture

```
text
  ↓
HybridTokenizer
  ↓
Token IDs
  ↓
Token Embedding + Position Embedding
  ↓
2 × Causal Multi-Head Self-Attention
  ↓
Feed-Forward + Residual Connections
  ↓
Language Model Head
  ↓
Next-token logits
```

The initial mobile-oriented configuration is intentionally small:

- context: 64
- model width: 32
- heads: 2
- feed-forward width: 64
- layers: 2

Level 1 remains unchanged. Level 2 training/backpropagation is deliberately
isolated until forward correctness, checkpoint compatibility, and tokenizer
behavior are validated.
