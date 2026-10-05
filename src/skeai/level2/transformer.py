"""Tiny Transformer language model for SkeAI Level 2.

This is the first Level 2 architecture milestone. It keeps the existing
from-scratch Tensor/C++ backend and adds token embeddings, learned positions,
causal multi-head self-attention, a feed-forward block, residual connections,
and language-model logits.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List

from ..tensor import Tensor


@dataclass(frozen=True)
class TransformerConfig:
    context_length: int = 64
    d_model: int = 32
    n_heads: int = 2
    feed_forward_size: int = 64
    n_layers: int = 2
    max_vocab_size: int = 512
    seed: int = 42

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive.")
        if self.d_model <= 0 or self.feed_forward_size <= 0:
            raise ValueError("model dimensions must be positive.")
        if self.n_heads <= 0 or self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive.")
        if self.max_vocab_size <= 0:
            raise ValueError("max_vocab_size must be positive.")


def _rand_matrix(rows: int, cols: int, rng: random.Random, scale: float) -> Tensor:
    return Tensor([
        [rng.uniform(-scale, scale) for _ in range(cols)]
        for _ in range(rows)
    ])


def _add_rows(matrix: Tensor, bias: Tensor) -> Tensor:
    rows = matrix.to_list()
    bias_values = bias.to_list()
    return Tensor([
        [float(value) + float(bias_values[col]) for col, value in enumerate(row)]
        for row in rows
    ])


def _relu(matrix: Tensor) -> Tensor:
    return Tensor([
        [max(0.0, float(value)) for value in row]
        for row in matrix.to_list()
    ])


def _layer_norm(matrix: Tensor, eps: float = 1e-5) -> Tensor:
    result: List[List[float]] = []
    for row in matrix.to_list():
        mean = sum(row) / len(row)
        variance = sum((value - mean) ** 2 for value in row) / len(row)
        scale = 1.0 / math.sqrt(variance + eps)
        result.append([(value - mean) * scale for value in row])
    return Tensor(result)


def _softmax_rows(scores: Tensor, mask_causal: bool) -> Tensor:
    raw = scores.to_list()
    result: List[List[float]] = []
    for row_index, row in enumerate(raw):
        allowed = row[:]
        if mask_causal:
            for column in range(row_index + 1, len(allowed)):
                allowed[column] = -1e30

        maximum = max(allowed)
        exponentials = [math.exp(value - maximum) for value in allowed]
        total = sum(exponentials)
        result.append([value / total for value in exponentials])
    return Tensor(result)


class TinyTransformerLM:
    """Small causal Transformer intended for Level 2 experiments."""

    def __init__(self, vocab_size: int, config: TransformerConfig | None = None) -> None:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")

        self.config = config or TransformerConfig()
        if vocab_size > self.config.max_vocab_size:
            raise ValueError(
                f"vocab_size {vocab_size} exceeds max_vocab_size "
                f"{self.config.max_vocab_size}"
            )

        self.vocab_size = vocab_size
        self.head_dim = self.config.d_model // self.config.n_heads
        rng = random.Random(self.config.seed)
        d = self.config.d_model
        ff = self.config.feed_forward_size
        scale = 1.0 / math.sqrt(d)

        self.token_embedding = _rand_matrix(vocab_size, d, rng, scale)
        self.position_embedding = _rand_matrix(
            self.config.context_length,
            d,
            rng,
            scale,
        )

        self.blocks: list[dict[str, Tensor]] = []
        for _ in range(self.config.n_layers):
            self.blocks.append(
                {
                    "wq": _rand_matrix(d, d, rng, scale),
                    "wk": _rand_matrix(d, d, rng, scale),
                    "wv": _rand_matrix(d, d, rng, scale),
                    "wo": _rand_matrix(d, d, rng, scale),
                    "w1": _rand_matrix(d, ff, rng, scale),
                    "w2": _rand_matrix(ff, d, rng, scale),
                }
            )

        self.lm_head = _rand_matrix(d, vocab_size, rng, scale)

    def parameter_count(self) -> int:
        total = self.token_embedding.size + self.position_embedding.size
        total += self.lm_head.size
        total += sum(value.size for block in self.blocks for value in block.values())
        return total

    def _embed(self, token_ids: List[int]) -> Tensor:
        if not token_ids:
            raise ValueError("token_ids cannot be empty.")
        if len(token_ids) > self.config.context_length:
            token_ids = token_ids[-self.config.context_length:]

        token_rows = self.token_embedding.to_list()
        position_rows = self.position_embedding.to_list()
        d = self.config.d_model

        return Tensor([
            [
                float(token_rows[token_id][col])
                + float(position_rows[position][col])
                for col in range(d)
            ]
            for position, token_id in enumerate(token_ids)
        ])

    def forward(self, token_ids: List[int]) -> Tensor:
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in token_ids):
            raise ValueError("Token ID out of range.")

        x = self._embed(token_ids)
        scale = 1.0 / math.sqrt(self.head_dim)

        for block in self.blocks:
            norm_x = _layer_norm(x)
            q = norm_x.matmul(block["wq"])
            k = norm_x.matmul(block["wk"])
            v = norm_x.matmul(block["wv"])

            # One attention stream per head. The tensors stay small enough that
            # the existing C++ matmul kernel remains useful.
            attended_heads: List[Tensor] = []
            q_values = q.to_list()
            k_values = k.to_list()
            v_values = v.to_list()

            for head in range(self.config.n_heads):
                start = head * self.head_dim
                end = start + self.head_dim
                qh = Tensor([row[start:end] for row in q_values])
                kh = Tensor([row[start:end] for row in k_values])
                vh = Tensor([row[start:end] for row in v_values])

                scores = qh.matmul(kh.transpose()) * scale
                probs = _softmax_rows(scores, mask_causal=True)
                attended_heads.append(probs.matmul(vh))

            seq = len(token_ids)
            merged: List[List[float]] = []
            for row in range(seq):
                merged.append(
                    [
                        float(head.to_list()[row][col])
                        for head in attended_heads
                        for col in range(self.head_dim)
                    ]
                )

            attention_output = Tensor(merged).matmul(block["wo"])
            x = x + attention_output

            ff_input = _layer_norm(x)
            hidden = _relu(ff_input.matmul(block["w1"]))
            x = x + hidden.matmul(block["w2"])

        x = _layer_norm(x)
        return x.matmul(self.lm_head)

    def next_logits(self, token_ids: List[int]) -> List[float]:
        return self.forward(token_ids).to_list()[-1]

    def state_dict(self) -> dict[str, Any]:
        return {
            "token_embedding": self.token_embedding.to_list(),
            "position_embedding": self.position_embedding.to_list(),
            "blocks": [
                {name: tensor.to_list() for name, tensor in block.items()}
                for block in self.blocks
            ],
            "lm_head": self.lm_head.to_list(),
        }

    def save_checkpoint(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "type": "skeai_level2_transformer",
            "vocab_size": self.vocab_size,
            "config": asdict(self.config),
            "state": self.state_dict(),
        }
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)

    @classmethod
    def load_checkpoint(cls, path: str | Path) -> "TinyTransformerLM":
        with Path(path).open("r", encoding="utf-8") as file:
            payload = json.load(file)

        if payload.get("type") != "skeai_level2_transformer":
            raise ValueError("Unsupported Level 2 checkpoint.")

        config = TransformerConfig(**payload["config"])
        model = cls(int(payload["vocab_size"]), config)
        state = payload["state"]

        model.token_embedding = Tensor(state["token_embedding"])
        model.position_embedding = Tensor(state["position_embedding"])
        model.lm_head = Tensor(state["lm_head"])

        blocks = state.get("blocks")
        if not isinstance(blocks, list) or len(blocks) != config.n_layers:
            raise ValueError("Checkpoint block count does not match the model.")

        model.blocks = [
            {name: Tensor(values) for name, values in block.items()}
            for block in blocks
        ]
        return model


__all__ = ["TinyTransformerLM", "TransformerConfig"]
