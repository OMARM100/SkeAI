"""Medium-sized causal Transformer language model for SkeAI Level 2.

The model remains fully from-scratch and keeps the native C++ training path,
but the default configuration is now large enough to learn useful language
representations and longer conversational context. It uses token embeddings,
learned positions, causal multi-head self-attention, pre-normalized residual
blocks, feed-forward layers, and a final language-model projection.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List

from .. import engine
from ..tensor import Tensor


@dataclass(frozen=True)
class TransformerConfig:
    context_length: int = 128
    d_model: int = 128
    n_heads: int = 4
    feed_forward_size: int = 512
    n_layers: int = 4
    max_vocab_size: int = 2048
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

def _zeros(rows: int, cols: int) -> list[list[float]]:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def _transpose(values: list[list[float]]) -> list[list[float]]:
    if not values:
        return []
    return [list(column) for column in zip(*values)]


def _matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    if not a or not b:
        raise ValueError("matmul requires non-empty matrices.")
    if len(a[0]) != len(b):
        raise ValueError("Incompatible matrix shapes.")

    left = Tensor(a)
    right = Tensor(b)
    return left.matmul(right).to_list()


def _add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [
        [x + y for x, y in zip(arow, brow)]
        for arow, brow in zip(a, b)
    ]


def _relu_backward(dy: list[list[float]], x: list[list[float]]) -> list[list[float]]:
    return [
        [dy_value if x_value > 0.0 else 0.0 for dy_value, x_value in zip(drow, xrow)]
        for drow, xrow in zip(dy, x)
    ]


def _layer_norm_forward(
    values: list[list[float]],
    eps: float = 1e-5,
) -> tuple[list[list[float]], list[float], list[float]]:
    normalized: list[list[float]] = []
    means: list[float] = []
    inv_stds: list[float] = []
    for row in values:
        mean = sum(row) / len(row)
        variance = sum((value - mean) ** 2 for value in row) / len(row)
        inv_std = 1.0 / math.sqrt(variance + eps)
        means.append(mean)
        inv_stds.append(inv_std)
        normalized.append([(value - mean) * inv_std for value in row])
    return normalized, means, inv_stds


def _layer_norm_backward(
    dy: list[list[float]],
    x: list[list[float]],
    means: list[float],
    inv_stds: list[float],
) -> list[list[float]]:
    width = len(x[0])
    dx: list[list[float]] = []
    for row_index, (dy_row, x_row) in enumerate(zip(dy, x)):
        x_centered = [value - means[row_index] for value in x_row]
        x_hat = [value * inv_stds[row_index] for value in x_centered]
        sum_dy = sum(dy_row)
        sum_dy_xhat = sum(
            dy_value * xhat_value
            for dy_value, xhat_value in zip(dy_row, x_hat)
        )
        dx.append([
            (inv_stds[row_index] / width)
            * (
                width * dy_value
                - sum_dy
                - xhat_value * sum_dy_xhat
            )
            for dy_value, xhat_value in zip(dy_row, x_hat)
        ])
    return dx


def _softmax_backward(
    probabilities: list[list[float]],
    dprobabilities: list[list[float]],
) -> list[list[float]]:
    if not probabilities or not dprobabilities:
        raise ValueError("softmax backward requires non-empty matrices.")
    if len(probabilities) != len(dprobabilities):
        raise ValueError("softmax backward row counts must match.")
    if len(probabilities[0]) != len(dprobabilities[0]):
        raise ValueError("softmax backward column counts must match.")

    probability_tensor = Tensor(probabilities)
    gradient_tensor = Tensor(dprobabilities)
    storage = engine.softmax_backward(
        probability_tensor._storage,
        gradient_tensor._storage,
        len(probabilities),
        len(probabilities[0]),
    )
    return Tensor._from_storage(
        storage,
        (len(probabilities), len(probabilities[0])),
    ).to_list()


def _causal_softmax(values: list[list[float]]) -> list[list[float]]:
    if not values:
        raise ValueError("causal softmax requires non-empty input.")
    columns = len(values[0])
    if columns <= 0 or any(len(row) != columns for row in values):
        raise ValueError("causal softmax input must be rectangular.")

    tensor = Tensor(values)
    storage = engine.causal_softmax(
        tensor._storage,
        len(values),
        columns,
    )
    return Tensor._from_storage(
        storage,
        (len(values), columns),
    ).to_list()


def _outer_accumulate(
    target: list[list[float]],
    left: list[list[float]],
    right: list[list[float]],
) -> None:
    for row, left_row in enumerate(left):
        for col, right_row in enumerate(right):
            target[row][col] += sum(
                x * y for x, y in zip(left_row, right_row)
            )
        

class TinyTransformerLM:
    """Medium causal Transformer retained under the stable Level 2 class name."""

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
        self._last_cache: dict[str, Any] | None = None


    def parameters(self) -> dict[str, Tensor]:
        parameters: dict[str, Tensor] = {
            "token_embedding": self.token_embedding,
            "position_embedding": self.position_embedding,
            "lm_head": self.lm_head,
        }
        for index, block in enumerate(self.blocks):
            for name, tensor in block.items():
                parameters[f"blocks.{index}.{name}"] = tensor
        return parameters

    def gradients(self) -> dict[str, Tensor]:
        return {
            name: Tensor.zeros(parameter.shape)
            for name, parameter in self.parameters().items()
        }

    def _forward_cached(self, token_ids: List[int]) -> Tensor:
        if not token_ids:
            raise ValueError("token_ids cannot be empty.")
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in token_ids):
            raise ValueError("Token ID out of range.")

        if len(token_ids) > self.config.context_length:
            token_ids = token_ids[-self.config.context_length:]

        d = self.config.d_model
        token_rows = self.token_embedding.to_list()
        position_rows = self.position_embedding.to_list()

        x = [
            [
                float(token_rows[token_id][col]) + float(position_rows[position][col])
                for col in range(d)
            ]
            for position, token_id in enumerate(token_ids)
        ]

        cache: dict[str, Any] = {
            "token_ids": list(token_ids),
            "embeddings": [row[:] for row in x],
            "blocks": [],
        }
        scale = 1.0 / math.sqrt(self.head_dim)

        for block in self.blocks:
            wq = block["wq"].to_list()
            wk = block["wk"].to_list()
            wv = block["wv"].to_list()
            wo = block["wo"].to_list()
            w1 = block["w1"].to_list()
            w2 = block["w2"].to_list()

            norm_x, mean1, inv1 = _layer_norm_forward(x)
            q = _matmul(norm_x, wq)
            k = _matmul(norm_x, wk)
            v = _matmul(norm_x, wv)

            head_caches = []
            merged = _zeros(len(token_ids), d)

            for head in range(self.config.n_heads):
                start = head * self.head_dim
                end = start + self.head_dim
                qh = [row[start:end] for row in q]
                kh = [row[start:end] for row in k]
                vh = [row[start:end] for row in v]

                # Q @ K^T gives one score for every query/key pair.
                scores = _matmul(qh, _transpose(kh))
                scores = [
                    [value * scale for value in row]
                    for row in scores
                ]
                probs = _causal_softmax(scores)
                attended = _matmul(probs, vh)

                for row_index, attended_row in enumerate(attended):
                    merged[row_index][start:end] = attended_row

                head_caches.append({
                    "qh": qh,
                    "kh": kh,
                    "vh": vh,
                    "scores": scores,
                    "probs": probs,
                })

            attention_output = _matmul(merged, wo)
            residual = _add(x, attention_output)

            norm_residual, mean2, inv2 = _layer_norm_forward(residual)
            hidden_pre = _matmul(norm_residual, w1)
            hidden = [[max(0.0, value) for value in row] for row in hidden_pre]
            feed_forward = _matmul(hidden, w2)
            next_x = _add(residual, feed_forward)

            cache["blocks"].append({
                "input": [row[:] for row in x],
                "norm_x": norm_x,
                "mean1": mean1,
                "inv1": inv1,
                "q": q,
                "k": k,
                "v": v,
                "heads": head_caches,
                "merged": merged,
                "attention_output": attention_output,
                "residual": residual,
                "norm_residual": norm_residual,
                "mean2": mean2,
                "inv2": inv2,
                "hidden_pre": hidden_pre,
                "hidden": hidden,
                "next_x": next_x,
            })
            x = next_x

        final_norm, final_mean, final_inv = _layer_norm_forward(x)
        lm_head = self.lm_head.to_list()
        logits = _matmul(final_norm, lm_head)

        cache["final_input"] = x
        cache["final_norm"] = final_norm
        cache["final_mean"] = final_mean
        cache["final_inv"] = final_inv
        self._last_cache = cache
        return Tensor(logits)

    def forward(self, token_ids: List[int]) -> Tensor:
        return self._forward_cached(token_ids)

    def backward(
        self,
        dlogits: Tensor,
        gradients: dict[str, Tensor] | None = None,
    ) -> dict[str, Tensor]:
        if self._last_cache is None:
            raise RuntimeError("forward must be called before backward.")

        cache = self._last_cache
        grads = gradients or self.gradients()
        scale = 1.0 / math.sqrt(self.head_dim)
        dlogits_values = dlogits.to_list()
        final_norm = cache["final_norm"]
        lm_head = self.lm_head.to_list()

        grad_lm_head = _matmul(_transpose(final_norm), dlogits_values)
        grad_final_norm = _matmul(dlogits_values, _transpose(lm_head))
        grad_x = _layer_norm_backward(
            grad_final_norm,
            cache["final_input"],
            cache["final_mean"],
            cache["final_inv"],
        )

        block_grads = [
            {name: _zeros(*tensor.shape) for name, tensor in block.items()}
            for block in self.blocks
        ]

        for block_index in range(len(self.blocks) - 1, -1, -1):
            block = self.blocks[block_index]
            block_cache = cache["blocks"][block_index]
            wq = block["wq"].to_list()
            wk = block["wk"].to_list()
            wv = block["wv"].to_list()
            wo = block["wo"].to_list()
            w1 = block["w1"].to_list()
            w2 = block["w2"].to_list()

            d_next = grad_x
            d_residual = [row[:] for row in d_next]

            hidden = block_cache["hidden"]
            norm_residual = block_cache["norm_residual"]

            d_hidden = _matmul(d_next, _transpose(w2))
            d_w2 = _matmul(_transpose(hidden), d_next)
            d_hidden_pre = _relu_backward(d_hidden, block_cache["hidden_pre"])
            d_norm_residual = _matmul(d_hidden_pre, _transpose(w1))
            d_w1 = _matmul(_transpose(norm_residual), d_hidden_pre)

            d_residual = _add(
                d_residual,
                _layer_norm_backward(
                    d_norm_residual,
                    block_cache["residual"],
                    block_cache["mean2"],
                    block_cache["inv2"],
                ),
            )

            d_attention_output = d_residual
            d_merged = _matmul(d_attention_output, _transpose(wo))
            d_wo = _matmul(_transpose(block_cache["merged"]), d_attention_output)

            q = block_cache["q"]
            k = block_cache["k"]
            v = block_cache["v"]
            d_q = _zeros(len(q), len(q[0]))
            d_k = _zeros(len(k), len(k[0]))
            d_v = _zeros(len(v), len(v[0]))
            d_merged_heads = d_merged

            for head_index, head_cache in enumerate(block_cache["heads"]):
                start = head_index * self.head_dim
                end = start + self.head_dim
                dq_head = _zeros(len(q), self.head_dim)
                dk_head = _zeros(len(k), self.head_dim)
                dv_head = _zeros(len(v), self.head_dim)

                d_attended = [
                    row[start:end]
                    for row in d_merged_heads
                ]
                probs = head_cache["probs"]
                vh = head_cache["vh"]
                qh = head_cache["qh"]
                kh = head_cache["kh"]

                d_probs = _matmul(d_attended, _transpose(vh))
                d_vh = _matmul(_transpose(probs), d_attended)
                d_scores = _softmax_backward(probs, d_probs)

                d_qh = _matmul(d_scores, kh)
                d_kh = _matmul(_transpose(d_scores), qh)

                for row in range(len(d_qh)):
                    for col in range(self.head_dim):
                        dq_head[row][col] = d_qh[row][col] * scale
                        dk_head[row][col] = d_kh[row][col] * scale
                        dv_head[row][col] = d_vh[row][col]

                for row in range(len(d_q)):
                    d_q[row][start:end] = dq_head[row]
                    d_k[row][start:end] = dk_head[row]
                    d_v[row][start:end] = dv_head[row]

            d_norm_x = _add(
                _matmul(d_q, _transpose(wq)),
                _add(
                    _matmul(d_k, _transpose(wk)),
                    _matmul(d_v, _transpose(wv)),
                ),
            )

            d_wq = _matmul(_transpose(block_cache["norm_x"]), d_q)
            d_wk = _matmul(_transpose(block_cache["norm_x"]), d_k)
            d_wv = _matmul(_transpose(block_cache["norm_x"]), d_v)

            d_input = _add(
                d_residual,
                _layer_norm_backward(
                    d_norm_x,
                    block_cache["input"],
                    block_cache["mean1"],
                    block_cache["inv1"],
                ),
            )

            grad_x = d_input
            block_grads[block_index]["wq"] = d_wq
            block_grads[block_index]["wk"] = d_wk
            block_grads[block_index]["wv"] = d_wv
            block_grads[block_index]["wo"] = d_wo
            block_grads[block_index]["w1"] = d_w1
            block_grads[block_index]["w2"] = d_w2

        token_grad = _zeros(self.vocab_size, self.config.d_model)
        position_grad = _zeros(self.config.context_length, self.config.d_model)
        for row_index, token_id in enumerate(cache["token_ids"]):
            for col, value in enumerate(grad_x[row_index]):
                token_grad[token_id][col] += value
                position_grad[row_index][col] += value

        grads["token_embedding"] = Tensor(token_grad)
        grads["position_embedding"] = Tensor(position_grad)
        grads["lm_head"] = Tensor(grad_lm_head)

        for index, block_grad in enumerate(block_grads):
            for name, values in block_grad.items():
                grads[f"blocks.{index}.{name}"] = Tensor(values)

        return grads

    def generate(
        self,
        token_ids: list[int],
        *,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_k: int = 0,
        seed: int | None = None,
        eos_token_id: int | None = None,
    ) -> list[int]:
        """Generate token IDs autoregressively from an existing token prefix."""
        if not token_ids:
            raise ValueError("token_ids cannot be empty.")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens cannot be negative.")
        if temperature < 0.0:
            raise ValueError("temperature cannot be negative.")
        if top_k < 0:
            raise ValueError("top_k cannot be negative.")
        if eos_token_id is not None and not 0 <= eos_token_id < self.vocab_size:
            raise ValueError("eos_token_id is out of range.")

        generated = list(token_ids)
        if any(token_id < 0 or token_id >= self.vocab_size for token_id in generated):
            raise ValueError("Token ID out of range.")

        rng = random.Random(seed)

        for _ in range(max_new_tokens):
            logits = self.next_logits(generated)
            if temperature == 0.0:
                next_token = max(
                    range(self.vocab_size),
                    key=lambda index: logits[index],
                )
            else:
                scale = 1.0 / temperature
                candidates = list(range(self.vocab_size))
                if top_k:
                    candidates = sorted(
                        candidates,
                        key=lambda index: logits[index],
                        reverse=True,
                    )[:min(top_k, self.vocab_size)]

                maximum = max(logits[index] * scale for index in candidates)
                weights = [
                    math.exp(logits[index] * scale - maximum)
                    for index in candidates
                ]
                next_token = rng.choices(candidates, weights=weights, k=1)[0]

            generated.append(next_token)

            if eos_token_id is not None and next_token == eos_token_id:
                break

        return generated

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
