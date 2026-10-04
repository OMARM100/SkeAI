"""SkeAI: a from-scratch lightweight AI experiment."""

__version__ = "0.1.0"

from .tensor import Tensor
from .tokenizer import CharacterTokenizer

__all__ = ["CharacterTokenizer", "Tensor"]
