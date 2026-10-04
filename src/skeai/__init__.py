"""SkeAI: a from-scratch lightweight AI experiment."""

__version__ = "0.1.0"

from .layers import Dense, ReLU, Tanh
from .loss import CrossEntropyLoss, MeanSquaredError
from .optimizer import SGD
from .tensor import Tensor
from .tokenizer import CharacterTokenizer

__all__ = [
    "CharacterTokenizer",
    "CrossEntropyLoss",
    "Dense",
    "MeanSquaredError",
    "ReLU",
    "SGD",
    "Tensor",
    "Tanh",
]
