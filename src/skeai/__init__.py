"""SkeAI: a from-scratch lightweight AI experiment."""

__version__ = "0.2.0"

from .benchmark import BenchmarkResult, benchmark
from .layers import Dense, ReLU, Tanh
from .dataset import CharacterLanguageDataset
from .language_model import TinyCharacterLanguageModel
from .loss import CrossEntropyLoss, MeanSquaredError
from .generation import generate_text
from .model import Sequential
from .trainer import Trainer
from .optimizer import SGD
from .tensor import Tensor
from .tokenizer import CharacterTokenizer

__all__ = [
    "BenchmarkResult",
    "CharacterLanguageDataset",
    "CharacterTokenizer",
    "CrossEntropyLoss",
    "Dense",
    "TinyCharacterLanguageModel",
    "MeanSquaredError",
    "ReLU",
    "SGD",
    "Sequential",
    "Tensor",
    "Tanh",
    "Trainer",
    "benchmark",
    "generate_text",
]
