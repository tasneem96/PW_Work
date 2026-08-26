"""Top-k retrieval over GloVe word embeddings (6B / 50d and friends)."""

from .index import GloveIndex, SearchResult, tokenize
from .loader import load_glove_text

__all__ = ["GloveIndex", "SearchResult", "tokenize", "load_glove_text"]
__version__ = "0.1.0"
