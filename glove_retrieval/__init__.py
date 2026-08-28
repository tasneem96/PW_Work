"""Top-k retrieval over GloVe word embeddings (6B / 50d and friends).

Two interchangeable backends:

* :class:`GloveIndex`  -- exact brute-force scan over a NumPy matrix
* :class:`FaissIndex`  -- an existing (or freshly built) FAISS database

``FaissIndex`` is imported lazily so that ``faiss`` stays an optional
dependency for anyone using the NumPy backend.
"""

from .index import GloveIndex, SearchResult, tokenize
from .loader import load_glove_text

__all__ = ["GloveIndex", "FaissIndex", "SearchResult", "tokenize", "load_glove_text"]
__version__ = "0.2.0"


def __getattr__(name: str):
    if name == "FaissIndex":
        from .faiss_backend import FaissIndex

        return FaissIndex
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
