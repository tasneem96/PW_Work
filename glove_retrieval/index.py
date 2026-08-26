"""Top-k nearest-neighbour retrieval over a GloVe embedding matrix."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from . import loader

Metric = str
METRICS = ("cosine", "dot", "euclidean")

_TOKEN_RE = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True)
class SearchResult:
    """One retrieved neighbour."""

    rank: int
    word: str
    score: float
    index: int

    def __str__(self) -> str:  # pragma: no cover - display helper
        return f"{self.rank:>3}. {self.word:<24} {self.score:+.4f}"


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer matching GloVe 6B's uncased vocabulary."""
    return _TOKEN_RE.findall(text.lower())


class GloveIndex:
    """An in-memory GloVe index supporting exact top-k search.

    Search is a single dense mat-vec product.  For GloVe 6B/50d
    (400k x 50 float32, 80 MB) an exact scan takes a few milliseconds, so
    there is no need for an approximate index.
    """

    def __init__(self, vocab: Sequence[str], vectors: np.ndarray):
        vectors = np.asarray(vectors)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be 2-D, got shape {vectors.shape}")
        if len(vocab) != vectors.shape[0]:
            raise ValueError(
                f"vocab has {len(vocab)} entries but matrix has {vectors.shape[0]} rows"
            )
        self.vocab: List[str] = list(vocab)
        self.vectors = vectors
        self._word_to_index: Dict[str, int] = {w: i for i, w in enumerate(self.vocab)}
        self._unit: np.ndarray | None = None
        self._sq_norms: np.ndarray | None = None

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_text(
        cls, path: str | os.PathLike[str], limit: int | None = None
    ) -> "GloveIndex":
        vocab, matrix = loader.load_glove_text(path, limit=limit)
        return cls(vocab, matrix)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        limit: int | None = None,
        cache: bool = True,
        cache_dir: str | os.PathLike[str] | None = None,
    ) -> "GloveIndex":
        """Load vectors from ``path``, using (and populating) a ``.npy`` cache.

        ``path`` may be a GloVe text file (optionally ``.gz``/``.zip``) or a
        cache directory previously written by :meth:`save`.
        """
        path = Path(path)
        if path.is_dir():
            vocab, matrix = loader.read_cache(path)
            return cls(vocab, matrix)

        if not path.is_file():
            raise FileNotFoundError(
                f"no GloVe vectors at {path} -- run scripts/download_glove.sh "
                "or pass --vectors /path/to/glove.6B.50d.txt"
            )

        if not cache:
            return cls.from_text(path, limit=limit)

        cache_dir = Path(cache_dir) if cache_dir else loader.default_cache_dir(path, limit)
        source = loader.GloveSource.of(path, limit)
        if loader.cache_is_valid(cache_dir, source):
            vocab, matrix = loader.read_cache(cache_dir)
            return cls(vocab, matrix)

        vocab, matrix = loader.load_glove_text(path, limit=limit)
        try:
            loader.write_cache(cache_dir, vocab, matrix, source)
        except OSError:
            pass  # a read-only data dir is not a reason to fail the query
        return cls(vocab, matrix)

    def save(self, cache_dir: str | os.PathLike[str]) -> Path:
        source = loader.GloveSource(path="<in-memory>", size=0, mtime=0, limit=None)
        return loader.write_cache(cache_dir, self.vocab, np.asarray(self.vectors), source)

    # ------------------------------------------------------------------
    # basic accessors
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.vectors.shape[1])

    def __len__(self) -> int:
        return len(self.vocab)

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and word.lower() in self._word_to_index

    def index_of(self, word: str) -> int:
        try:
            return self._word_to_index[word.lower()]
        except KeyError:
            raise KeyError(f"{word!r} is not in the vocabulary") from None

    def vector(self, word: str) -> np.ndarray:
        return np.asarray(self.vectors[self.index_of(word)], dtype=np.float32)

    # ------------------------------------------------------------------
    # query construction
    # ------------------------------------------------------------------
    def encode(self, text: str, strict: bool = False) -> np.ndarray:
        """Embed a word or phrase as the mean of its in-vocabulary tokens."""
        tokens = tokenize(text)
        if not tokens:
            raise ValueError(f"no tokens in query {text!r}")
        known = [t for t in tokens if t in self._word_to_index]
        missing = [t for t in tokens if t not in self._word_to_index]
        if strict and missing:
            raise KeyError(f"out-of-vocabulary tokens: {', '.join(missing)}")
        if not known:
            raise KeyError(f"none of {tokens} are in the vocabulary")
        rows = np.asarray(self.vectors[[self._word_to_index[t] for t in known]], dtype=np.float32)
        return rows.mean(axis=0)

    def as_query(self, query: str | Sequence[float] | np.ndarray) -> np.ndarray:
        """Coerce a query -- text or a raw ``dim``-length vector -- to float32."""
        if isinstance(query, str):
            return self.encode(query)
        vector = np.asarray(query, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"query embedding has {vector.shape[0]} dimensions, index has {self.dim}"
            )
        return vector

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------
    def _unit_vectors(self) -> np.ndarray:
        if self._unit is None:
            matrix = np.asarray(self.vectors, dtype=np.float32)
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            np.maximum(norms, 1e-12, out=norms)
            self._unit = matrix / norms
        return self._unit

    def _squared_norms(self) -> np.ndarray:
        if self._sq_norms is None:
            matrix = np.asarray(self.vectors, dtype=np.float32)
            self._sq_norms = np.einsum("ij,ij->i", matrix, matrix)
        return self._sq_norms

    def scores(
        self, query: str | Sequence[float] | np.ndarray, metric: Metric = "cosine"
    ) -> np.ndarray:
        """Similarity of every vocabulary entry to ``query`` (higher is better)."""
        vector = self.as_query(query)
        if metric == "cosine":
            norm = float(np.linalg.norm(vector))
            if norm == 0.0:
                raise ValueError("cosine similarity is undefined for a zero query vector")
            return self._unit_vectors() @ (vector / norm)
        if metric == "dot":
            return np.asarray(self.vectors, dtype=np.float32) @ vector
        if metric == "euclidean":
            # Negated distance, so that "higher is better" holds for every metric.
            dots = np.asarray(self.vectors, dtype=np.float32) @ vector
            sq = self._squared_norms() - 2.0 * dots + float(vector @ vector)
            return -np.sqrt(np.maximum(sq, 0.0))
        raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")

    def search(
        self,
        query: str | Sequence[float] | np.ndarray,
        k: int = 10,
        metric: Metric = "cosine",
        exclude: Iterable[str] = (),
    ) -> List[SearchResult]:
        """Return the ``k`` nearest vocabulary entries to ``query``."""
        if k <= 0:
            raise ValueError("k must be positive")
        scores = np.asarray(self.scores(query, metric=metric), dtype=np.float32)

        excluded = {w.lower() for w in exclude}
        if excluded:
            rows = [self._word_to_index[w] for w in excluded if w in self._word_to_index]
            if rows:
                scores = scores.copy()
                scores[rows] = -np.inf

        k = min(k, len(self.vocab))
        # argpartition is O(n); a full sort of 400k scores would dominate runtime.
        top = np.argpartition(-scores, k - 1)[:k] if k < len(scores) else np.arange(len(scores))
        top = top[np.argsort(-scores[top], kind="stable")]
        return [
            SearchResult(rank=i + 1, word=self.vocab[int(j)], score=float(scores[j]), index=int(j))
            for i, j in enumerate(top)
        ]

    def most_similar(
        self, word: str, k: int = 10, metric: Metric = "cosine"
    ) -> List[SearchResult]:
        """Neighbours of ``word``, excluding the query word itself."""
        return self.search(self.vector(word), k=k, metric=metric, exclude=[word])

    def analogy(
        self, positive: Sequence[str], negative: Sequence[str] = (), k: int = 10
    ) -> List[SearchResult]:
        """``king - man + woman``-style queries."""
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in positive:
            vector += self.vector(word)
        for word in negative:
            vector -= self.vector(word)
        return self.search(vector, k=k, exclude=[*positive, *negative])
