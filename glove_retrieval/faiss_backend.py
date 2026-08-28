"""Top-k retrieval backed by a FAISS index.

Use this when the vectors already live in a FAISS database rather than a GloVe
text file.  :class:`FaissIndex` exposes the same ``search``/``encode``/
``most_similar`` surface as :class:`~glove_retrieval.index.GloveIndex`, so the
CLI and any calling code work unchanged against either backend.

FAISS stores only vectors and integer ids, never strings, so a *labels* sidecar
supplies the id -> word mapping.  Several common formats are accepted; see
:func:`load_labels`.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .index import METRICS, SearchResult, tokenize

try:  # pragma: no cover - exercised by the import-guard test
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

LABEL_SUFFIXES = (".labels.txt", ".labels.json", ".vocab.txt", ".vocab.json", ".txt", ".json", ".npy")


def _require_faiss():
    if faiss is None:
        raise ImportError(
            "faiss is not installed -- `pip install faiss-cpu` (or faiss-gpu) "
            "to use the FAISS backend"
        )
    return faiss


# ----------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------
def load_labels(path: str | os.PathLike[str]) -> Dict[int, str] | List[str]:
    """Load an id -> label mapping from a sidecar file.

    Accepted formats:

    * ``.txt``  -- one label per line; line number is the FAISS id
    * ``.tsv``/``.csv`` -- ``id<sep>label`` per line, or one label per line
    * ``.json`` -- a list (positional) or an object ``{"0": "the", ...}``
    * ``.npy``  -- an array of strings (positional)
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"labels file not found: {path}")

    if path.suffix == ".npy":
        return [str(x) for x in np.load(path, allow_pickle=True).ravel().tolist()]

    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            for key in ("labels", "vocab", "words", "ids", "id_to_word"):
                if key in payload and isinstance(payload[key], (list, dict)):
                    payload = payload[key]
                    break
        if isinstance(payload, dict):
            return {int(k): str(v) for k, v in payload.items()}
        return [str(x) for x in payload]

    lines = path.read_text(encoding="utf-8").splitlines()
    lines = [ln for ln in lines if ln.strip()]
    sep = "\t" if path.suffix == ".tsv" else ("," if path.suffix == ".csv" else None)
    if sep and lines and sep in lines[0]:
        mapping: Dict[int, str] = {}
        for ln in lines:
            head, _, tail = ln.partition(sep)
            try:
                mapping[int(head)] = tail.strip()
            except ValueError:  # a header row, or not an id column after all
                return [ln.strip() for ln in lines]
        return mapping
    return [ln.strip() for ln in lines]


def find_labels(index_path: str | os.PathLike[str]) -> Path | None:
    """Guess the labels sidecar sitting next to a FAISS index file."""
    index_path = Path(index_path)
    stem = index_path.with_suffix("")
    for candidate in (
        *(Path(str(stem) + s) for s in LABEL_SUFFIXES),
        *(Path(str(index_path) + s) for s in LABEL_SUFFIXES),
        index_path.parent / "labels.txt",
        index_path.parent / "vocab.txt",
        index_path.parent / "labels.json",
    ):
        if candidate.is_file() and candidate != index_path:
            return candidate
    return None


# ----------------------------------------------------------------------
# index introspection
# ----------------------------------------------------------------------
def describe(index) -> Dict[str, object]:
    """Human-readable summary of a loaded FAISS index."""
    _require_faiss()
    info: Dict[str, object] = {
        "type": type(index).__name__,
        "dim": int(index.d),
        "ntotal": int(index.ntotal),
        "metric": "inner_product" if index.metric_type == faiss.METRIC_INNER_PRODUCT else "l2",
        "trained": bool(index.is_trained),
    }
    inner = _downcast(index)
    if hasattr(inner, "nlist"):
        info["nlist"] = int(inner.nlist)
        info["nprobe"] = int(getattr(inner, "nprobe", 1))
    if hasattr(inner, "hnsw"):
        info["ef_search"] = int(inner.hnsw.efSearch)
    return info


def _downcast(index):
    """Peel IndexIDMap/IndexPreTransform wrappers to reach the real index."""
    seen = 0
    while seen < 4:
        inner = getattr(index, "index", None)
        if inner is None:
            break
        index = inner
        seen += 1
    return index


def _is_idmap(index) -> bool:
    return type(index).__name__.startswith("IndexIDMap")


def _unwrap_idmap(index):
    """Strip IndexIDMap/IndexIDMap2 wrappers, returning the storage index.

    This is not cosmetic.  ``reconstruct_n`` on an IDMap wrapper does not raise
    -- it trips a C++ assertion and calls ``abort()``, killing the interpreter
    outright, so no ``try``/``except`` can save us.  Every bulk reconstruction
    has to go through the wrapped index instead.
    """
    seen = 0
    while _is_idmap(index) and seen < 4:
        inner = getattr(index, "index", None)
        if inner is None:
            break
        index = inner
        seen += 1
    return index


def _idmap_position(index, fid: int) -> int | None:
    """Translate an external IDMap id into the storage index's row position."""
    if not _is_idmap(index):
        return None
    try:
        ids = faiss.vector_to_array(index.id_map)
    except Exception:  # noqa: BLE001
        return None
    hits = np.nonzero(ids == fid)[0]
    return int(hits[0]) if hits.size else None


def _reconstruct_sample(index, n: int = 64) -> np.ndarray | None:
    """Pull a few stored vectors, so we can tell whether they are unit-norm."""
    if index.ntotal == 0:
        return None
    target = _unwrap_idmap(index)
    inner = _downcast(target)
    if hasattr(inner, "make_direct_map"):
        try:
            inner.make_direct_map()
        except Exception:  # noqa: BLE001 - direct map is best-effort
            pass
    count = int(min(n, target.ntotal))
    for attempt in (
        lambda: target.reconstruct_n(0, count),
        lambda: np.stack([target.reconstruct(i) for i in range(count)]),
    ):
        try:
            return np.asarray(attempt(), dtype=np.float32)
        except Exception:  # noqa: BLE001 - e.g. IndexLSH cannot reconstruct
            continue
    return None


def stored_vectors_are_normalized(index, tol: float = 0.05) -> bool | None:
    """Were the vectors in this index L2-normalized before being added?

    Returns None when the index cannot reconstruct its vectors at all.

    The check is on the *median* norm rather than every norm, because quantized
    indexes (PQ, SQ) reconstruct approximately: a cosine-built ``IVF64,PQ10``
    reconstructs norms spread over roughly 0.87-1.11.  An exact-norm test would
    call that un-normalized and silently downgrade the search from cosine to a
    dot product.  Un-normalized embeddings are nowhere near this band -- GloVe
    6B/50d norms run about 2-12 -- so the median is a reliable separator.
    """
    sample = _reconstruct_sample(index)
    if sample is None or sample.size == 0:
        return None
    norms = np.linalg.norm(np.asarray(sample, dtype=np.float64), axis=1)
    norms = norms[norms > 0]
    if norms.size == 0:
        return None
    return bool(abs(float(np.median(norms)) - 1.0) <= tol)


# ----------------------------------------------------------------------
# the index
# ----------------------------------------------------------------------
class FaissIndex:
    """Query a FAISS database with the GloveIndex API.

    ``normalize`` controls whether query embeddings are L2-normalized before
    the search.  For a cosine-similarity database (unit-norm vectors in an
    inner-product index) the query must be normalized too, or the ranking is a
    dot product instead.  Left as ``None`` it is auto-detected by reconstructing
    a sample of stored vectors.
    """

    def __init__(
        self,
        index,
        labels: Sequence[str] | Dict[int, str] | None = None,
        normalize: bool | None = None,
        nprobe: int | None = None,
        source: str | None = None,
    ):
        _require_faiss()
        self.index = index
        self.labels = labels
        self.source = source
        self._word_to_id: Dict[str, int] | None = None
        self._warned_nprobe = False

        self.storage_normalized = stored_vectors_are_normalized(index)
        if normalize is None:
            if self.storage_normalized is None:
                normalize = False
                if self.is_inner_product:
                    warnings.warn(
                        f"{type(index).__name__} cannot reconstruct vectors, so it is "
                        "unknown whether they are unit-norm; querying as a raw dot "
                        "product. Pass normalize=True (--normalize) for a cosine index.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            else:
                normalize = self.storage_normalized and self.is_inner_product
        elif normalize and self.storage_normalized is False and self.is_inner_product:
            warnings.warn(
                "normalize=True but the stored vectors are not unit-norm; scores will "
                "be ||v||*cos(v, q), which ranks differently from cosine similarity. "
                "Normalize the vectors when building the index for true cosine search.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.normalize = bool(normalize)

        if nprobe is not None:
            self.nprobe = nprobe

    # -- constructors ---------------------------------------------------
    @classmethod
    def open(
        cls,
        index_path: str | os.PathLike[str],
        labels: str | os.PathLike[str] | Sequence[str] | Dict[int, str] | None = None,
        normalize: bool | None = None,
        nprobe: int | None = None,
        mmap: bool = False,
    ) -> "FaissIndex":
        """Open an existing FAISS database written by ``faiss.write_index``."""
        _require_faiss()
        index_path = Path(index_path)
        if not index_path.is_file():
            raise FileNotFoundError(f"no FAISS index at {index_path}")
        flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY if mmap else 0
        index = faiss.read_index(str(index_path), flags)

        if labels is None:
            found = find_labels(index_path)
            resolved = load_labels(found) if found else None
        elif isinstance(labels, (str, os.PathLike)):
            resolved = load_labels(labels)
        else:
            resolved = labels

        if resolved is not None:
            expected = index.ntotal
            actual = len(resolved)
            if isinstance(resolved, list) and actual < expected:
                raise ValueError(
                    f"labels file has {actual} entries but the index holds {expected} "
                    "vectors; ids beyond the labels would be unnamed"
                )
        return cls(index, resolved, normalize=normalize, nprobe=nprobe, source=str(index_path))

    @classmethod
    def build(
        cls,
        vocab: Sequence[str],
        vectors: np.ndarray,
        metric: str = "cosine",
        factory: str = "Flat",
        train_size: int | None = None,
    ) -> "FaissIndex":
        """Build a FAISS index from vectors (e.g. parsed GloVe 6B/50d).

        ``metric='cosine'`` L2-normalizes a copy of the vectors and uses an
        inner-product index, which is the standard cosine setup.
        """
        _require_faiss()
        if metric not in ("cosine", "dot", "euclidean"):
            raise ValueError(f"unknown metric {metric!r}; expected one of {METRICS}")

        matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        if len(vocab) != matrix.shape[0]:
            raise ValueError(
                f"vocab has {len(vocab)} entries but matrix has {matrix.shape[0]} rows"
            )
        if metric == "cosine":
            matrix = matrix.copy()
            faiss.normalize_L2(matrix)
        faiss_metric = (
            faiss.METRIC_L2 if metric == "euclidean" else faiss.METRIC_INNER_PRODUCT
        )
        index = faiss.index_factory(int(matrix.shape[1]), factory, faiss_metric)
        if not index.is_trained:
            sample = matrix
            if train_size and train_size < matrix.shape[0]:
                rows = np.random.default_rng(0).choice(matrix.shape[0], train_size, replace=False)
                sample = matrix[np.sort(rows)]
            index.train(sample)
        index.add(matrix)
        return cls(index, list(vocab), normalize=(metric == "cosine"))

    def save(
        self,
        index_path: str | os.PathLike[str],
        labels_path: str | os.PathLike[str] | None = None,
    ) -> Path:
        _require_faiss()
        index_path = Path(index_path)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        if self.labels is not None:
            if labels_path is None:
                labels_path = index_path.with_suffix(index_path.suffix + ".labels.txt")
            labels_path = Path(labels_path)
            if isinstance(self.labels, dict):
                body = "\n".join(f"{k}\t{v}" for k, v in sorted(self.labels.items()))
                labels_path = labels_path.with_suffix(".tsv")
            else:
                body = "\n".join(self.labels)
            labels_path.write_text(body + "\n", encoding="utf-8")
        return index_path

    # -- properties -----------------------------------------------------
    @property
    def dim(self) -> int:
        return int(self.index.d)

    @property
    def is_inner_product(self) -> bool:
        return self.index.metric_type == faiss.METRIC_INNER_PRODUCT

    @property
    def metric(self) -> str:
        """The similarity this index actually computes.

        Only an inner-product index over unit-norm vectors *with* a normalized
        query is cosine; everything else on an IP index ranks by a dot product
        (scaled by ``||v||`` if only the query was normalized).
        """
        if not self.is_inner_product:
            return "euclidean"
        if self.normalize and self.storage_normalized is not False:
            return "cosine"
        return "dot"

    @property
    def nprobe(self) -> int | None:
        inner = _downcast(self.index)
        return int(inner.nprobe) if hasattr(inner, "nprobe") else None

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        inner = _downcast(self.index)
        if not hasattr(inner, "nprobe"):
            raise AttributeError(f"{type(inner).__name__} has no nprobe (it is not an IVF index)")
        inner.nprobe = int(value)
        self._warned_nprobe = False

    def __len__(self) -> int:
        return int(self.index.ntotal)

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and word.lower() in self._vocab_map()

    def describe(self) -> Dict[str, object]:
        info = describe(self.index)
        info.update(
            {
                "source": self.source,
                "labels": self.labels is not None,
                "normalize_query": self.normalize,
                "storage_normalized": self.storage_normalized,
                "effective_metric": self.metric,
            }
        )
        return info

    # -- labels ---------------------------------------------------------
    def label(self, fid: int) -> str:
        if self.labels is None:
            return str(fid)
        if isinstance(self.labels, dict):
            return self.labels.get(int(fid), str(fid))
        return self.labels[int(fid)] if 0 <= fid < len(self.labels) else str(fid)

    def _vocab_map(self) -> Dict[str, int]:
        if self._word_to_id is None:
            if self.labels is None:
                self._word_to_id = {}
            elif isinstance(self.labels, dict):
                self._word_to_id = {str(v).lower(): int(k) for k, v in self.labels.items()}
            else:
                self._word_to_id = {w.lower(): i for i, w in enumerate(self.labels)}
        return self._word_to_id

    def id_of(self, word: str) -> int:
        try:
            return self._vocab_map()[word.lower()]
        except KeyError:
            hint = "" if self.labels is not None else " (no labels file loaded)"
            raise KeyError(f"{word!r} is not in the vocabulary{hint}") from None

    def vector(self, word: str) -> np.ndarray:
        """Reconstruct a stored vector by word.

        Note: for a cosine index the reconstructed vector is unit-norm, since
        that is what was added to FAISS.
        """
        fid = int(self.id_of(word))
        # reconstruct(id) is safe on every index type (unlike reconstruct_n),
        # but IVF needs a direct map and IndexIDMap v1 has no by-id support.
        for attempt in (
            lambda: self.index.reconstruct(fid),
            lambda: (self._make_direct_map(), self.index.reconstruct(fid))[1],
            lambda: self._reconstruct_via_idmap(fid),
        ):
            try:
                vector = attempt()
            except Exception:  # noqa: BLE001 - try the next strategy
                continue
            if vector is not None:
                return np.asarray(vector, dtype=np.float32)
        raise RuntimeError(
            f"{type(self.index).__name__} cannot reconstruct vectors; "
            "pass query embeddings directly instead of looking words up"
        )

    def _make_direct_map(self) -> None:
        inner = _downcast(_unwrap_idmap(self.index))
        if hasattr(inner, "make_direct_map"):
            inner.make_direct_map()

    def _reconstruct_via_idmap(self, fid: int):
        """IndexIDMap (v1) cannot reconstruct by id; go through the row position."""
        position = _idmap_position(self.index, fid)
        if position is None:
            return None
        return _unwrap_idmap(self.index).reconstruct(int(position))

    def encode(self, text: str, strict: bool = False) -> np.ndarray:
        """Embed a word or phrase as the mean of its in-vocabulary tokens."""
        tokens = tokenize(text)
        if not tokens:
            raise ValueError(f"no tokens in query {text!r}")
        vocab = self._vocab_map()
        known = [t for t in tokens if t in vocab]
        missing = [t for t in tokens if t not in vocab]
        if strict and missing:
            raise KeyError(f"out-of-vocabulary tokens: {', '.join(missing)}")
        if not known:
            raise KeyError(f"none of {tokens} are in the vocabulary")
        return np.mean([self.vector(t) for t in known], axis=0).astype(np.float32)

    # -- search ---------------------------------------------------------
    def as_query(self, query: str | Sequence[float] | np.ndarray) -> np.ndarray:
        """Coerce a query to a contiguous ``(n, dim)`` float32 batch."""
        if isinstance(query, str):
            query = self.encode(query)
        batch = np.asarray(query, dtype=np.float32)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)
        if batch.ndim != 2:
            raise ValueError(f"query must be 1-D or 2-D, got shape {batch.shape}")
        if batch.shape[1] != self.dim:
            raise ValueError(
                f"query embedding has {batch.shape[1]} dimensions, index has {self.dim}"
            )
        batch = np.ascontiguousarray(batch)
        if self.normalize:
            batch = batch.copy()
            faiss.normalize_L2(batch)
        return batch

    def search(
        self,
        query: str | Sequence[float] | np.ndarray,
        k: int = 10,
        exclude: Iterable[str] = (),
        nprobe: int | None = None,
    ) -> List[SearchResult]:
        """Return the ``k`` nearest entries in the FAISS database."""
        if k <= 0:
            raise ValueError("k must be positive")
        if nprobe is not None:
            self.nprobe = nprobe
        self._warn_if_underprobing()

        batch = self.as_query(query)
        if batch.shape[0] != 1:
            raise ValueError("search() takes a single query; use search_batch() for many")

        excluded_ids = {self.id_of(w) for w in exclude if w.lower() in self._vocab_map()}
        # Over-fetch so that excluded hits do not shrink the result set.
        want = min(k + len(excluded_ids), len(self))
        distances, ids = self.index.search(batch, max(want, 1))

        results: List[SearchResult] = []
        for dist, fid in zip(distances[0], ids[0]):
            fid = int(fid)
            if fid < 0 or fid in excluded_ids:  # -1 pads a short result set
                continue
            # L2 indexes return squared distances; negate so higher is better.
            score = -float(np.sqrt(max(float(dist), 0.0))) if not self.is_inner_product else float(dist)
            results.append(SearchResult(rank=len(results) + 1, word=self.label(fid), score=score, index=fid))
            if len(results) == k:
                break
        return results

    def search_batch(
        self, queries: np.ndarray, k: int = 10, nprobe: int | None = None
    ) -> List[List[SearchResult]]:
        """Search many query embeddings at once (one FAISS call)."""
        if nprobe is not None:
            self.nprobe = nprobe
        self._warn_if_underprobing()
        batch = self.as_query(queries)
        distances, ids = self.index.search(batch, k)
        out: List[List[SearchResult]] = []
        for row_d, row_i in zip(distances, ids):
            row: List[SearchResult] = []
            for dist, fid in zip(row_d, row_i):
                fid = int(fid)
                if fid < 0:
                    continue
                score = (
                    float(dist)
                    if self.is_inner_product
                    else -float(np.sqrt(max(float(dist), 0.0)))
                )
                row.append(
                    SearchResult(rank=len(row) + 1, word=self.label(fid), score=score, index=fid)
                )
            out.append(row)
        return out

    def most_similar(self, word: str, k: int = 10) -> List[SearchResult]:
        return self.search(self.vector(word), k=k, exclude=[word])

    def analogy(
        self, positive: Sequence[str], negative: Sequence[str] = (), k: int = 10
    ) -> List[SearchResult]:
        vector = np.zeros(self.dim, dtype=np.float32)
        for word in positive:
            vector += self.vector(word)
        for word in negative:
            vector -= self.vector(word)
        return self.search(vector, k=k, exclude=[*positive, *negative])

    def _warn_if_underprobing(self) -> None:
        """nprobe=1 on an IVF index silently returns poor recall. Say so once."""
        if self._warned_nprobe:
            return
        inner = _downcast(self.index)
        if hasattr(inner, "nprobe") and int(inner.nprobe) <= 1 and int(getattr(inner, "nlist", 1)) > 1:
            self._warned_nprobe = True
            print(
                f"note: IVF index with nprobe=1 over {inner.nlist} lists scans ~"
                f"{100.0 / inner.nlist:.1f}% of the data; pass --nprobe to trade "
                "speed for recall",
                file=sys.stderr,
            )
