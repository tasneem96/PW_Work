"""Loading GloVe vectors from text files, with a fast on-disk cache.

The canonical distribution (``glove.6B.zip`` from Stanford NLP) ships plain
text files -- one token per line, followed by ``dim`` space separated floats::

    the 0.418 0.24968 -0.41242 ...

Parsing 400k lines of text takes ~20s and a lot of transient memory, so the
first load is cached as a ``.npy`` matrix plus a vocabulary file.  Subsequent
loads are memory-mapped and effectively instant.
"""

from __future__ import annotations

import gzip
import io
import json
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Iterator, List, Sequence, Tuple

import numpy as np

CACHE_VERSION = 1
VECTORS_FILE = "vectors.npy"
VOCAB_FILE = "vocab.txt"
META_FILE = "meta.json"


@dataclass(frozen=True)
class GloveSource:
    """Where a set of vectors came from, for cache invalidation and display."""

    path: str
    size: int
    mtime: int
    limit: int | None

    @classmethod
    def of(cls, path: str | os.PathLike[str], limit: int | None) -> "GloveSource":
        stat = os.stat(path)
        return cls(str(Path(path).resolve()), stat.st_size, int(stat.st_mtime), limit)


def _open_text(path: Path) -> IO[str]:
    """Open a GloVe file, transparently handling ``.gz`` and ``.zip``."""
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    if path.suffix == ".zip":
        archive = zipfile.ZipFile(path)
        names = [n for n in archive.namelist() if n.endswith(".txt")]
        if len(names) != 1:
            raise ValueError(
                f"{path} contains {len(names)} .txt members ({names!r}); "
                "unzip it and point --vectors at the file you want"
            )
        return io.TextIOWrapper(archive.open(names[0]), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _first_data_line(handle: IO[str]) -> str:
    """Return the first vector line, skipping a word2vec-style ``N dim`` header."""
    for line in handle:
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            continue  # word2vec header, e.g. "400000 50"
        return line
    raise ValueError("file contains no vectors")


def infer_dim(path: str | os.PathLike[str]) -> int:
    """Infer the embedding dimension from the first vector line."""
    with _open_text(Path(path)) as handle:
        return len(_first_data_line(handle).split()) - 1


def _iter_rows(handle: IO[str], dim: int, limit: int | None) -> Iterator[Tuple[str, List[str]]]:
    count = 0
    for lineno, line in enumerate(handle, start=1):
        line = line.rstrip("\n")
        if not line.strip():
            continue
        parts = line.split(" ")
        if len(parts) == 2 and all(p.isdigit() for p in parts):
            continue  # word2vec header
        if len(parts) < dim + 1:
            raise ValueError(
                f"line {lineno}: expected {dim} floats, found {len(parts) - 1}"
            )
        # Tokens are space-free in GloVe 6B, but split from the right anyway so
        # that vocabularies containing spaces still parse.
        token = " ".join(parts[: len(parts) - dim])
        yield token, parts[len(parts) - dim :]
        count += 1
        if limit is not None and count >= limit:
            return


def load_glove_text(
    path: str | os.PathLike[str],
    limit: int | None = None,
    dtype: np.dtype | str = np.float32,
) -> Tuple[List[str], np.ndarray]:
    """Parse a GloVe text file into ``(vocab, matrix)``.

    ``matrix[i]`` is the embedding for ``vocab[i]``.  ``limit`` keeps only the
    first N lines; GloVe files are sorted by descending corpus frequency, so a
    limit yields the N most common tokens.
    """
    path = Path(path)
    dim = infer_dim(path)
    vocab: List[str] = []
    chunks: List[np.ndarray] = []
    buffer: List[str] = []
    block = 50_000

    def flush() -> None:
        if buffer:
            chunks.append(np.array(buffer, dtype=dtype).reshape(-1, dim))
            buffer.clear()

    with _open_text(path) as handle:
        for token, values in _iter_rows(handle, dim, limit):
            vocab.append(token)
            buffer.extend(values)
            if len(buffer) >= block * dim:
                flush()
    flush()

    if not chunks:
        raise ValueError(f"{path} contains no vectors")
    matrix = np.concatenate(chunks, axis=0) if len(chunks) > 1 else chunks[0]
    return vocab, np.ascontiguousarray(matrix)


def cache_is_valid(cache_dir: str | os.PathLike[str], source: GloveSource) -> bool:
    meta_path = Path(cache_dir) / META_FILE
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if meta.get("cache_version") != CACHE_VERSION:
        return False
    return meta.get("source") == source.__dict__


def write_cache(
    cache_dir: str | os.PathLike[str],
    vocab: Sequence[str],
    matrix: np.ndarray,
    source: GloveSource,
) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / VECTORS_FILE, matrix)
    (cache_dir / VOCAB_FILE).write_text("\n".join(vocab) + "\n", encoding="utf-8")
    meta = {
        "cache_version": CACHE_VERSION,
        "dim": int(matrix.shape[1]),
        "count": int(matrix.shape[0]),
        "dtype": str(matrix.dtype),
        "source": source.__dict__,
    }
    (cache_dir / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return cache_dir


def read_cache(
    cache_dir: str | os.PathLike[str], mmap: bool = True
) -> Tuple[List[str], np.ndarray]:
    cache_dir = Path(cache_dir)
    matrix = np.load(cache_dir / VECTORS_FILE, mmap_mode="r" if mmap else None)
    vocab = (cache_dir / VOCAB_FILE).read_text(encoding="utf-8").splitlines()
    if len(vocab) != matrix.shape[0]:
        raise ValueError(
            f"corrupt cache in {cache_dir}: {len(vocab)} tokens vs {matrix.shape[0]} rows"
        )
    return vocab, matrix


def default_cache_dir(path: str | os.PathLike[str], limit: int | None = None) -> Path:
    """Cache location for a given source file: ``<file>.cache`` beside it."""
    suffix = ".cache" if limit is None else f".top{limit}.cache"
    return Path(path).with_suffix(Path(path).suffix + suffix)
