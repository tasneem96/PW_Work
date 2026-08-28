"""HNSW retrieval for the ANN-Benchmarks HDF5 datasets (e.g. glove-25-angular).

Those files hold four datasets::

    train      (n, d) float32   the database vectors
    test       (nq, d) float32  the query vectors
    neighbors  (nq, 100) int    exact top-100 ids for each query
    distances  (nq, 100) float  the matching exact distances

The ``-angular`` datasets are ground-truthed by **cosine**, so an index that
ranks by raw L2 or raw inner product will disagree with ``neighbors`` no matter
how well HNSW is tuned.  :func:`faiss_index` handles that.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

try:  # pragma: no cover
    import faiss
except ImportError:  # pragma: no cover
    faiss = None

ANGULAR = ("angular", "cosine")


def _require_faiss():
    if faiss is None:
        raise ImportError("faiss is not installed -- `pip install faiss-cpu`")
    return faiss


def as_float32(x) -> np.ndarray:
    """C-contiguous float32 view, copying only when necessary.

    h5py hands back float32 already, but a slice, a float64 array or an HDF5
    dataset object would all make faiss raise.
    """
    return np.ascontiguousarray(np.asarray(x), dtype=np.float32)


def normalized(x) -> np.ndarray:
    """L2-normalize a *copy*.

    ``faiss.normalize_L2`` works in place.  Calling it on the caller's ``xb``
    would silently rewrite their database array, so every normalization here
    goes through a copy.
    """
    _require_faiss()
    out = as_float32(x).copy()
    faiss.normalize_L2(out)
    return out


# ----------------------------------------------------------------------
# the index
# ----------------------------------------------------------------------
def build_hnsw(
    data,
    M: int = 32,
    ef_construction: int = 200,
    metric: str = "angular",
):
    """Build an HNSW graph index over ``data``.

    ``M`` is the number of neighbours per node (memory and recall both grow
    with it); ``ef_construction`` is the build-time candidate list.
    """
    _require_faiss()
    vectors = normalized(data) if metric in ANGULAR else as_float32(data)
    d = vectors.shape[1]

    # For unit-norm vectors ||a-b||^2 = 2-2cos, so L2 and inner product give
    # the same ordering; inner product is used so the returned scores are
    # cosine similarities directly.
    faiss_metric = faiss.METRIC_INNER_PRODUCT if metric in ANGULAR else faiss.METRIC_L2
    index = faiss.IndexHNSWFlat(d, M, faiss_metric)
    index.hnsw.efConstruction = int(ef_construction)
    index.add(vectors)
    return index


@dataclass
class HnswRun:
    """Timings and parameters from one :func:`faiss_index` call."""

    build_seconds: float
    query_seconds: float
    queries_per_second: float
    M: int
    ef_construction: int
    ef_search: int
    metric: str
    ntotal: int


def faiss_index(
    data,
    query,
    k: int = 100,
    M: int = 32,
    ef_construction: int = 200,
    ef_search: int | None = None,
    metric: str = "angular",
    stats: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, HnswRun]:
    """Build an HNSW index over ``data`` and return the top-``k`` for ``query``.

    Returns ``(distances, indices)`` in faiss order, so::

        D, I = faiss_index(xb, xq, k=100)
        recall = recall_at_k(I, ground_truth, k=10)

    With ``metric='angular'`` (the default, matching ``glove-25-angular``) both
    sides are L2-normalized and ``D`` holds **cosine similarity** -- higher is
    better, 1.0 is identical.  Pass ``stats=True`` to also get an
    :class:`HnswRun` with timings.

    ``ef_search`` defaults to ``max(2*k, 64)``.  It is never allowed below
    ``k``: HNSW's own default is 16, and searching for 100 neighbours with a
    candidate list of 16 cannot return 100 good ones -- recall collapses in a
    way that looks like a broken index rather than a tuning knob.
    """
    _require_faiss()
    if k <= 0:
        raise ValueError("k must be positive")
    if metric not in ANGULAR and metric not in ("euclidean", "l2"):
        raise ValueError(f"unknown metric {metric!r}; use 'angular' or 'euclidean'")

    queries = normalized(query) if metric in ANGULAR else as_float32(query)

    start = time.perf_counter()
    index = build_hnsw(data, M=M, ef_construction=ef_construction, metric=metric)
    build_seconds = time.perf_counter() - start

    if index.d != queries.shape[1]:
        raise ValueError(
            f"query vectors are {queries.shape[1]}-d but the database is {index.d}-d"
        )
    if k > index.ntotal:
        raise ValueError(f"k={k} exceeds the {index.ntotal} vectors in the index")

    requested = max(2 * k, 64) if ef_search is None else int(ef_search)
    index.hnsw.efSearch = max(requested, k)

    start = time.perf_counter()
    distances, indices = index.search(queries, k)
    query_seconds = time.perf_counter() - start

    if not stats:
        return distances, indices
    run = HnswRun(
        build_seconds=build_seconds,
        query_seconds=query_seconds,
        queries_per_second=len(queries) / query_seconds if query_seconds else float("inf"),
        M=M,
        ef_construction=ef_construction,
        ef_search=int(index.hnsw.efSearch),
        metric=metric,
        ntotal=int(index.ntotal),
    )
    return distances, indices, run


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
def recall_at_k(found: np.ndarray, truth: np.ndarray, k: int | None = None) -> float:
    """Mean fraction of the exact top-k that the index actually returned.

    Compares ids, so it does not depend on how the file defines distance.
    Both arrays are ``(n_queries, >= k)``.
    """
    found = np.asarray(found)
    truth = np.asarray(truth)
    if found.shape[0] != truth.shape[0]:
        raise ValueError(
            f"{found.shape[0]} result rows vs {truth.shape[0]} ground-truth rows"
        )
    k = min(found.shape[1], truth.shape[1]) if k is None else int(k)
    if k > found.shape[1] or k > truth.shape[1]:
        raise ValueError(
            f"recall@{k} needs {k} columns; have {found.shape[1]} found, {truth.shape[1]} truth"
        )
    hits = sum(len(set(f[:k].tolist()) & set(t[:k].tolist())) for f, t in zip(found, truth))
    return hits / (k * found.shape[0])


# Candidate conventions for an "angular" distance, given cosine similarity.
DISTANCE_CONVENTIONS: Dict[str, object] = {
    "cosine_similarity": lambda cos: cos,
    "1 - cosine": lambda cos: 1.0 - cos,
    "euclidean_on_unit": lambda cos: np.sqrt(np.maximum(2.0 - 2.0 * cos, 0.0)),
    "squared_euclidean_on_unit": lambda cos: np.maximum(2.0 - 2.0 * cos, 0.0),
    "arccos": lambda cos: np.arccos(np.clip(cos, -1.0, 1.0)),
}


def detect_convention_for_results(
    found_ids: np.ndarray,
    cosine: np.ndarray,
    truth_ids: np.ndarray,
    truth_distances: np.ndarray,
    tol: float = 1e-3,
) -> str | None:
    """Detect the file's distance convention from an approximate search.

    An ANN result and the ground truth do not line up cell for cell -- that is
    the whole point of an approximate index -- so comparing them elementwise
    compares distances to *different* neighbours and matches nothing.  Only the
    positions where the returned id equals the exact id are comparable.
    """
    found_ids = np.asarray(found_ids)
    truth_ids = np.asarray(truth_ids)
    cosine = np.asarray(cosine)
    truth_distances = np.asarray(truth_distances)
    # Callers routinely benchmark a subset of the queries; line the arrays up on
    # their common rows and columns rather than trusting them to match.
    rows = min(found_ids.shape[0], truth_ids.shape[0], truth_distances.shape[0], cosine.shape[0])
    width = min(found_ids.shape[1], truth_ids.shape[1], truth_distances.shape[1], cosine.shape[1])
    if rows == 0 or width == 0:
        return None
    mask = found_ids[:rows, :width] == truth_ids[:rows, :width]
    if not mask.any():
        return None
    return detect_distance_convention(
        cosine[:rows, :width][mask], truth_distances[:rows, :width][mask], tol=tol
    )


def detect_distance_convention(
    cosine: np.ndarray, reference: np.ndarray, tol: float = 1e-3
) -> str | None:
    """Work out how a dataset's ``distances`` relate to cosine similarity.

    ANN-Benchmarks files are not self-describing on this point and the
    convention has varied, so this measures it against the file instead of
    assuming.  Returns the best-matching name, or None if nothing fits.
    """
    cosine = np.asarray(cosine, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    best, best_error = None, np.inf
    for name, fn in DISTANCE_CONVENTIONS.items():
        error = float(np.nanmax(np.abs(np.asarray(fn(cosine)) - reference)))
        if error < best_error:
            best, best_error = name, error
    return best if best_error <= tol else None


# ----------------------------------------------------------------------
# dataset loading
# ----------------------------------------------------------------------
def load_dataset(path: str | Path) -> Dict[str, np.ndarray]:
    """Read an ANN-Benchmarks HDF5 file into memory."""
    import h5py  # imported lazily: only the benchmark path needs it

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- download it from "
            "https://github.com/erikbern/ann-benchmarks (e.g. glove-25-angular.hdf5)"
        )
    with h5py.File(path, "r") as f:
        missing = {"train", "test", "neighbors"} - set(f.keys())
        if missing:
            raise ValueError(f"{path} is missing dataset(s): {sorted(missing)}")
        out = {
            "train": f["train"][:],
            "test": f["test"][:],
            "neighbors": f["neighbors"][:],
        }
        if "distances" in f:
            out["distances"] = f["distances"][:]
        out["metric"] = f.attrs.get("distance", "angular")
    return out
