"""Corpora and queries declared by the frozen protocol.

Two dataset kinds exist.

``synthetic``
    Generated on demand from the frozen root seed. No download, no cache, no
    checksum drift: the same protocol reproduces the same bytes anywhere. These
    carry the Phase 0/1 development and smoke runs.
``external``
    Real corpora and embedding models (SIFT, GloVe, sentence-transformer
    embeddings of a text corpus). The protocol declares them with a loader
    contract and an ``available`` flag. Requesting an unavailable dataset fails
    loudly with the path and checksum it expects, instead of silently falling
    back to synthetic data and quietly changing what a result means.

Queries are generated as perturbed corpus points, so a query has a meaningful
nearest neighbour and recall is not dominated by ties among far-away vectors.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .protocol import Protocol
from .rng import generator
from .vectors import VectorStore, make_store


class DatasetUnavailable(RuntimeError):
    """A declared external dataset is not present in this environment."""


@dataclass(frozen=True, eq=False)
class Dataset:
    dataset_id: str
    store: VectorStore
    queries: np.ndarray
    spec: dict[str, Any]

    @property
    def n(self) -> int:
        return self.store.n

    @property
    def dim(self) -> int:
        return self.store.dim

    @property
    def n_queries(self) -> int:
        return int(self.queries.shape[0])

    def fingerprint(self) -> dict[str, Any]:
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(self.queries).tobytes())
        return {
            "dataset_id": self.dataset_id,
            "kind": self.spec.get("kind"),
            "n": self.n,
            "dim": self.dim,
            "n_queries": self.n_queries,
            "numeric_type": self.store.numeric_type,
            "corpus_hash": self.store.content_hash(),
            "query_hash": h.hexdigest(),
        }


# --------------------------------------------------------------------------
# synthetic generators
# --------------------------------------------------------------------------

def _gaussian_clusters(
    *, n: int, dim: int, seed: int, dataset_id: str, params: dict[str, Any]
) -> np.ndarray:
    n_clusters = int(params.get("n_clusters", 64))
    center_scale = float(params.get("center_scale", 1.0))
    within_sigma = float(params.get("within_sigma", 0.35))
    rng = generator(seed, "corpus", dataset_id)
    centers = rng.normal(scale=center_scale, size=(n_clusters, dim))
    assign = rng.integers(0, n_clusters, size=n)
    return centers[assign] + rng.normal(scale=within_sigma, size=(n, dim))


def _gaussian_isotropic(
    *, n: int, dim: int, seed: int, dataset_id: str, params: dict[str, Any]
) -> np.ndarray:
    scale = float(params.get("scale", 1.0))
    rng = generator(seed, "corpus", dataset_id)
    return rng.normal(scale=scale, size=(n, dim))


def _skewed_norms(
    *, n: int, dim: int, seed: int, dataset_id: str, params: dict[str, Any]
) -> np.ndarray:
    """Clustered directions with a heavy-tailed norm distribution.

    Cosine search ignores norms, exponent bit flips do not. Keeping one corpus
    with a wide norm spread stops the later bit-class analysis from being an
    artifact of unit-norm data.
    """
    base = _gaussian_clusters(n=n, dim=dim, seed=seed, dataset_id=dataset_id, params=params)
    rng = generator(seed, "corpus-norms", dataset_id)
    scale = np.exp(rng.normal(loc=0.0, scale=float(params.get("log_norm_sigma", 1.0)), size=(n, 1)))
    return base * scale


GENERATORS: dict[str, Callable[..., np.ndarray]] = {
    "gaussian_clusters": _gaussian_clusters,
    "gaussian_isotropic": _gaussian_isotropic,
    "gaussian_clusters_skewed_norms": _skewed_norms,
}


def _make_queries(
    corpus: np.ndarray,
    *,
    n_queries: int,
    seed: int,
    dataset_id: str,
    spec: dict[str, Any],
) -> np.ndarray:
    """Queries for a corpus of exactly the size being evaluated.

    The query stream is addressed by corpus size as well as dataset id. A run
    that subsets the corpus therefore gets its own reproducible query set drawn
    from the vectors that are actually present, instead of perturbations of
    corpus points that the subset dropped, which would quietly turn queries
    into near-random vectors and change how hard search is. The corpus size is
    recorded in the dataset fingerprint, so a subset run is never mistaken for
    a full one.
    """
    query_spec = spec.get("queries", {"generator": "perturbed_corpus", "sigma": 0.1})
    kind = str(query_spec.get("generator", "perturbed_corpus"))
    rng = generator(seed, "queries", dataset_id, f"n={corpus.shape[0]}")
    if kind == "perturbed_corpus":
        # sigma is the *relative* perturbation size: the added noise has norm
        # about sigma times the mean corpus-vector norm, so sigma = 0.12 means a
        # query sits at cosine ~0.99 from its source point regardless of
        # dimension. Treating sigma as a per-coordinate scale instead would make
        # the perturbation grow with sqrt(d) and, at d = 128, leave queries
        # almost orthogonal to their source, which is not a retrieval workload.
        sigma = float(query_spec.get("sigma", 0.1))
        # Distinct source points when the corpus is large enough; with a
        # subset corpus smaller than the declared query count, sampling with
        # replacement is the only option and is recorded by the corpus size in
        # the fingerprint.
        picks = rng.choice(
            corpus.shape[0], size=int(n_queries), replace=int(n_queries) > corpus.shape[0]
        )
        scale = sigma * float(np.mean(np.linalg.norm(corpus, axis=1))) / np.sqrt(corpus.shape[1])
        return corpus[picks] + rng.normal(scale=scale, size=(int(n_queries), corpus.shape[1]))
    if kind == "same_distribution":
        return rng.normal(
            scale=float(np.std(corpus)), size=(int(n_queries), corpus.shape[1])
        )
    raise ValueError(f"unknown query generator {kind!r} for dataset {dataset_id!r}")


# --------------------------------------------------------------------------
# external loader contract
# --------------------------------------------------------------------------

def _load_external(spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    loader = spec.get("loader", {})
    root = Path(str(loader.get("root", "data"))).expanduser()
    corpus_path = root / str(loader.get("corpus_file", "corpus.npy"))
    query_path = root / str(loader.get("query_file", "queries.npy"))
    if not spec.get("available", False) or not corpus_path.exists():
        raise DatasetUnavailable(
            f"dataset {spec['id']!r} is declared but not available here.\n"
            f"  expected corpus: {corpus_path}\n"
            f"  expected queries: {query_path}\n"
            f"  embedding model: {spec.get('embedding_model', 'n/a')}\n"
            f"  sha256 policy: {loader.get('checksums', 'declare checksums before first use')}\n"
            f"Phase 8 runs the frozen grid on external data; Phase 0/1 runs use the synthetic "
            f"datasets declared in the same protocol. Do not substitute one for the other."
        )
    corpus = np.load(corpus_path)
    queries = np.load(query_path)
    return corpus, queries


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------

def load_dataset(
    protocol: Protocol,
    dataset_id: str,
    *,
    numeric_type: str | None = None,
    n: int | None = None,
    n_queries: int | None = None,
) -> Dataset:
    """Materialize a declared dataset in a declared storage type.

    ``n`` and ``n_queries`` may only shrink what the protocol declares (smoke
    runs); growing them would evaluate a corpus the protocol never froze.
    """
    spec = protocol.dataset(dataset_id)
    declared_n = int(spec["n"])
    declared_q = int(spec["n_queries"])
    use_n = declared_n if n is None else int(n)
    use_q = declared_q if n_queries is None else int(n_queries)
    if use_n > declared_n or use_q > declared_q:
        raise ValueError(
            f"dataset {dataset_id!r} declares n={declared_n}, n_queries={declared_q}; a run may "
            f"subset but not exceed the frozen size (asked n={use_n}, n_queries={use_q})"
        )
    numeric_type = numeric_type or spec["numeric_types"][0]
    if numeric_type not in spec["numeric_types"]:
        raise ValueError(
            f"dataset {dataset_id!r} does not declare numeric type {numeric_type!r} "
            f"(declared: {spec['numeric_types']})"
        )

    if spec["kind"] == "synthetic":
        gen_name = str(spec["generator"])
        if gen_name not in GENERATORS:
            raise ValueError(f"unknown generator {gen_name!r} for dataset {dataset_id!r}")
        corpus = GENERATORS[gen_name](
            n=declared_n,
            dim=int(spec["dim"]),
            seed=protocol.root_seed,
            dataset_id=dataset_id,
            params=spec.get("params", {}),
        )
        corpus = np.ascontiguousarray(corpus[:use_n])
        queries = _make_queries(
            corpus,
            n_queries=declared_q,
            seed=protocol.root_seed,
            dataset_id=dataset_id,
            spec=spec,
        )
    elif spec["kind"] == "external":
        corpus, queries = _load_external(spec)
        corpus = np.ascontiguousarray(corpus[:use_n])
    else:
        raise ValueError(f"unknown dataset kind {spec['kind']!r} for {dataset_id!r}")

    queries = np.ascontiguousarray(queries[:use_q], dtype=np.float32)
    if corpus.shape[1] != int(spec["dim"]):
        raise ValueError(
            f"dataset {dataset_id!r} declares dim={spec['dim']} but data has {corpus.shape[1]}"
        )
    store = make_store(corpus, numeric_type, label=f"D[{dataset_id}/{numeric_type}]")
    store.assert_finite()
    return Dataset(dataset_id=dataset_id, store=store, queries=queries, spec=spec)
