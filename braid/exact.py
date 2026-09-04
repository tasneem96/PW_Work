"""Exact search: i*(q), the clean exact top-k, and an independent cross-check.

Phase 1's exit gate requires that exact-search answers be independently
checked. "Independently" here means three code paths that share no ranking
logic: a chunked matrix formulation, a per-query loop written from the
definition, and (when installed) hnswlib's brute-force index, which is a
different language and a different author.

Ties are reported rather than hidden. Every downstream argmax claim, including
the geometry margin of Section 8, is fragile exactly where two candidates sit
within float noise of each other, so the tie rate is a first-class diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import similarity as sim
from .vectors import VectorStore

DEFAULT_TIE_TOL = 1e-6


@dataclass(frozen=True, eq=False)
class ExactResult:
    """Top-k ids and similarity scores ("larger is better") per query."""

    ids: np.ndarray  # (n_queries, k) int64
    scores: np.ndarray  # (n_queries, k) float32
    convention: str
    k: int

    @property
    def nearest(self) -> np.ndarray:
        """i*(q) for every query."""
        return self.ids[:, 0].copy()

    def top_set(self, query_index: int) -> set[int]:
        return set(int(i) for i in self.ids[query_index])

    def tie_mask(self, tol: float = DEFAULT_TIE_TOL) -> np.ndarray:
        """Queries whose top-1 is within ``tol`` of the runner-up."""
        if self.k < 2:
            return np.zeros(self.ids.shape[0], dtype=bool)
        return (self.scores[:, 0] - self.scores[:, 1]) <= tol


def exact_topk(
    queries: np.ndarray,
    store: VectorStore,
    k: int,
    convention: str = "cosine",
    *,
    chunk: int = 256,
) -> ExactResult:
    """Chunked matrix implementation (the one used in experiments)."""
    sim.check_convention(convention)
    q = sim.as_f32(queries)
    if q.ndim == 1:
        q = q[None, :]
    k = int(min(k, store.n))
    ids = np.empty((q.shape[0], k), dtype=np.int64)
    scores = np.empty((q.shape[0], k), dtype=np.float32)
    vectors = store.as_f32()
    for start in range(0, q.shape[0], chunk):
        block = q[start : start + chunk]
        s = sim.similarity_matrix(block, vectors, convention)  # type: ignore[arg-type]
        part = np.argpartition(-s, kth=k - 1, axis=1)[:, :k]
        part_scores = np.take_along_axis(s, part, axis=1)
        order = np.argsort(-part_scores, axis=1, kind="stable")
        block_ids = np.take_along_axis(part, order, axis=1)
        # Break exact ties by smaller id so the answer is deterministic.
        block_scores = np.take_along_axis(part_scores, order, axis=1)
        for row in range(block_ids.shape[0]):
            _stable_tie_sort(block_ids[row], block_scores[row])
        ids[start : start + block.shape[0]] = block_ids
        scores[start : start + block.shape[0]] = block_scores
    return ExactResult(ids=ids, scores=scores, convention=convention, k=k)


def _stable_tie_sort(ids: np.ndarray, scores: np.ndarray) -> None:
    """In-place: among bitwise-equal scores, order by ascending id."""
    order = np.lexsort((ids, -scores))
    ids[:] = ids[order]
    scores[:] = scores[order]


def exact_topk_reference(
    queries: np.ndarray,
    store: VectorStore,
    k: int,
    convention: str = "cosine",
) -> ExactResult:
    """Per-query loop written straight from the definitions in Section 3.2."""
    sim.check_convention(convention)
    q = sim.as_f32(queries)
    if q.ndim == 1:
        q = q[None, :]
    vectors = store.as_f32()
    k = int(min(k, store.n))
    ids = np.empty((q.shape[0], k), dtype=np.int64)
    scores = np.empty((q.shape[0], k), dtype=np.float32)
    for row in range(q.shape[0]):
        if convention == "cosine":
            qn = q[row] / max(float(np.linalg.norm(q[row])), 1e-12)
            s = np.array(
                [
                    float(np.dot(qn, v / max(float(np.linalg.norm(v)), 1e-12)))
                    for v in vectors
                ],
                dtype=np.float32,
            )
        else:
            s = np.array(
                [-float(np.sum((v - q[row]) ** 2)) for v in vectors], dtype=np.float32
            )
        order = np.lexsort((np.arange(s.size), -s))[:k]
        ids[row] = order
        scores[row] = s[order]
    return ExactResult(ids=ids, scores=scores, convention=convention, k=k)


def exact_topk_native(
    queries: np.ndarray,
    store: VectorStore,
    k: int,
    convention: str = "cosine",
) -> ExactResult | None:
    """hnswlib brute force, when the optional native dependency is installed.

    hnswlib 0.8.0 is inconsistent about cosine: ``Index`` normalizes stored
    items and queries, while ``BFIndex`` does not, so ``BFIndex(space="cosine")``
    actually returns ``1 - <q, e>`` on raw vectors. We therefore normalize
    explicitly and use the inner-product space, which reproduces
    ``d = 1 - cos`` exactly. The same quirk is why the deployed-index parity
    check in :mod:`braid.hnsw.native` feeds hnswlib pre-normalized vectors.
    """
    try:
        import hnswlib
    except Exception:  # pragma: no cover - optional dependency
        return None
    q = sim.as_f32(queries)
    if q.ndim == 1:
        q = q[None, :]
    if convention == "cosine":
        space = "ip"
        items = sim.normalize(store.as_f32())
        q_native = sim.normalize(q)
    else:
        space = "l2"
        items = store.as_f32()
        q_native = q
    index = hnswlib.BFIndex(space=space, dim=store.dim)
    index.init_index(max_elements=store.n)
    index.add_items(items, np.arange(store.n))
    k = int(min(k, store.n))
    labels, distances = index.knn_query(q_native, k=k)
    scores = sim.distance_from_similarity(
        np.asarray(distances, dtype=np.float32), convention  # type: ignore[arg-type]
    )
    return ExactResult(
        ids=np.asarray(labels, dtype=np.int64),
        scores=np.asarray(scores, dtype=np.float32),
        convention=convention,
        k=k,
    )


def cross_check_exact(
    queries: np.ndarray,
    store: VectorStore,
    k: int,
    convention: str = "cosine",
    *,
    tol: float = 1e-4,
    max_queries: int | None = 64,
) -> dict[str, Any]:
    """Compare the three exact implementations on the same queries.

    Agreement is judged on *scores*, not id lists: with genuine ties two
    correct implementations may legitimately return different ids. A query
    counts as a mismatch only when the returned score multisets differ beyond
    ``tol``, which is a real ranking disagreement.
    """
    q = sim.as_f32(queries)
    if q.ndim == 1:
        q = q[None, :]
    if max_queries is not None:
        q = q[: int(max_queries)]

    primary = exact_topk(q, store, k, convention)
    loop = exact_topk_reference(q, store, k, convention)
    native = exact_topk_native(q, store, k, convention)

    report: dict[str, Any] = {
        "n_queries_checked": int(q.shape[0]),
        "k": int(primary.k),
        "convention": convention,
        "numeric_type": store.numeric_type,
        "tol": tol,
        "tie_rate": float(primary.tie_mask().mean()),
        "comparisons": {},
        "passed": True,
    }

    for name, other in (("reference_loop", loop), ("hnswlib_brute_force", native)):
        if other is None:
            report["comparisons"][name] = {"available": False}
            continue
        score_gap = float(np.max(np.abs(np.sort(primary.scores, axis=1) - np.sort(other.scores, axis=1))))
        id_agreement = float(
            np.mean([
                len(set(primary.ids[i].tolist()) & set(other.ids[i].tolist())) / primary.k
                for i in range(primary.ids.shape[0])
            ])
        )
        top1_agreement = float(np.mean(primary.ids[:, 0] == other.ids[:, 0]))
        passed = score_gap <= tol
        report["comparisons"][name] = {
            "available": True,
            "max_score_gap": score_gap,
            "mean_topk_id_agreement": id_agreement,
            "top1_id_agreement": top1_agreement,
            "passed": bool(passed),
        }
        report["passed"] = bool(report["passed"] and passed)
    return report
