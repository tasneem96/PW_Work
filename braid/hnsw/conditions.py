"""The four evaluation conditions of Section 15.

exact
    Exact search over the (possibly corrupted) vectors. Answers "did the
    geometry itself change?"
hnsw_clean
    (G(D), D). The pre-attack baseline; the only condition that defines which
    queries are eligible for an attack-success claim.
hnsw_stale
    (G(D), D'). The graph still encodes clean geometry while node distances are
    computed from corrupted vectors. This is the attack condition.
hnsw_rebuilt
    (G(D'), D'). A fresh graph over the corrupted vectors, which separates
    "the corruption broke the geometry" from "the stale graph amplified it".

Phase 1 runs all four with D' = D. That is not a trivial exercise: it is the
identity check that the stale path is genuinely the same code as the clean path
(bitwise-identical outputs and work counters) and that a rebuild with the same
seed reproduces the same graph. Any later stale-versus-clean gap can then be
attributed to the corruption rather than to two subtly different code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..exact import ExactResult, exact_topk
from ..vectors import VectorStore
from .params import HnswParams
from .reference import HnswGraph, build_index, search_many
from .trace import ExposurePolicy, QueryTrace, TraceLevel

CONDITIONS = ("exact", "hnsw_clean", "hnsw_stale", "hnsw_rebuilt")


@dataclass(frozen=True, eq=False)
class ConditionResult:
    """Retrieved ids per query for one condition, plus its traces."""

    name: str
    ids: np.ndarray
    scores: np.ndarray
    traces: list[QueryTrace] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def id_sets(self) -> list[set[int]]:
        return [set(int(i) for i in row if i >= 0) for row in self.ids]

    def work_frame(self) -> dict[str, np.ndarray]:
        """Per-query work counters as arrays, empty for the exact condition."""
        if not self.traces:
            return {}
        keys = ("distance_evals", "expansions", "unique_visited", "latency_ns")
        return {
            key: np.array([t.work_summary()[key] for t in self.traces], dtype=np.float64)
            for key in keys
        }


def evaluate_conditions(
    *,
    clean_store: VectorStore,
    queries: np.ndarray,
    params: HnswParams,
    k: int = 10,
    ef_search: int = 50,
    corrupted_store: VectorStore | None = None,
    graph_clean: HnswGraph | None = None,
    graph_rebuilt: HnswGraph | None = None,
    conditions: Sequence[str] = CONDITIONS,
    trace_level: TraceLevel = TraceLevel.COUNTERS,
    exposure: ExposurePolicy | None = None,
    query_ids: Sequence[int] | None = None,
) -> dict[str, ConditionResult]:
    """Run the requested conditions on one query set.

    ``corrupted_store`` defaults to ``clean_store``, i.e. D' = D, which is the
    Phase 1 identity configuration. Phase 2 onwards passes a real D'.
    """
    for name in conditions:
        if name not in CONDITIONS:
            raise ValueError(f"unknown condition {name!r}; expected a subset of {CONDITIONS}")
    corrupted = corrupted_store if corrupted_store is not None else clean_store
    if corrupted.n != clean_store.n or corrupted.dim != clean_store.dim:
        raise ValueError("D' must have the same shape as D: bit flips do not add or remove vectors")

    graph_clean = graph_clean or build_index(clean_store, params)
    out: dict[str, ConditionResult] = {}

    if "exact" in conditions:
        exact: ExactResult = exact_topk(queries, corrupted, k, params.convention)
        out["exact"] = ConditionResult(
            name="exact",
            ids=exact.ids,
            scores=exact.scores,
            meta={
                "store": corrupted.label,
                "store_hash": corrupted.content_hash(),
                "tie_rate": float(exact.tie_mask().mean()),
            },
        )

    for name, graph, store in (
        ("hnsw_clean", graph_clean, clean_store),
        ("hnsw_stale", graph_clean, corrupted),
    ):
        if name not in conditions:
            continue
        ids, distances, traces = search_many(
            graph,
            store,
            queries,
            k=k,
            ef_search=ef_search,
            trace_level=trace_level,
            exposure=exposure,
            query_ids=query_ids,
        )
        out[name] = ConditionResult(
            name=name,
            ids=ids,
            scores=-distances,
            traces=traces,
            meta={
                "graph_hash": graph.structure_hash(),
                "graph_built_over": graph.build_meta.get("store_label"),
                "store": store.label,
                "store_hash": store.content_hash(),
                "ef_search": int(ef_search),
                "k": int(k),
            },
        )

    if "hnsw_rebuilt" in conditions:
        rebuilt = graph_rebuilt or build_index(corrupted, params)
        ids, distances, traces = search_many(
            rebuilt,
            corrupted,
            queries,
            k=k,
            ef_search=ef_search,
            trace_level=trace_level,
            exposure=exposure,
            query_ids=query_ids,
        )
        out["hnsw_rebuilt"] = ConditionResult(
            name="hnsw_rebuilt",
            ids=ids,
            scores=-distances,
            traces=traces,
            meta={
                "graph_hash": rebuilt.structure_hash(),
                "graph_built_over": rebuilt.build_meta.get("store_label"),
                "store": corrupted.label,
                "store_hash": corrupted.content_hash(),
                "ef_search": int(ef_search),
                "k": int(k),
            },
        )
    return out


def identity_check(results: dict[str, ConditionResult]) -> dict[str, Any]:
    """With D' = D, clean and stale must agree exactly. Verifies that they do."""
    if "hnsw_clean" not in results or "hnsw_stale" not in results:
        return {"checked": False, "reason": "clean and stale conditions both required"}
    clean, stale = results["hnsw_clean"], results["hnsw_stale"]
    ids_equal = bool(np.array_equal(clean.ids, stale.ids))
    scores_equal = bool(np.allclose(clean.scores, stale.scores, atol=0, rtol=0, equal_nan=True))
    clean_work, stale_work = clean.work_frame(), stale.work_frame()
    work_equal = all(
        bool(np.array_equal(clean_work[key], stale_work[key]))
        for key in ("distance_evals", "expansions", "unique_visited")
        if key in clean_work and key in stale_work
    )
    return {
        "checked": True,
        "ids_equal": ids_equal,
        "scores_equal": scores_equal,
        "deterministic_work_equal": work_equal,
        "passed": bool(ids_equal and scores_equal and work_equal),
    }
