"""Replay a faiss HNSW search step by step, recording the path through the graph.

faiss reports only counters (``ndis``, ``nhops``) for a search -- it never hands
back the nodes it visited.  This module walks the same graph with the same
algorithm in Python, so the traversal can be inspected: the entry point, the
greedy hops down the upper layers, and the beam expansion at layer 0.

Because it is a re-implementation, it is only worth anything if it matches.
:func:`verify_against_faiss` checks two things against the real search: the
top-k must be identical, and the number of distance computations must equal
faiss's own ``hnsw_stats.ndis``.  The second is the strict one -- two different
traversals essentially never agree on that count.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

try:  # pragma: no cover
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


def _require_faiss():
    if faiss is None:
        raise ImportError("faiss is not installed -- `pip install faiss-cpu`")
    return faiss


# ----------------------------------------------------------------------
# graph accessors
# ----------------------------------------------------------------------
class HnswGraph:
    """Read-only view of an IndexHNSW's graph, with the neighbour array cached."""

    def __init__(self, index):
        _require_faiss()
        self.index = index
        self.hnsw = index.hnsw
        # hnsw.neighbors is a MaybeOwnedVectorInt32 in faiss >= 1.9: .at() raises
        # TypeError and it is not subscriptable, so read it out once.
        self.neighbors = faiss.vector_to_array(self.hnsw.neighbors)
        self.entry_point = int(self.hnsw.entry_point)
        self.max_level = int(self.hnsw.max_level)
        self._begin = np.zeros(1, dtype="uint64")
        self._end = np.zeros(1, dtype="uint64")

    def level_of(self, node: int) -> int:
        return self.hnsw.levels.at(int(node)) - 1

    def neighbors_of(self, node: int, level: int) -> np.ndarray:
        self.hnsw.neighbor_range(int(node), int(level),
                                 faiss.swig_ptr(self._begin), faiss.swig_ptr(self._end))
        nb = self.neighbors[int(self._begin[0]):int(self._end[0])]
        # faiss stops at the first -1 rather than skipping it
        stop = np.argmax(nb < 0) if (nb < 0).any() else len(nb)
        return nb[:stop]


# ----------------------------------------------------------------------
# the trace
# ----------------------------------------------------------------------
@dataclass
class Hop:
    """One greedy move on an upper layer."""

    level: int
    source: int
    target: int
    distance: float


@dataclass
class Expansion:
    """One node popped from the candidate list at layer 0."""

    step: int
    node: int
    distance: float
    discovered: List[int] = field(default_factory=list)


@dataclass
class Trace:
    """Everything the search touched, in order."""

    entry_point: int
    hops: List[Hop]
    expansions: List[Expansion]
    visited: Dict[int, float]
    results: List[Tuple[int, float]]      # (node, internal distance), best first
    ndis: int
    ef_search: int
    k: int
    max_level: int = 0
    level_entry: Dict[int, int] = field(default_factory=dict)   # level -> node on arrival

    def path_at_level(self, level: int) -> List[int]:
        """The nodes the greedy walk stood on at one level, in order.

        A level with no hops still has one entry: the walk arrived and found it
        was already at a local minimum, so it dropped straight through.
        """
        start = self.level_entry.get(level)
        seq = [start] if start is not None else []
        for hop in self.hops:
            if hop.level == level:
                seq.append(hop.target)
        return seq

    @property
    def greedy_path(self) -> List[int]:
        """Entry point, then each node the greedy descent moved to."""
        return [self.entry_point] + [h.target for h in self.hops]

    @property
    def expanded_nodes(self) -> List[int]:
        return [e.node for e in self.expansions]

    def public_score(self, distance: float, metric_is_ip: bool) -> float:
        """Internal distance -> the number faiss would report."""
        return -distance if metric_is_ip else distance

    def summary(self) -> str:
        lines = [
            f"entry point : {self.entry_point}",
            f"greedy path : {' -> '.join(map(str, self.greedy_path))}"
            f"   ({len(self.hops)} hops down levels {self.hops[0].level if self.hops else '-'}..1)",
            f"layer 0     : {len(self.expansions)} nodes expanded, "
            f"{len(self.visited)} visited, {self.ndis} distances computed",
            f"efSearch={self.ef_search}  k={self.k}",
        ]
        return "\n".join(lines)


def _distance_fn(index, query: np.ndarray):
    """faiss ranks by a distance (smaller is better), so IP becomes -IP."""
    _require_faiss()
    is_ip = index.metric_type == faiss.METRIC_INNER_PRODUCT
    q = np.asarray(query, dtype=np.float32).reshape(-1)

    def distance(node: int) -> float:
        v = index.reconstruct(int(node))
        return -float(v @ q) if is_ip else float(((v - q) ** 2).sum())

    return distance, is_ip


def search_with_trace(
    index, query, k: int = 10, ef_search: int | None = None
) -> Tuple[np.ndarray, np.ndarray, Trace]:
    """Run faiss's HNSW search in Python, returning ``(distances, ids, trace)``.

    ``distances`` use the index's public convention, so they line up with what
    ``index.search`` returns: cosine similarity for an inner-product index,
    **squared** euclidean distance for an L2 one (which is what faiss reports).
    """
    _require_faiss()
    graph = HnswGraph(index)
    qdis, is_ip = _distance_fn(index, query)
    ef = max(int(ef_search if ef_search is not None else index.hnsw.efSearch), k)

    if graph.entry_point == -1:
        empty = np.zeros((1, k), dtype=np.float32), np.full((1, k), -1, dtype=np.int64)
        return (*empty, Trace(-1, [], [], {}, [], 0, ef, k))  # empty index

    # faiss does not count the entry point's own distance in hnsw_stats.ndis,
    # so neither do we -- that is what makes the counts line up exactly.
    ndis = 0
    visited: Dict[int, float] = {}

    # --- greedy descent through the upper layers ----------------------
    nearest = graph.entry_point
    d_nearest = qdis(nearest)
    visited[nearest] = d_nearest
    hops: List[Hop] = []
    level_entry: Dict[int, int] = {}

    for level in range(graph.max_level, 0, -1):
        level_entry[level] = nearest
        while True:
            previous = nearest
            for neighbor in graph.neighbors_of(previous, level):
                neighbor = int(neighbor)
                d = qdis(neighbor)
                ndis += 1
                visited.setdefault(neighbor, d)
                if d < d_nearest:
                    nearest, d_nearest = neighbor, d
            if nearest == previous:
                break
            hops.append(Hop(level=level, source=previous, target=nearest, distance=d_nearest))

    # --- beam search at layer 0 ---------------------------------------
    candidates = [(d_nearest, nearest)]            # min-heap, the frontier
    frontier = [(-d_nearest, nearest)]             # max-heap on distance, capacity ef
    results: List[Tuple[float, int]] = [(-d_nearest, nearest)]   # capacity k
    seen = {nearest}
    expansions: List[Expansion] = []

    while candidates:
        d0, node = heapq.heappop(candidates)
        # Stop once the nearest unexplored candidate is worse than the whole
        # ef-sized frontier: nothing reachable through it can still get in.
        if len(frontier) >= ef and d0 > -frontier[0][0]:
            break
        expansion = Expansion(step=len(expansions), node=node, distance=d0)
        for neighbor in graph.neighbors_of(node, 0):
            neighbor = int(neighbor)
            if neighbor in seen:
                continue
            seen.add(neighbor)
            d = qdis(neighbor)
            ndis += 1
            visited.setdefault(neighbor, d)
            expansion.discovered.append(neighbor)
            if len(frontier) < ef or d < -frontier[0][0]:
                heapq.heappush(candidates, (d, neighbor))
                heapq.heappush(frontier, (-d, neighbor))
                if len(frontier) > ef:
                    heapq.heappop(frontier)
            if len(results) < k:
                heapq.heappush(results, (-d, neighbor))
            elif d < -results[0][0]:
                heapq.heapreplace(results, (-d, neighbor))
        expansions.append(expansion)

    ordered = sorted(((d, n) for d, n in ((-nd, n) for nd, n in results)))
    level_entry[0] = nearest
    trace = Trace(
        entry_point=graph.entry_point,
        hops=hops,
        expansions=expansions,
        visited=visited,
        results=[(n, d) for d, n in ordered],
        ndis=ndis,
        ef_search=ef,
        k=k,
        max_level=graph.max_level,
        level_entry=level_entry,
    )

    ids = np.full((1, k), -1, dtype=np.int64)
    dists = np.zeros((1, k), dtype=np.float32)
    for i, (node, d) in enumerate(trace.results[:k]):
        ids[0, i] = node
        # faiss L2 indexes report squared distances; don't take the root.
        dists[0, i] = -d if is_ip else d
    return dists, ids, trace


# ----------------------------------------------------------------------
# verification
# ----------------------------------------------------------------------
def verify_against_faiss(index, query, k: int = 10, ef_search: int | None = None) -> Dict[str, object]:
    """Check the replay against the real search: same top-k, same ``ndis``."""
    _require_faiss()
    q = np.ascontiguousarray(np.asarray(query, dtype=np.float32).reshape(1, -1))
    if ef_search is not None:
        index.hnsw.efSearch = max(int(ef_search), k)

    faiss.cvar.hnsw_stats.reset()
    faiss_d, faiss_i = index.search(q, k)
    faiss_ndis = int(faiss.cvar.hnsw_stats.ndis)

    ours_d, ours_i, trace = search_with_trace(index, q, k=k, ef_search=index.hnsw.efSearch)
    return {
        "ids_match": bool(np.array_equal(faiss_i, ours_i)),
        "distances_match": bool(np.allclose(faiss_d, ours_d, atol=1e-5)),
        "ndis_match": trace.ndis == faiss_ndis,
        "faiss_ndis": faiss_ndis,
        "trace_ndis": trace.ndis,
        "faiss_ids": faiss_i[0].tolist(),
        "trace_ids": ours_i[0].tolist(),
    }
