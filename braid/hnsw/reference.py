"""An instrumented HNSW, faithful to Malkov & Yashunin [1] and to hnswlib's rules.

Why a reference implementation instead of instrumenting hnswlib directly:
Phase 1 needs the visited set, candidate expansions, per-layer neighbour lists,
stopping events, and the ability to search a *clean graph over corrupted
vectors* (the stale condition of Section 15). hnswlib exposes none of that, and
its cosine space normalizes stored copies at insert time, which would move the
attack surface off the raw stored coordinates the threat model targets.

The cost of that choice is that this implementation, not the deployed one, is
what the surrogate and the attacks are measured against. That is why
:mod:`braid.hnsw.native` cross-checks recall and graph statistics against
hnswlib on the same corpora, and why the Phase 8 falsification matrix must be
re-run on the native index before any claim leaves the lab.

Algorithm correspondence:
  build_index / _insert        Algorithm 1 (INSERT)
  _search_layer                Algorithm 2 (SEARCH-LAYER)
  _select_neighbors_heuristic  Algorithm 4 (SELECT-NEIGHBORS-HEURISTIC)
  search                       Algorithm 5 (K-NN-SEARCH)
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

from ..rng import generator
from ..vectors import VectorStore
from .oracle import DistanceOracle
from .params import HnswParams
from .trace import ExposurePolicy, QueryTrace, TraceLevel, TraceRecorder


@dataclass
class HnswGraph:
    """The graph G(D): layered adjacency, entry point, and build provenance.

    The graph holds no vectors. Which vectors a search compares against is
    supplied per search, which is what keeps the clean / stale / rebuilt
    conditions honest instead of three near-copies of the same code.
    """

    params: HnswParams
    n: int
    element_levels: np.ndarray
    links: list[dict[int, list[int]]]
    entry_point: int | None = None
    max_level: int = -1
    build_meta: dict[str, Any] = field(default_factory=dict)

    # -- structure -----------------------------------------------------------
    def neighbors(self, node: int, layer: int) -> list[int]:
        if layer < 0 or layer >= len(self.links):
            return []
        return self.links[layer].get(int(node), [])

    def degree(self, node: int, layer: int) -> int:
        return len(self.neighbors(node, layer))

    def layer_count(self) -> int:
        return len(self.links)

    def edge_count(self, layer: int | None = None) -> int:
        layers = range(self.layer_count()) if layer is None else [int(layer)]
        return sum(len(vs) for lc in layers for vs in self.links[lc].values())

    def nodes_at_layer(self, layer: int) -> list[int]:
        return sorted(self.links[int(layer)])

    def stats(self) -> dict[str, Any]:
        per_layer = []
        for layer in range(self.layer_count()):
            degrees = [len(v) for v in self.links[layer].values()] or [0]
            per_layer.append(
                {
                    "layer": layer,
                    "nodes": len(self.links[layer]),
                    "edges": int(sum(degrees)),
                    "mean_degree": float(np.mean(degrees)),
                    "max_degree": int(np.max(degrees)),
                    "degree_cap": self.params.max_degree(layer),
                }
            )
        return {
            "n": self.n,
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "layers": self.layer_count(),
            "edges": self.edge_count(),
            "per_layer": per_layer,
            "params": self.params.as_dict(),
            "build_meta": dict(self.build_meta),
        }

    def structure_hash(self) -> str:
        """Hash of the adjacency, for "is this the same graph" checks."""
        import hashlib

        h = hashlib.sha256()
        h.update(str(self.params.as_dict()).encode())
        h.update(str(self.entry_point).encode())
        for layer in range(self.layer_count()):
            for node in sorted(self.links[layer]):
                h.update(f"{layer}:{node}:{self.links[layer][node]}".encode())
        return h.hexdigest()

    def validate(self) -> list[str]:
        """Structural invariants that a correct HNSW build must satisfy."""
        problems: list[str] = []
        for layer in range(self.layer_count()):
            cap = self.params.max_degree(layer)
            for node, neighbors in self.links[layer].items():
                if len(neighbors) > cap:
                    problems.append(
                        f"node {node} at layer {layer} has degree {len(neighbors)} > cap {cap}"
                    )
                if len(set(neighbors)) != len(neighbors):
                    problems.append(f"node {node} at layer {layer} has duplicate neighbours")
                if node in neighbors:
                    problems.append(f"node {node} at layer {layer} links to itself")
                if self.element_levels[node] < layer:
                    problems.append(
                        f"node {node} appears at layer {layer} above its level "
                        f"{int(self.element_levels[node])}"
                    )
                for v in neighbors:
                    if self.element_levels[v] < layer:
                        problems.append(
                            f"edge {node}->{v} at layer {layer} points to a node whose level is "
                            f"{int(self.element_levels[v])}"
                        )
        if self.entry_point is None:
            problems.append("graph has no entry point")
        elif int(self.element_levels[self.entry_point]) != self.max_level:
            problems.append("entry point is not a node of maximum level")
        return problems


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------

def assign_levels(n: int, params: HnswParams) -> np.ndarray:
    """Element levels: floor(-ln(U) * mL), drawn from the frozen build seed."""
    rng = generator(params.seed, "hnsw-levels", f"M={params.M}", f"n={n}")
    u = rng.random(n)
    u = np.maximum(u, np.finfo(np.float64).tiny)
    return np.floor(-np.log(u) * params.mL).astype(np.int64)


def build_index(
    store: VectorStore,
    params: HnswParams,
    *,
    insertion_order: Sequence[int] | None = None,
    progress_every: int | None = None,
) -> HnswGraph:
    """Build G(D) by inserting every element in a deterministic order."""
    oracle = DistanceOracle(store, params.convention)
    levels = assign_levels(store.n, params)
    order = list(range(store.n)) if insertion_order is None else [int(i) for i in insertion_order]
    if sorted(order) != list(range(store.n)):
        raise ValueError("insertion_order must be a permutation of all element ids")

    max_level = int(levels.max()) if store.n else -1
    graph = HnswGraph(
        params=params,
        n=store.n,
        element_levels=levels,
        links=[{} for _ in range(max_level + 1)],
        entry_point=None,
        max_level=-1,
    )

    started = time.perf_counter_ns()
    for count, node in enumerate(order):
        _insert(graph, oracle, node)
        if progress_every and (count + 1) % progress_every == 0:
            print(f"  build {count + 1}/{store.n}", flush=True)
    elapsed = time.perf_counter_ns() - started

    graph.build_meta = {
        "store_label": store.label,
        "store_hash": store.content_hash(),
        "numeric_type": store.numeric_type,
        "convention": params.convention,
        "build_ns": int(elapsed),
        "insertion_order": "identity" if insertion_order is None else "explicit",
        "level_histogram": {
            str(level): int(count)
            for level, count in zip(*np.unique(levels, return_counts=True))
        },
    }
    return graph


def _insert(graph: HnswGraph, oracle: DistanceOracle, node: int) -> None:
    params = graph.params
    level = int(graph.element_levels[node])
    for layer in range(level + 1):
        graph.links[layer].setdefault(node, [])

    if graph.entry_point is None:
        graph.entry_point = node
        graph.max_level = level
        return

    query = oracle.prepared_row(node)
    entry = int(graph.entry_point)
    current = float(oracle.uncounted_distances(query, [entry])[0])

    # descend the layers above the new element's level, greedily
    for layer in range(graph.max_level, level, -1):
        entry, current = _greedy_descend(graph, oracle, query, entry, current, layer, None)

    entry_points = [entry]
    for layer in range(min(level, graph.max_level), -1, -1):
        candidates = _search_layer(
            graph, oracle, query, entry_points, params.ef_construction, layer, None, exclude=node
        )
        # The new element gets at most M links at every layer, including layer 0.
        # The larger max_M0 cap applies only when an existing node's list is
        # shrunk in _connect. hnswlib does the same; reading Algorithm 1 as
        # "select max_M0 neighbours at layer 0" roughly doubles layer-0
        # out-degree and makes the index easier to search than the deployed one.
        selected = _select_neighbors_heuristic(graph, oracle, node, candidates, params.M, layer)
        graph.links[layer][node] = list(selected)
        for neighbor in selected:
            _connect(graph, oracle, neighbor, node, layer)
        # hnswlib descends from the closest selected neighbour rather than from
        # the whole candidate set that Algorithm 1 passes down. Following
        # hnswlib keeps build breadth comparable to the deployed system.
        entry_points = [selected[0]] if selected else entry_points

    if level > graph.max_level:
        graph.max_level = level
        graph.entry_point = node


def _connect(graph: HnswGraph, oracle: DistanceOracle, node: int, new_neighbor: int, layer: int) -> None:
    """Add the reverse edge, shrinking with the heuristic when over capacity."""
    links = graph.links[layer].setdefault(node, [])
    if new_neighbor in links or new_neighbor == node:
        return
    cap = graph.params.max_degree(layer)
    if len(links) < cap:
        links.append(new_neighbor)
        return
    query = oracle.prepared_row(node)
    candidate_ids = links + [new_neighbor]
    distances = oracle.uncounted_distances(query, candidate_ids)
    candidates = sorted(zip((float(d) for d in distances), candidate_ids))
    kept = _select_neighbors_heuristic(graph, oracle, node, candidates, cap, layer)
    graph.links[layer][node] = list(kept)


def _select_neighbors_heuristic(
    graph: HnswGraph,
    oracle: DistanceOracle,
    base: int,
    candidates: Sequence[tuple[float, int]],
    m: int,
    layer: int,
) -> list[int]:
    """Algorithm 4, with hnswlib's defaults: no candidate extension, keep pruned."""
    params = graph.params
    working = sorted((float(d), int(v)) for d, v in candidates if int(v) != int(base))
    if params.extend_candidates:
        extended: dict[int, float] = {v: d for d, v in working}
        base_query = oracle.prepared_row(base)
        for _, v in list(working):
            for w in graph.neighbors(v, layer):
                if w != base and w not in extended:
                    extended[w] = float(oracle.uncounted_distances(base_query, [w])[0])
        working = sorted((d, v) for v, d in extended.items())
    if not working:
        return []

    candidate_ids = [v for _, v in working]
    among = oracle.pairwise(candidate_ids)
    position = {v: idx for idx, v in enumerate(candidate_ids)}

    kept: list[tuple[float, int]] = []
    pruned: list[tuple[float, int]] = []
    # best_to_kept[j] = min over already-kept r of d(candidate_j, r); updated in
    # one vectorized step per accepted neighbour, so the admission test is O(1).
    best_to_kept = np.full(len(candidate_ids), np.inf, dtype=np.float32)
    for dist_base_cand, cand in working:
        if len(kept) >= m:
            pruned.append((dist_base_cand, cand))
            continue
        pos = position[cand]
        if dist_base_cand < float(best_to_kept[pos]):
            kept.append((dist_base_cand, cand))
            best_to_kept = np.minimum(best_to_kept, among[:, pos])
        else:
            pruned.append((dist_base_cand, cand))

    if params.keep_pruned_connections:
        for item in pruned:
            if len(kept) >= m:
                break
            kept.append(item)
    return [v for _, v in kept]


# --------------------------------------------------------------------------
# search
# --------------------------------------------------------------------------

def _greedy_descend(
    graph: HnswGraph,
    oracle: DistanceOracle,
    query: np.ndarray,
    entry: int,
    current: float,
    layer: int,
    recorder: TraceRecorder | None,
) -> tuple[int, float]:
    """Greedy hill-descent on one upper layer (the loop inside Algorithm 5)."""
    if recorder is not None:
        recorder.event("layer_enter", layer=layer, node=entry, mode="greedy")
    changed = True
    while changed:
        changed = False
        neighbors = graph.neighbors(entry, layer)
        if recorder is not None:
            recorder.note_neighbor_list(entry, layer, neighbors)
        if not neighbors:
            break
        if recorder is not None:
            distances = oracle.distances(
                query, neighbors, recorder=recorder, layer=layer, context="greedy"
            )
        else:
            distances = oracle.uncounted_distances(query, neighbors)
        best = int(np.argmin(distances))
        if float(distances[best]) < current:
            if recorder is not None:
                recorder.note_greedy_hop(entry, int(neighbors[best]), layer)
            entry, current = int(neighbors[best]), float(distances[best])
            changed = True
    if recorder is not None:
        recorder.note_stop("greedy_local_minimum", layer=layer, node=entry)
        recorder.event("layer_exit", layer=layer, node=entry, mode="greedy")
    return entry, current


def _search_layer(
    graph: HnswGraph,
    oracle: DistanceOracle,
    query: np.ndarray,
    entry_points: Iterable[int],
    ef: int,
    layer: int,
    recorder: TraceRecorder | None,
    *,
    exclude: int | None = None,
) -> list[tuple[float, int]]:
    """Algorithm 2: best-first search of one layer, keeping the ef best results."""
    entries = [int(v) for v in dict.fromkeys(int(v) for v in entry_points) if v != exclude]
    if not entries:
        return []
    if recorder is not None:
        recorder.event("layer_enter", layer=layer, node=entries[0], mode="best_first", ef=ef)
        distances = oracle.distances(
            query, entries, recorder=recorder, layer=layer, context="entry"
        )
    else:
        distances = oracle.uncounted_distances(query, entries)

    visited: set[int] = set(entries)
    if exclude is not None:
        visited.add(int(exclude))
    candidates: list[tuple[float, int]] = []  # min-heap on distance
    results: list[tuple[float, int]] = []  # max-heap on distance, via negation
    for node, dist in zip(entries, (float(d) for d in distances)):
        heapq.heappush(candidates, (dist, node))
        heapq.heappush(results, (-dist, node))
        if recorder is not None:
            recorder.note_candidate_push(node, layer, dist)
    while len(results) > ef:
        dropped = heapq.heappop(results)
        if recorder is not None:
            recorder.note_prune(dropped[1], layer, -dropped[0])

    while candidates:
        dist_c, node_c = heapq.heappop(candidates)
        furthest = -results[0][0] if results else math.inf
        if len(results) >= ef and dist_c > furthest:
            if recorder is not None:
                recorder.note_stop(
                    "candidate_worse_than_furthest",
                    layer=layer,
                    node=node_c,
                    candidate_distance=dist_c,
                    furthest_result_distance=furthest,
                )
            break
        if recorder is not None:
            recorder.note_expansion(node_c, layer, dist_c)
        neighbors = graph.neighbors(node_c, layer)
        if recorder is not None:
            recorder.note_neighbor_list(node_c, layer, neighbors)
        fresh = [v for v in neighbors if v not in visited]
        if not fresh:
            continue
        visited.update(fresh)
        if recorder is not None:
            fresh_distances = oracle.distances(
                query, fresh, recorder=recorder, layer=layer, context="expansion"
            )
        else:
            fresh_distances = oracle.uncounted_distances(query, fresh)
        for node_v, dist_v in zip(fresh, (float(d) for d in fresh_distances)):
            furthest = -results[0][0] if results else math.inf
            if len(results) < ef or dist_v < furthest:
                heapq.heappush(candidates, (dist_v, node_v))
                heapq.heappush(results, (-dist_v, node_v))
                if recorder is not None:
                    recorder.note_candidate_push(node_v, layer, dist_v)
                if len(results) > ef:
                    dropped = heapq.heappop(results)
                    if recorder is not None:
                        recorder.note_prune(dropped[1], layer, -dropped[0])
            elif recorder is not None:
                recorder.note_prune(node_v, layer, dist_v)
    else:
        if recorder is not None:
            recorder.note_stop("candidate_queue_empty", layer=layer)

    if recorder is not None:
        recorder.event("layer_exit", layer=layer, node=None, mode="best_first")
    return sorted((-d, v) for d, v in results)


@dataclass(frozen=True, eq=False)
class SearchResult:
    ids: np.ndarray
    distances: np.ndarray
    trace: QueryTrace | None = None


def search(
    graph: HnswGraph,
    store: VectorStore,
    query: np.ndarray,
    *,
    k: int = 10,
    ef_search: int = 50,
    trace_level: TraceLevel = TraceLevel.COUNTERS,
    exposure: ExposurePolicy | None = None,
    query_index: int = -1,
    oracle: DistanceOracle | None = None,
) -> SearchResult:
    """Algorithm 5 over ``graph`` with distances taken from ``store``.

    ``store`` is a parameter rather than a property of the graph so that the
    clean, stale, and rebuilt conditions of Section 15 all run this one code
    path: clean is (G(D), D), stale is (G(D), D'), rebuilt is (G(D'), D').
    """
    if graph.entry_point is None:
        raise ValueError("cannot search an empty graph")
    if store.n != graph.n:
        raise ValueError(
            f"graph was built over {graph.n} elements but the store holds {store.n}; the stale "
            f"condition requires the same element set with different values"
        )
    oracle = oracle or DistanceOracle(store, graph.params.convention)
    recorder = TraceRecorder(query_index=query_index, level=trace_level, exposure=exposure)
    active = recorder if recorder.enabled else None

    started = time.perf_counter_ns()
    prepared = oracle.prepare_query(query)
    if active is not None:
        active.event("query_start", layer=graph.max_level, ef_search=ef_search, k=k)

    entry = int(graph.entry_point)
    if active is not None:
        entry_distance = float(
            oracle.distances(
                prepared, [entry], recorder=active, layer=graph.max_level, context="entry_point"
            )[0]
        )
        active.note_entry(entry, graph.max_level, entry_distance)
    else:
        entry_distance = float(oracle.uncounted_distances(prepared, [entry])[0])

    current = entry_distance
    for layer in range(graph.max_level, 0, -1):
        entry, current = _greedy_descend(graph, oracle, prepared, entry, current, layer, active)

    ef = max(int(ef_search), int(k))
    results = _search_layer(graph, oracle, prepared, [entry], ef, 0, active)
    top = results[: int(k)]
    ids = np.array([v for _, v in top], dtype=np.int64)
    distances = np.array([d for d, _ in top], dtype=np.float32)
    latency = time.perf_counter_ns() - started

    trace = recorder.finish(ids, distances, latency) if recorder.enabled else None
    return SearchResult(ids=ids, distances=distances, trace=trace)


def search_many(
    graph: HnswGraph,
    store: VectorStore,
    queries: np.ndarray,
    *,
    k: int = 10,
    ef_search: int = 50,
    trace_level: TraceLevel = TraceLevel.COUNTERS,
    exposure: ExposurePolicy | None = None,
    query_ids: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[QueryTrace]]:
    """Search a batch, returning ids, distances, and one trace per query."""
    q = np.ascontiguousarray(queries, dtype=np.float32)
    if q.ndim == 1:
        q = q[None, :]
    oracle = DistanceOracle(store, graph.params.convention)
    ids = np.full((q.shape[0], int(k)), -1, dtype=np.int64)
    distances = np.full((q.shape[0], int(k)), np.nan, dtype=np.float32)
    traces: list[QueryTrace] = []
    for row in range(q.shape[0]):
        query_index = int(query_ids[row]) if query_ids is not None else row
        result = search(
            graph,
            store,
            q[row],
            k=k,
            ef_search=ef_search,
            trace_level=trace_level,
            exposure=exposure,
            query_index=query_index,
            oracle=oracle,
        )
        found = result.ids.size
        ids[row, :found] = result.ids
        distances[row, :found] = result.distances
        if result.trace is not None:
            traces.append(result.trace)
    return ids, distances, traces
