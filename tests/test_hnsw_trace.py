"""Tests for the HNSW traversal replay.

The replay is only useful if it is faiss's actual traversal, so most of these
compare against the real search: same top-k, same distances, and -- the strict
one -- the same number of distance computations as hnsw_stats.ndis.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

faiss = pytest.importorskip("faiss")

from glove_retrieval.hnsw_trace import (  # noqa: E402
    HnswGraph,
    search_with_trace,
    verify_against_faiss,
)

D_DIM, N = 25, 5_000


def build(metric=faiss.METRIC_INNER_PRODUCT, M=16, seed=0):
    rng = np.random.default_rng(seed)
    xb = rng.normal(size=(N, D_DIM)).astype(np.float32)
    if metric == faiss.METRIC_INNER_PRODUCT:
        faiss.normalize_L2(xb)
    index = faiss.IndexHNSWFlat(D_DIM, M, metric)
    index.hnsw.efConstruction = 200
    index.add(xb)
    return index, xb


@pytest.fixture(scope="module")
def ip_index():
    return build()


@pytest.fixture(scope="module")
def l2_index():
    return build(metric=faiss.METRIC_L2, M=32, seed=1)


def queries(index, n, seed=99):
    rng = np.random.default_rng(seed)
    out = rng.normal(size=(n, D_DIM)).astype(np.float32)
    if index.metric_type == faiss.METRIC_INNER_PRODUCT:
        out /= np.linalg.norm(out, axis=1, keepdims=True)
    return out


# ----------------------------------------------------------------------
# the thing that matters: is this faiss's traversal?
# ----------------------------------------------------------------------
@pytest.mark.parametrize("ef", [16, 64, 256])
@pytest.mark.parametrize("k", [1, 10, 100])
def test_matches_faiss_on_cosine(ip_index, ef, k):
    index, _ = ip_index
    for q in queries(index, 8):
        report = verify_against_faiss(index, q, k=k, ef_search=ef)
        assert report["ids_match"], report
        assert report["distances_match"], report
        assert report["ndis_match"], report


@pytest.mark.parametrize("ef,k", [(16, 10), (64, 10), (256, 100)])
def test_matches_faiss_on_l2(l2_index, ef, k):
    index, _ = l2_index
    for q in queries(index, 8, seed=7):
        report = verify_against_faiss(index, q, k=k, ef_search=ef)
        assert report["ids_match"], report
        assert report["distances_match"], report
        assert report["ndis_match"], report


def test_l2_distances_are_squared(l2_index):
    """faiss L2 indexes report squared distances; the replay must not sqrt them."""
    index, xb = l2_index
    q = xb[3].copy()
    D, I, _ = search_with_trace(index, q, k=1, ef_search=64)
    exact = float(((xb[I[0][0]] - q) ** 2).sum())
    assert D[0][0] == pytest.approx(exact, abs=1e-4)


def test_ip_distances_are_cosine_similarity(ip_index):
    index, xb = ip_index
    q = xb[11].copy()
    D, I, _ = search_with_trace(index, q, k=3, ef_search=64)
    assert D[0][0] == pytest.approx(1.0, abs=1e-4)      # the query is in the index
    assert I[0][0] == 11
    assert float(xb[I[0][1]] @ q) == pytest.approx(D[0][1], abs=1e-5)


# ----------------------------------------------------------------------
# the trace itself
# ----------------------------------------------------------------------
def test_trace_structure(ip_index):
    index, _ = ip_index
    q = queries(index, 1)[0]
    _, I, trace = search_with_trace(index, q, k=10, ef_search=64)

    assert trace.entry_point == index.hnsw.entry_point
    assert trace.max_level == index.hnsw.max_level
    assert trace.greedy_path[0] == trace.entry_point
    assert trace.ef_search == 64 and trace.k == 10
    assert len(trace.expansions) > 0
    # everything returned was visited, and every expansion was too
    assert {int(i) for i in I[0] if i >= 0} <= set(trace.visited)
    assert set(trace.expanded_nodes) <= set(trace.visited)
    # Every visited node cost at least one distance computation, except the
    # entry point, which faiss does not count. ndis can exceed the number of
    # distinct nodes: the greedy descent re-evaluates neighbours it already
    # saw on a higher level.
    assert len(trace.visited) <= trace.ndis + 1
    assert trace.ndis >= len(trace.expansions)


def test_greedy_descent_strictly_improves(ip_index):
    """A greedy hop is only recorded when it finds something closer."""
    index, _ = ip_index
    for q in queries(index, 5, seed=3):
        _, _, trace = search_with_trace(index, q, k=10, ef_search=64)
        by_level = {}
        for hop in trace.hops:
            by_level.setdefault(hop.level, []).append(hop.distance)
        for level, dists in by_level.items():
            assert dists == sorted(dists, reverse=False) or all(
                a > b for a, b in zip(dists, dists[1:])
            ), f"level {level}: {dists}"
        # levels are walked from the top down, never revisited
        levels = [h.level for h in trace.hops]
        assert levels == sorted(levels, reverse=True)


def test_path_at_level_covers_every_level(ip_index):
    index, _ = ip_index
    q = queries(index, 1, seed=5)[0]
    _, _, trace = search_with_trace(index, q, k=10, ef_search=64)
    for level in range(trace.max_level, -1, -1):
        path = trace.path_at_level(level)
        assert path, f"level {level} has no entry node"
        assert path[0] == trace.level_entry[level]
    # the walk is continuous: each level starts where the one above ended
    for level in range(trace.max_level, 0, -1):
        assert trace.path_at_level(level)[-1] == trace.path_at_level(level - 1)[0]
    assert trace.path_at_level(trace.max_level)[0] == trace.entry_point


def test_expansions_record_what_they_discovered(ip_index):
    index, _ = ip_index
    q = queries(index, 1, seed=8)[0]
    _, _, trace = search_with_trace(index, q, k=10, ef_search=64)
    discovered = [n for e in trace.expansions for n in e.discovered]
    assert len(discovered) == len(set(discovered)), "a node is discovered only once"
    # every distance computed at layer 0 belongs to exactly one discovery
    upper = sum(len(HnswGraph(index).neighbors_of(h.source, h.level)) for h in trace.hops)
    assert len(discovered) <= trace.ndis
    assert upper <= trace.ndis
    assert [e.step for e in trace.expansions] == list(range(len(trace.expansions)))


def test_ef_search_is_floored_at_k(ip_index):
    index, _ = ip_index
    q = queries(index, 1)[0]
    _, _, trace = search_with_trace(index, q, k=50, ef_search=8)
    assert trace.ef_search == 50


def test_larger_ef_visits_more(ip_index):
    index, _ = ip_index
    q = queries(index, 1, seed=12)[0]
    small = search_with_trace(index, q, k=10, ef_search=16)[2]
    large = search_with_trace(index, q, k=10, ef_search=128)[2]
    assert large.ndis > small.ndis
    assert len(large.expansions) > len(small.expansions)


def test_neighbors_stop_at_the_first_negative(ip_index):
    """faiss pads a node's neighbour slots with -1 and breaks on the first one."""
    index, _ = ip_index
    graph = HnswGraph(index)
    for node in range(200):
        nb = graph.neighbors_of(node, 0)
        assert (nb >= 0).all()
        assert len(nb) <= index.hnsw.nb_neighbors(0)


def test_summary_is_printable(ip_index):
    index, _ = ip_index
    _, _, trace = search_with_trace(index, queries(index, 1)[0], k=5, ef_search=32)
    text = trace.summary()
    assert str(trace.entry_point) in text and "efSearch=32" in text
