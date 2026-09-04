"""The instrumented HNSW: structure, determinism, and search behaviour."""

from __future__ import annotations

import numpy as np
import pytest

from braid.exact import exact_topk
from braid.hnsw.params import HnswParams
from braid.hnsw.reference import assign_levels, build_index, search, search_many
from braid.hnsw.trace import TraceLevel
from braid.metrics import recall_at_k
from braid.vectors import make_store


def test_graph_satisfies_its_structural_invariants(small_dataset, small_params):
    graph = build_index(small_dataset.store, small_params)
    assert graph.validate() == []
    assert graph.entry_point is not None
    assert graph.n == small_dataset.n


def test_degree_caps_are_respected(small_dataset, small_params):
    graph = build_index(small_dataset.store, small_params)
    for layer in range(graph.layer_count()):
        cap = small_params.max_degree(layer)
        assert max((len(v) for v in graph.links[layer].values()), default=0) <= cap


def test_build_is_a_function_of_the_seed(small_dataset, small_params):
    first = build_index(small_dataset.store, small_params)
    second = build_index(small_dataset.store, small_params)
    assert first.structure_hash() == second.structure_hash()
    other = build_index(
        small_dataset.store,
        HnswParams(
            M=small_params.M,
            ef_construction=small_params.ef_construction,
            seed=small_params.seed + 1,
            convention=small_params.convention,
        ),
    )
    assert other.structure_hash() != first.structure_hash()


def test_level_assignment_follows_the_declared_rule():
    params = HnswParams(M=16, seed=3)
    levels = assign_levels(20000, params)
    # geometric-like decay: each level should hold roughly 1/M of the one below
    counts = np.bincount(levels)
    assert counts[0] > counts[1] > counts[2]
    ratio = counts[1] / counts[0]
    assert 0.02 < ratio < 0.12


def test_recall_improves_with_ef_and_reaches_exact(small_dataset, small_params):
    graph = build_index(small_dataset.store, small_params)
    truth = [
        set(int(i) for i in row)
        for row in exact_topk(small_dataset.queries, small_dataset.store, 10, "cosine").ids
    ]
    recalls = []
    for ef in (10, 50, 200):
        ids, _, _ = search_many(
            graph, small_dataset.store, small_dataset.queries, k=10, ef_search=ef
        )
        recalls.append(recall_at_k(truth, ids, 10))
    assert recalls == sorted(recalls)
    assert recalls[-1] > 0.95


def test_search_returns_sorted_distances(small_dataset, small_params):
    graph = build_index(small_dataset.store, small_params)
    result = search(graph, small_dataset.store, small_dataset.queries[0], k=10, ef_search=50)
    assert result.ids.size == 10
    assert np.all(np.diff(result.distances) >= -1e-6)


def test_ef_search_is_raised_to_k(small_dataset, small_params):
    graph = build_index(small_dataset.store, small_params)
    result = search(graph, small_dataset.store, small_dataset.queries[0], k=10, ef_search=1)
    assert result.ids.size == 10


def test_truncating_a_larger_k_matches_searching_at_that_k(small_dataset, small_params):
    """The sweep relies on this to avoid re-searching per k."""
    graph = build_index(small_dataset.store, small_params)
    big, _, _ = search_many(
        graph, small_dataset.store, small_dataset.queries, k=10, ef_search=50
    )
    small, _, _ = search_many(
        graph, small_dataset.store, small_dataset.queries, k=1, ef_search=50
    )
    assert np.array_equal(big[:, :1], small)


def test_searching_a_store_of_the_wrong_size_is_refused(small_dataset, small_params):
    graph = build_index(small_dataset.store, small_params)
    smaller = make_store(small_dataset.store.as_f32()[:10], "fp32")
    with pytest.raises(ValueError):
        search(graph, smaller, small_dataset.queries[0], k=1, ef_search=10)


def test_l2_convention_also_builds_and_searches(small_dataset):
    params = HnswParams(M=8, ef_construction=50, seed=1, convention="l2")
    graph = build_index(small_dataset.store, params)
    assert graph.validate() == []
    truth = [
        set(int(i) for i in row)
        for row in exact_topk(small_dataset.queries, small_dataset.store, 5, "l2").ids
    ]
    ids, _, _ = search_many(
        graph, small_dataset.store, small_dataset.queries, k=5, ef_search=100
    )
    assert recall_at_k(truth, ids, 5) > 0.9


def test_trace_level_does_not_change_the_answer(small_dataset, small_params):
    graph = build_index(small_dataset.store, small_params)
    answers = []
    for level in (TraceLevel.NONE, TraceLevel.COUNTERS, TraceLevel.FULL):
        ids, _, _ = search_many(
            graph,
            small_dataset.store,
            small_dataset.queries[:8],
            k=10,
            ef_search=50,
            trace_level=level,
        )
        answers.append(ids)
    assert np.array_equal(answers[0], answers[1])
    assert np.array_equal(answers[1], answers[2])
