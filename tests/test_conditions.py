"""Section 15's conditions, in the D' = D configuration Phase 1 can check."""

from __future__ import annotations

import numpy as np

from braid.hnsw.conditions import CONDITIONS, evaluate_conditions, identity_check
from braid.hnsw.reference import build_index


def test_all_four_conditions_are_produced(small_dataset, small_params):
    results = evaluate_conditions(
        clean_store=small_dataset.store,
        queries=small_dataset.queries[:8],
        params=small_params,
        k=10,
        ef_search=50,
    )
    assert set(results) == set(CONDITIONS)


def test_stale_equals_clean_when_the_vectors_are_unchanged(small_dataset, small_params):
    graph = build_index(small_dataset.store, small_params)
    results = evaluate_conditions(
        clean_store=small_dataset.store,
        queries=small_dataset.queries[:8],
        params=small_params,
        k=10,
        ef_search=50,
        graph_clean=graph,
        graph_rebuilt=graph,
    )
    check = identity_check(results)
    assert check["passed"], check
    assert np.array_equal(results["hnsw_clean"].ids, results["hnsw_stale"].ids)


def test_stale_uses_the_corrupted_vectors_on_the_clean_graph(small_dataset, small_params):
    """A hand-made perturbation stands in for a real bit flip (Phase 2)."""
    graph = build_index(small_dataset.store, small_params)
    perturbed = small_dataset.store.as_f32().copy()
    rng = np.random.default_rng(3)
    victims = rng.choice(perturbed.shape[0], size=40, replace=False)
    perturbed[victims] *= -1.0  # a sign flip is the crudest possible corruption
    corrupted = small_dataset.store.with_data(perturbed, "D'")

    results = evaluate_conditions(
        clean_store=small_dataset.store,
        queries=small_dataset.queries[:16],
        params=small_params,
        k=10,
        ef_search=50,
        corrupted_store=corrupted,
        graph_clean=graph,
    )
    assert not np.array_equal(results["hnsw_clean"].ids, results["hnsw_stale"].ids)
    # the stale condition must have searched the clean graph
    assert results["hnsw_stale"].meta["graph_hash"] == graph.structure_hash()
    assert results["hnsw_stale"].meta["store_hash"] == corrupted.content_hash()
    # and the rebuilt condition must have built a different graph
    assert results["hnsw_rebuilt"].meta["graph_hash"] != graph.structure_hash()


def test_a_corrupted_store_must_keep_the_same_shape(small_dataset, small_params):
    import pytest

    truncated = small_dataset.store.with_data(small_dataset.store.as_f32()[:10], "D'")
    with pytest.raises(ValueError):
        evaluate_conditions(
            clean_store=small_dataset.store,
            queries=small_dataset.queries[:2],
            params=small_params,
            corrupted_store=truncated,
        )


def test_work_counters_are_available_per_condition(small_dataset, small_params):
    results = evaluate_conditions(
        clean_store=small_dataset.store,
        queries=small_dataset.queries[:8],
        params=small_params,
        k=10,
        ef_search=50,
    )
    frame = results["hnsw_stale"].work_frame()
    assert set(frame) == {"distance_evals", "expansions", "unique_visited", "latency_ns"}
    assert frame["distance_evals"].size == 8
    assert results["exact"].work_frame() == {}
