"""Phase 1 gate: exact answers must be independently checked."""

from __future__ import annotations

import numpy as np
import pytest

from braid.exact import cross_check_exact, exact_topk, exact_topk_native, exact_topk_reference
from braid.vectors import make_store


@pytest.mark.parametrize("convention", ["cosine", "l2"])
@pytest.mark.parametrize("numeric_type", ["fp32", "fp16"])
def test_three_implementations_agree(convention, numeric_type):
    rng = np.random.default_rng(5)
    store = make_store(rng.normal(size=(400, 24)), numeric_type)
    queries = rng.normal(size=(16, 24)).astype(np.float32)
    report = cross_check_exact(queries, store, 10, convention)
    assert report["passed"], report
    assert report["comparisons"]["reference_loop"]["available"]


def test_top1_matches_a_hand_computed_argmax():
    store = make_store(
        np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]], dtype=np.float32), "fp32"
    )
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    result = exact_topk(query, store, 3, "cosine")
    assert int(result.nearest[0]) == 0
    assert result.ids[0].tolist() == [0, 1, 2]
    assert result.scores[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_ties_are_reported_not_hidden():
    store = make_store(
        np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=np.float32), "fp32"
    )
    query = np.array([[1.0, 0.0]], dtype=np.float32)
    result = exact_topk(query, store, 2, "cosine")
    assert bool(result.tie_mask()[0])
    # deterministic tie-break by smaller id
    assert result.ids[0].tolist() == [0, 1]


def test_reference_loop_and_matrix_paths_are_not_the_same_code():
    rng = np.random.default_rng(7)
    store = make_store(rng.normal(size=(120, 8)), "fp32")
    queries = rng.normal(size=(5, 8)).astype(np.float32)
    matrix = exact_topk(queries, store, 5, "cosine")
    loop = exact_topk_reference(queries, store, 5, "cosine")
    assert np.array_equal(matrix.ids, loop.ids)
    assert np.allclose(matrix.scores, loop.scores, atol=1e-5)


def test_native_brute_force_uses_our_cosine_convention():
    rng = np.random.default_rng(9)
    store = make_store(rng.normal(size=(200, 12)), "fp32")
    queries = rng.normal(size=(8, 12)).astype(np.float32)
    native = exact_topk_native(queries, store, 5, "cosine")
    if native is None:
        pytest.skip("hnswlib not installed")
    matrix = exact_topk(queries, store, 5, "cosine")
    assert np.allclose(np.sort(native.scores, axis=1), np.sort(matrix.scores, axis=1), atol=1e-5)
    assert float(native.scores.max()) <= 1.0 + 1e-5
