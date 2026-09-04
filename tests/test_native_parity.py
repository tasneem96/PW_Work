"""Cross-checks against hnswlib, the deployed implementation."""

from __future__ import annotations

import numpy as np
import pytest

from braid.hnsw.native import (
    build_native,
    native_graph_of,
    native_search,
    native_version,
    parity_report,
)
from braid.hnsw.reference import build_index

pytestmark = pytest.mark.skipif(native_version() is None, reason="hnswlib not installed")


def test_recall_parity_is_within_the_declared_tolerance(small_dataset, small_params, protocol):
    spec = protocol.doc["hnsw"]["parity_tolerance"]
    report = parity_report(
        small_dataset.store,
        small_dataset.queries,
        small_params,
        ef_grid=[10, 50, 200],
        k=10,
        recall_tolerance=float(spec["tolerance"]),
        build_seeds=int(spec["build_seeds"]),
    )
    assert report["available"]
    assert report["passed"], report["cells"]


def test_parity_reports_seed_spread_next_to_the_gap(small_dataset, small_params):
    """A gap must be readable against the noise floor that produced it."""
    report = parity_report(
        small_dataset.store,
        small_dataset.queries,
        small_params,
        ef_grid=[10, 50],
        k=10,
        build_seeds=3,
    )
    assert len(report["build_seeds"]) == 3
    assert len(report["per_seed"]) == 3
    for cell in report["cells"]:
        assert cell["seed_spread_reference"] >= 0.0
        assert cell["seed_spread_native"] >= 0.0
        assert cell["reference_recall_range"][0] <= cell["reference_recall_mean"]
        assert cell["mean_recall_gap"] >= 0.0
        assert "gap_below_seed_spread" in cell


def test_layer0_degree_matches_hnswlib(small_dataset, small_params):
    """The check that caught the keepPrunedConnections mis-declaration."""
    reference = build_index(small_dataset.store, small_params)
    native = native_graph_of(build_native(small_dataset.store, small_params), small_dataset.dim)
    ref_degree = reference.stats()["per_layer"][0]["mean_degree"]
    nat_degree = native.stats()["per_layer"][0]["mean_degree"]
    assert abs(ref_degree - nat_degree) < 0.15 * nat_degree


def test_parsed_native_graph_is_structurally_sane(small_dataset, small_params):
    native = native_graph_of(build_native(small_dataset.store, small_params), small_dataset.dim)
    assert native.M == small_params.M
    assert native.max_M0 == small_params.m0
    assert len(native.links[0]) == small_dataset.n
    for node, neighbors in native.links[0].items():
        assert len(neighbors) <= native.max_M0
        assert node not in neighbors
        assert all(0 <= v < small_dataset.n for v in neighbors)


def test_native_search_uses_our_distance_convention(small_dataset, small_params):
    index = build_native(small_dataset.store, small_params)
    _ids, distances = native_search(
        index,
        small_dataset.store,
        small_dataset.queries[:4],
        k=5,
        ef_search=50,
        convention="cosine",
    )
    assert np.all(distances >= -1e-5) and np.all(distances <= 2.0 + 1e-5)
