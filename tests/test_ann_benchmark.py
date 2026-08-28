"""Tests for the HNSW / ANN-Benchmarks path.

Uses a small dataset built the same way as glove-25-angular (25-d, cosine
ground truth, widely varying row norms) so that forgetting to normalize is a
visible failure rather than a rounding difference.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

faiss = pytest.importorskip("faiss")

from glove_retrieval.ann_benchmark import (  # noqa: E402
    as_float32,
    build_hnsw,
    detect_convention_for_results,
    detect_distance_convention,
    faiss_index,
    load_dataset,
    normalized,
    recall_at_k,
)

D_DIM, N_TRAIN, N_TEST, N_GT = 25, 8_000, 100, 100


def make_vectors(rng, n):
    """Clustered directions with magnitudes spread over ~0.1-10x, like GloVe."""
    centroids = rng.normal(0, 1.0, (30, D_DIM))
    directions = centroids[rng.integers(0, 30, n)] + rng.normal(0, 0.45, (n, D_DIM))
    return (directions * np.exp(rng.normal(0.0, 0.8, (n, 1)))).astype(np.float32)


@pytest.fixture(scope="module")
def dataset():
    rng = np.random.default_rng(20260828)
    train, test = make_vectors(rng, N_TRAIN), make_vectors(rng, N_TEST)
    a = np.asarray(train, np.float64) / np.linalg.norm(train, axis=1, keepdims=True)
    b = np.asarray(test, np.float64) / np.linalg.norm(test, axis=1, keepdims=True)
    cosine = b @ a.T
    neighbors = np.argsort(-cosine, axis=1, kind="stable")[:, :N_GT].astype(np.int32)
    distances = (1.0 - np.take_along_axis(cosine, neighbors, axis=1)).astype(np.float32)
    return {"train": train, "test": test, "neighbors": neighbors, "distances": distances}


# ----------------------------------------------------------------------
# the function under test
# ----------------------------------------------------------------------
def test_returns_faiss_shaped_results(dataset):
    D, I = faiss_index(dataset["train"], dataset["test"], k=10)
    assert D.shape == I.shape == (N_TEST, 10)
    assert I.dtype == np.int64 and D.dtype == np.float32
    assert (I >= 0).all() and (I < N_TRAIN).all()


def test_recall_against_exact_ground_truth(dataset):
    """The point of the exercise: HNSW must actually find the right neighbours."""
    D, I = faiss_index(dataset["train"], dataset["test"], k=10, ef_search=256)
    assert recall_at_k(I, dataset["neighbors"], 10) > 0.97
    assert recall_at_k(I, dataset["neighbors"], 1) > 0.97


def test_angular_scores_are_cosine_similarities(dataset):
    D, I = faiss_index(dataset["train"], dataset["test"], k=5)
    unit_train = normalized(dataset["train"])
    unit_test = normalized(dataset["test"])
    expected = np.einsum("ij,ij->i", unit_test, unit_train[I[:, 0]])
    assert np.allclose(D[:, 0], expected, atol=1e-5)
    assert (D <= 1.0 + 1e-5).all() and (D >= -1.0 - 1e-5).all()
    # scores are similarities: non-increasing across each row
    assert (np.diff(D, axis=1) <= 1e-6).all()


def test_ignoring_angular_wrecks_recall(dataset):
    """glove-*-angular is cosine ground truth; raw L2 on unnormalized rows is not."""
    _, angular = faiss_index(dataset["train"], dataset["test"], k=10, ef_search=256)
    _, l2 = faiss_index(dataset["train"], dataset["test"], k=10, ef_search=256, metric="euclidean")
    angular_recall = recall_at_k(angular, dataset["neighbors"], 10)
    l2_recall = recall_at_k(l2, dataset["neighbors"], 10)
    assert angular_recall > 0.97
    assert l2_recall < 0.6, "test data is too unit-norm to catch a missing normalization"


def test_caller_arrays_are_never_modified(dataset):
    """faiss.normalize_L2 works in place; xb and xq must survive untouched."""
    train, test = dataset["train"].copy(), dataset["test"].copy()
    before_train, before_test = train.copy(), test.copy()
    faiss_index(train, test, k=5)
    assert np.array_equal(train, before_train)
    assert np.array_equal(test, before_test)


def test_ef_search_is_never_below_k(dataset):
    """HNSW defaults efSearch to 16; searching k=100 with that collapses recall."""
    _, _, run = faiss_index(dataset["train"], dataset["test"], k=100, ef_search=8, stats=True)
    assert run.ef_search == 100
    _, _, default_run = faiss_index(dataset["train"], dataset["test"], k=100, stats=True)
    assert default_run.ef_search == 200  # max(2k, 64)
    _, _, small_k = faiss_index(dataset["train"], dataset["test"], k=1, stats=True)
    assert small_k.ef_search == 64


def test_higher_ef_search_improves_recall(dataset):
    index = build_hnsw(dataset["train"], M=32, ef_construction=200)
    queries = normalized(dataset["test"])
    recalls = []
    for ef in (16, 64, 256):
        index.hnsw.efSearch = ef
        _, I = index.search(queries, 10)
        recalls.append(recall_at_k(I, dataset["neighbors"], 10))
    assert recalls == sorted(recalls), f"recall should not fall as efSearch grows: {recalls}"
    assert recalls[-1] > recalls[0]


def test_stats_are_reported(dataset):
    D, I, run = faiss_index(dataset["train"], dataset["test"], k=10, M=16, stats=True)
    assert run.ntotal == N_TRAIN and run.M == 16 and run.metric == "angular"
    assert run.build_seconds > 0 and run.query_seconds > 0
    assert run.queries_per_second == pytest.approx(N_TEST / run.query_seconds, rel=1e-6)


# ----------------------------------------------------------------------
# input handling
# ----------------------------------------------------------------------
def test_accepts_float64_and_non_contiguous_input(dataset):
    train64 = dataset["train"].astype(np.float64)
    sliced = np.asfortranarray(dataset["test"])[:50]
    D, I = faiss_index(train64, sliced, k=5)
    assert I.shape == (50, 5)


def test_dimension_mismatch_is_rejected(dataset):
    with pytest.raises(ValueError, match="24-d.*25-d"):
        faiss_index(dataset["train"], dataset["test"][:, :24], k=5)


def test_k_larger_than_the_database_is_rejected(dataset):
    with pytest.raises(ValueError, match="exceeds"):
        faiss_index(dataset["train"][:10], dataset["test"], k=50)


def test_bad_arguments(dataset):
    with pytest.raises(ValueError, match="k must be positive"):
        faiss_index(dataset["train"], dataset["test"], k=0)
    with pytest.raises(ValueError, match="unknown metric"):
        faiss_index(dataset["train"], dataset["test"], metric="hamming")


def test_normalized_copies_and_produces_unit_rows(dataset):
    original = dataset["train"][:100].copy()
    unit = normalized(original)
    assert np.allclose(np.linalg.norm(unit, axis=1), 1.0, atol=1e-6)
    assert np.array_equal(original, dataset["train"][:100]), "must not normalize in place"
    assert as_float32(original).dtype == np.float32


# ----------------------------------------------------------------------
# recall
# ----------------------------------------------------------------------
def test_recall_at_k_arithmetic():
    truth = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    assert recall_at_k(truth, truth, 4) == 1.0
    assert recall_at_k(np.array([[1, 2, 9, 9], [5, 6, 7, 8]]), truth, 4) == 0.75
    assert recall_at_k(np.array([[9, 9, 9, 9], [9, 9, 9, 9]]), truth, 4) == 0.0
    # order within the top-k does not matter, membership does
    assert recall_at_k(np.array([[4, 3, 2, 1], [8, 7, 6, 5]]), truth, 4) == 1.0
    # k defaults to the narrower of the two
    assert recall_at_k(truth[:, :2], truth) == 1.0


def test_recall_at_k_rejects_mismatched_input():
    truth = np.array([[1, 2], [3, 4]])
    with pytest.raises(ValueError, match="rows"):
        recall_at_k(np.array([[1, 2]]), truth)
    with pytest.raises(ValueError, match="recall@5"):
        recall_at_k(truth, truth, 5)


# ----------------------------------------------------------------------
# distance conventions
# ----------------------------------------------------------------------
def test_detects_each_known_convention():
    cos = np.linspace(-0.9, 0.9, 50)
    assert detect_distance_convention(cos, 1.0 - cos) == "1 - cosine"
    assert detect_distance_convention(cos, np.sqrt(2 - 2 * cos)) == "euclidean_on_unit"
    assert detect_distance_convention(cos, np.arccos(cos)) == "arccos"
    assert detect_distance_convention(cos, cos) == "cosine_similarity"
    assert detect_distance_convention(cos, cos * 3.7 + 1.0) is None


def test_convention_detection_masks_approximate_misses(dataset):
    """Elementwise comparison against an ANN result compares different neighbours."""
    D, I = faiss_index(dataset["train"], dataset["test"], k=100, ef_search=64)
    gt, gtd = dataset["neighbors"], dataset["distances"]
    assert not np.array_equal(I, gt), "fixture must be approximate for this test to mean anything"
    assert detect_convention_for_results(I, D, gt, gtd) == "1 - cosine"
    # the unmasked comparison is exactly the trap this guards against
    assert detect_distance_convention(D, gtd) is None


# ----------------------------------------------------------------------
# HDF5 round trip
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def hdf5_file(tmp_path_factory):
    pytest.importorskip("h5py")
    out = tmp_path_factory.mktemp("ann") / "sample-25-angular.hdf5"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "make_sample_ann_dataset.py"), str(out)],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return out


def test_load_dataset_round_trip(hdf5_file):
    data = load_dataset(hdf5_file)
    assert data["train"].shape[1] == 25 and data["train"].dtype == np.float32
    assert data["neighbors"].shape[1] == 100
    assert data["metric"] == "angular"
    D, I = faiss_index(data["train"], data["test"], k=10, ef_search=256)
    assert recall_at_k(I, data["neighbors"], 10) > 0.97


def test_load_dataset_errors(tmp_path):
    pytest.importorskip("h5py")
    import h5py

    with pytest.raises(FileNotFoundError, match="ann-benchmarks"):
        load_dataset(tmp_path / "missing.hdf5")
    bad = tmp_path / "bad.hdf5"
    with h5py.File(bad, "w") as f:
        f.create_dataset("train", data=np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="missing dataset"):
        load_dataset(bad)


def test_bench_script_runs(hdf5_file):
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bench_hnsw.py"), str(hdf5_file),
         "-k", "10", "--ef", "16", "256", "--queries", "100"],
        capture_output=True, text=True, cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr
    assert "recall@10" in proc.stdout
    assert "1 - cosine" in proc.stdout
