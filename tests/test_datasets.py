"""Corpora are reproducible from the seed, and unavailable data fails loudly."""

from __future__ import annotations

import numpy as np
import pytest

from braid.datasets import DatasetUnavailable, load_dataset


def test_synthetic_corpus_is_reproducible(protocol):
    a = load_dataset(protocol, "syn-clusters-d64", n=300, n_queries=16)
    b = load_dataset(protocol, "syn-clusters-d64", n=300, n_queries=16)
    assert a.store.content_hash() == b.store.content_hash()
    assert np.array_equal(a.queries, b.queries)


def test_numeric_type_changes_stored_bytes_but_not_shape(protocol):
    f32 = load_dataset(protocol, "syn-clusters-d64", numeric_type="fp32", n=200, n_queries=8)
    f16 = load_dataset(protocol, "syn-clusters-d64", numeric_type="fp16", n=200, n_queries=8)
    assert f32.store.bit_width == 32 and f16.store.bit_width == 16
    assert f32.store.data.shape == f16.store.data.shape
    assert f32.store.content_hash() != f16.store.content_hash()
    assert np.allclose(f32.store.as_f32(), f16.store.as_f32(), atol=1e-2)


def test_a_run_may_subset_but_not_exceed_the_declared_size(protocol):
    declared = int(protocol.dataset("syn-clusters-d64")["n"])
    with pytest.raises(ValueError):
        load_dataset(protocol, "syn-clusters-d64", n=declared + 1, n_queries=8)


def test_undeclared_numeric_type_is_refused(protocol):
    with pytest.raises(ValueError):
        load_dataset(protocol, "syn-clusters-d64", numeric_type="fp8", n=100, n_queries=4)


def test_external_dataset_reports_what_it_needs(protocol):
    with pytest.raises(DatasetUnavailable) as excinfo:
        load_dataset(protocol, "sift-1m")
    message = str(excinfo.value)
    assert "corpus.npy" in message and "sha256" in message


def test_queries_are_near_the_corpus(protocol):
    dataset = load_dataset(protocol, "syn-clusters-d64", n=500, n_queries=16)
    from braid.exact import exact_topk

    result = exact_topk(dataset.queries, dataset.store, 1, protocol.convention)
    # perturbed corpus points should have a genuinely close nearest neighbour
    assert float(result.scores[:, 0].mean()) > 0.9
