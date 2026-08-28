"""Tests for the FAISS backend.

Skipped entirely when faiss is not installed.  Run with:
    python -m pytest tests/test_faiss.py -q
"""

from __future__ import annotations

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

faiss = pytest.importorskip("faiss")

from glove_retrieval import GloveIndex  # noqa: E402
from glove_retrieval.faiss_backend import (  # noqa: E402
    FaissIndex,
    find_labels,
    load_labels,
    stored_vectors_are_normalized,
)

SAMPLE = ROOT / "data" / "sample.synthetic.50d.txt"


@pytest.fixture(scope="module")
def glove():
    return GloveIndex.load(SAMPLE, cache=False)


@pytest.fixture(scope="module")
def flat(glove):
    return FaissIndex.build(glove.vocab, glove.vectors, metric="cosine")


# ----------------------------------------------------------------------
# parity with the brute-force backend
# ----------------------------------------------------------------------
def test_flat_cosine_matches_numpy_backend_exactly(glove, flat):
    """An exact FAISS index must reproduce the brute-force ranking and scores."""
    for word in ("king", "coffee", "guitar", "rain"):
        query = glove.vector(word)
        reference = glove.search(query, k=10, metric="cosine")
        got = flat.search(query, k=10)
        assert [r.word for r in got] == [r.word for r in reference], word
        assert np.allclose(
            [r.score for r in got], [r.score for r in reference], atol=1e-5
        ), word


def test_l2_index_returns_negated_distance(glove):
    index = FaissIndex.build(glove.vocab, glove.vectors, metric="euclidean")
    assert index.metric == "euclidean"
    assert not index.normalize
    query = glove.vector("coffee")
    got = index.search(query, k=5)
    assert got[0].word == "coffee"
    assert got[0].score == pytest.approx(0.0, abs=1e-4)
    # higher is better, i.e. distances are negative and non-increasing
    assert all(a.score >= b.score for a, b in zip(got, got[1:]))
    reference = glove.search(query, k=5, metric="euclidean")
    assert [r.word for r in got] == [r.word for r in reference]
    assert np.allclose([r.score for r in got], [r.score for r in reference], atol=1e-4)


def test_dot_index_is_not_cosine(glove):
    """A raw inner-product index must not silently normalize the query."""
    dot = FaissIndex.build(glove.vocab, glove.vectors, metric="dot")
    assert dot.metric == "dot" and not dot.normalize
    query = glove.vector("king")
    assert [r.word for r in dot.search(query, k=5)] == [
        r.word for r in glove.search(query, k=5, metric="dot")
    ]
    # scaling the query scales dot scores but leaves cosine scores alone
    scaled = dot.search(query * 3.0, k=1)[0].score
    assert scaled == pytest.approx(dot.search(query, k=1)[0].score * 3.0, rel=1e-4)


# ----------------------------------------------------------------------
# query embeddings
# ----------------------------------------------------------------------
def test_raw_query_embedding_of_any_dtype(flat, glove):
    raw = glove.vector("bread")
    assert flat.search(raw.tolist(), k=1)[0].word == "bread"
    assert flat.search(raw.astype(np.float64), k=1)[0].word == "bread"
    assert flat.search(raw.reshape(1, -1), k=1)[0].word == "bread"


def test_query_is_not_mutated_by_normalization(flat, glove):
    query = glove.vector("bread").astype(np.float32)
    before = query.copy()
    flat.search(query, k=3)
    assert np.array_equal(query, before), "search() must not normalize in place"


def test_dimension_mismatch_is_rejected(flat):
    with pytest.raises(ValueError, match="3 dimensions"):
        flat.search([0.1, 0.2, 0.3], k=1)


def test_batch_search(flat, glove):
    queries = np.stack([glove.vector("king"), glove.vector("coffee")])
    rows = flat.search_batch(queries, k=3)
    assert len(rows) == 2
    assert rows[0][0].word == "king" and rows[1][0].word == "coffee"
    assert [r.rank for r in rows[0]] == [1, 2, 3]


def test_encode_and_helpers(flat, glove):
    assert flat.most_similar("king", k=1)[0].word != "king"
    assert flat.search(flat.encode("coffee tea"), k=1)[0].word in {"coffee", "tea"}
    with pytest.raises(KeyError):
        flat.encode("zzzzq")


# ----------------------------------------------------------------------
# result-set hygiene
# ----------------------------------------------------------------------
def test_k_larger_than_ntotal_drops_padding(glove):
    """FAISS pads short result sets with id -1; those must never surface."""
    small = FaissIndex.build(glove.vocab[:5], np.asarray(glove.vectors)[:5])
    got = small.search(glove.vector("king"), k=20)
    assert len(got) == 5
    assert all(r.index >= 0 for r in got)
    assert all(r.word != "-1" for r in got)


def test_exclusion_still_returns_k_results(flat, glove):
    """Excluding hits must over-fetch, not shrink the result set."""
    got = flat.search(glove.vector("king"), k=5, exclude=["king", "queen", "royal"])
    assert len(got) == 5
    assert not {"king", "queen", "royal"} & {r.word for r in got}
    assert [r.rank for r in got] == [1, 2, 3, 4, 5]


# ----------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------
def test_save_open_roundtrip(tmp_path, flat, glove):
    path = flat.save(tmp_path / "vectors.faiss")
    assert path.is_file()
    assert (tmp_path / "vectors.faiss.labels.txt").is_file()

    reopened = FaissIndex.open(path)  # labels auto-detected
    assert len(reopened) == len(flat)
    assert reopened.dim == 50
    assert reopened.metric == "cosine"
    query = glove.vector("guitar")
    assert [r.word for r in reopened.search(query, k=5)] == [
        r.word for r in flat.search(query, k=5)
    ]


def test_open_rejects_a_short_labels_file(tmp_path, flat):
    path = flat.save(tmp_path / "v.faiss")
    (tmp_path / "short.txt").write_text("king\nqueen\n", encoding="utf-8")
    with pytest.raises(ValueError, match="80 vectors"):
        FaissIndex.open(path, labels=tmp_path / "short.txt")


def test_open_missing_index(tmp_path):
    with pytest.raises(FileNotFoundError):
        FaissIndex.open(tmp_path / "nope.faiss")


def test_index_without_labels_returns_ids(tmp_path, glove):
    index = FaissIndex.build(glove.vocab, glove.vectors)
    index.labels = None
    faiss.write_index(index.index, str(tmp_path / "bare.faiss"))
    bare = FaissIndex.open(tmp_path / "bare.faiss")
    got = bare.search(glove.vector("king"), k=3)
    assert [r.word for r in got] == [str(r.index) for r in got]
    with pytest.raises(KeyError, match="no labels file loaded"):
        bare.id_of("king")


# ----------------------------------------------------------------------
# labels
# ----------------------------------------------------------------------
def test_label_formats(tmp_path):
    (tmp_path / "a.txt").write_text("the\nof\nand\n", encoding="utf-8")
    assert load_labels(tmp_path / "a.txt") == ["the", "of", "and"]

    (tmp_path / "b.json").write_text('["the", "of"]', encoding="utf-8")
    assert load_labels(tmp_path / "b.json") == ["the", "of"]

    (tmp_path / "c.json").write_text('{"0": "the", "7": "and"}', encoding="utf-8")
    assert load_labels(tmp_path / "c.json") == {0: "the", 7: "and"}

    (tmp_path / "d.json").write_text('{"labels": ["the", "of"]}', encoding="utf-8")
    assert load_labels(tmp_path / "d.json") == ["the", "of"]

    (tmp_path / "e.tsv").write_text("0\tthe\n5\tand\n", encoding="utf-8")
    assert load_labels(tmp_path / "e.tsv") == {0: "the", 5: "and"}

    np.save(tmp_path / "f.npy", np.array(["the", "of"]))
    assert load_labels(tmp_path / "f.npy") == ["the", "of"]

    with pytest.raises(FileNotFoundError):
        load_labels(tmp_path / "missing.txt")


def test_find_labels_prefers_a_sidecar(tmp_path):
    (tmp_path / "v.faiss").write_bytes(b"")
    assert find_labels(tmp_path / "v.faiss") is None
    (tmp_path / "v.faiss.labels.txt").write_text("a\n", encoding="utf-8")
    assert find_labels(tmp_path / "v.faiss").name == "v.faiss.labels.txt"


def test_non_contiguous_ids_with_dict_labels(glove):
    """IndexIDMap stores arbitrary ids; a dict mapping must resolve them."""
    vectors = np.ascontiguousarray(np.asarray(glove.vectors)[:10], dtype=np.float32)
    faiss.normalize_L2(vectors)
    ids = np.array([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000], dtype=np.int64)
    inner = faiss.IndexFlatIP(50)
    wrapped = faiss.IndexIDMap(inner)
    wrapped.add_with_ids(vectors, ids)

    labels = {int(i): glove.vocab[n] for n, i in enumerate(ids)}
    index = FaissIndex(wrapped, labels, normalize=True)
    got = index.search(glove.vector("king"), k=3)
    assert got[0].word == "king"
    assert got[0].index == 100
    assert index.id_of("queen") == 200
    assert all(r.word in set(labels.values()) for r in got)


# ----------------------------------------------------------------------
# normalization detection
# ----------------------------------------------------------------------
def test_detects_unit_norm_vectors(glove, tmp_path):
    cosine = FaissIndex.build(glove.vocab, glove.vectors, metric="cosine")
    assert stored_vectors_are_normalized(cosine.index) is True
    path = cosine.save(tmp_path / "c.faiss")
    assert FaissIndex.open(path).normalize is True

    raw = FaissIndex.build(glove.vocab, glove.vectors, metric="dot")
    assert stored_vectors_are_normalized(raw.index) is False
    path = raw.save(tmp_path / "d.faiss")
    assert FaissIndex.open(path).normalize is False


def test_explicit_normalize_overrides_detection(glove, tmp_path):
    """--normalize is honoured, but must not be *reported* as cosine."""
    raw = FaissIndex.build(glove.vocab, glove.vectors, metric="dot")
    path = raw.save(tmp_path / "d.faiss")
    with pytest.warns(RuntimeWarning, match="not unit-norm"):
        forced = FaissIndex.open(path, normalize=True)
    assert forced.normalize is True
    assert forced.storage_normalized is False
    # normalizing only the query gives ||v||*cos(v,q) -- a different ranking
    # from cosine, so claiming "cosine" here would be a lie.
    assert forced.metric == "dot"
    top = forced.search(glove.vector("king"), k=1)[0]
    assert top.score > 1.0, "scores still carry the stored vector's magnitude"


def test_normalize_is_ignored_for_l2(glove):
    l2 = FaissIndex.build(glove.vocab, glove.vectors, metric="euclidean")
    assert l2.metric == "euclidean"


# ----------------------------------------------------------------------
# IVF / approximate indexes
# ----------------------------------------------------------------------
def test_ivf_nprobe_controls_recall(glove):
    rng = np.random.default_rng(7)
    n = 20_000
    vectors = rng.normal(size=(n, 50)).astype(np.float32)
    vocab = [f"w{i}" for i in range(n)]
    exact = GloveIndex(vocab, vectors)
    ivf = FaissIndex.build(vocab, vectors, metric="cosine", factory="IVF128,Flat")
    assert ivf.nprobe is not None

    queries = [vectors[i] for i in rng.integers(0, n, 25)]
    truth = [{r.word for r in exact.search(q, k=10)} for q in queries]

    def recall(nprobe):
        hits = 0
        for q, want in zip(queries, truth):
            got = {r.word for r in ivf.search(q, k=10, nprobe=nprobe)}
            hits += len(got & want)
        return hits / (10 * len(queries))

    low, high = recall(1), recall(64)
    assert high > low, f"nprobe=64 recall {high} should beat nprobe=1 recall {low}"
    assert high > 0.95, f"nprobe=64 recall was only {high}"


def test_nprobe_on_a_flat_index_is_an_error(flat):
    assert flat.nprobe is None
    with pytest.raises(AttributeError, match="not an IVF index"):
        flat.nprobe = 8


def test_hnsw_index_works(glove):
    index = FaissIndex.build(glove.vocab, glove.vectors, metric="cosine", factory="HNSW32")
    got = index.search(glove.vector("piano"), k=3)
    assert got[0].word == "piano"


def test_describe_reports_ivf_parameters(glove):
    ivf = FaissIndex.build(glove.vocab[:80], np.asarray(glove.vectors)[:80], factory="IVF4,Flat")
    info = ivf.describe()
    assert info["type"] == "IndexIVFFlat"
    assert info["nlist"] == 4 and "nprobe" in info
    assert info["dim"] == 50 and info["ntotal"] == 80
    assert info["effective_metric"] == "cosine"


def test_lossy_quantized_storage_is_still_detected_as_normalized(glove):
    """PQ reconstructs approximately; an exact norm test would misread it.

    A cosine-built IVF/PQ index reconstructs norms spread over roughly
    0.87-1.11.  Reading that as "not normalized" would silently downgrade every
    query on the user's database from cosine to a dot product.
    """
    rng = np.random.default_rng(3)
    vectors = rng.normal(size=(5000, 50)).astype(np.float32)
    vocab = [f"w{i}" for i in range(5000)]
    built = FaissIndex.build(vocab, vectors, metric="cosine", factory="IVF64,PQ10")
    assert stored_vectors_are_normalized(built.index) is True
    assert FaissIndex(built.index).metric == "cosine"

    raw = FaissIndex.build(vocab, vectors, metric="dot", factory="IVF64,PQ10")
    assert stored_vectors_are_normalized(raw.index) is False


def test_unreconstructable_index_warns_instead_of_guessing(glove, monkeypatch):
    """When storage cannot be inspected we must say so, not guess a metric."""
    import glove_retrieval.faiss_backend as backend

    flat = FaissIndex.build(glove.vocab, glove.vectors, metric="cosine")
    monkeypatch.setattr(backend, "_reconstruct_sample", lambda index, n=64: None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        index = FaissIndex(flat.index)
    assert any("unit-norm" in str(w.message) for w in caught)
    assert index.storage_normalized is None
    assert index.normalize is False  # falls back to a plain dot product


def test_idmap_v1_reconstruction_does_not_abort(glove):
    """reconstruct_n on an IndexIDMap aborts the process; we must never call it."""
    vectors = np.ascontiguousarray(np.asarray(glove.vectors)[:20], dtype=np.float32)
    faiss.normalize_L2(vectors)
    ids = (np.arange(20, dtype=np.int64) + 1000) * 7
    wrapped = faiss.IndexIDMap(faiss.IndexFlatIP(50))
    wrapped.add_with_ids(vectors, ids)

    labels = {int(i): glove.vocab[n] for n, i in enumerate(ids)}
    index = FaissIndex(wrapped, labels)  # runs auto-detection
    assert index.storage_normalized is True
    assert index.metric == "cosine"
    # IndexIDMap v1 has no reconstruct-by-id; the position fallback must cover it
    assert np.allclose(index.vector("king"), vectors[0], atol=1e-6)
    assert index.most_similar("king", k=1)[0].word != "king"


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "glove_retrieval", *args],
        cwd=ROOT, capture_output=True, text=True,
    )


def test_cli_faiss_query(tmp_path, flat, glove):
    path = flat.save(tmp_path / "v.faiss")
    proc = _cli("--faiss", str(path), "--text", "king", "-k", "3", "--json")
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["backend"] == "faiss"
    assert payload["metric"] == "cosine"
    assert payload["results"][0]["word"] == "king"
    assert len(payload["results"]) == 3


def test_cli_faiss_raw_embedding(tmp_path, flat, glove):
    path = flat.save(tmp_path / "v.faiss")
    query = tmp_path / "q.json"
    query.write_text(json.dumps({"embedding": glove.vector("bacon").tolist()}), encoding="utf-8")
    proc = _cli("--faiss", str(path), "--vector-file", str(query), "-k", "1", "--json")
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["results"][0]["word"] == "bacon"


def test_cli_describe(tmp_path, flat):
    path = flat.save(tmp_path / "v.faiss")
    proc = _cli("--faiss", str(path), "--describe")
    assert proc.returncode == 0, proc.stderr
    info = json.loads(proc.stdout)
    assert info["type"] == "IndexFlatIP" and info["ntotal"] == 80


def test_cli_metric_conflict_is_rejected(tmp_path, flat):
    path = flat.save(tmp_path / "v.faiss")
    proc = _cli("--faiss", str(path), "--text", "king", "--metric", "euclidean")
    assert proc.returncode == 2
    assert "conflicts with the index" in proc.stderr


def test_cli_requires_a_query(tmp_path, flat):
    path = flat.save(tmp_path / "v.faiss")
    proc = _cli("--faiss", str(path))
    assert proc.returncode == 2
    assert "--text/--vector" in proc.stderr


def test_cli_rejects_faiss_flags_on_the_numpy_backend():
    proc = _cli("--vectors", str(SAMPLE), "--text", "king", "--nprobe", "8")
    assert proc.returncode == 2
    assert "only apply to --faiss" in proc.stderr
