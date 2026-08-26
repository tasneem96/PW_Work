"""Tests for the GloVe top-k retrieval package.

Run with:  python -m pytest tests -q   (or plain `python tests/test_retrieval.py`)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from glove_retrieval import GloveIndex  # noqa: E402
from glove_retrieval.cli import main, parse_vector_literal, read_vector_file  # noqa: E402
from glove_retrieval.index import tokenize  # noqa: E402
from glove_retrieval.loader import infer_dim, load_glove_text  # noqa: E402

SAMPLE = ROOT / "data" / "sample.synthetic.50d.txt"


def toy_index() -> GloveIndex:
    """A hand-built index whose nearest neighbours are known by construction."""
    vocab = ["a", "b", "c", "d"]
    vectors = np.array(
        [
            [1.0, 0.0],  # a
            [0.9, 0.1],  # b : closest to a by cosine and euclidean
            [0.0, 1.0],  # c : orthogonal to a
            [-1.0, 0.0],  # d : opposite of a
        ],
        dtype=np.float32,
    )
    return GloveIndex(vocab, vectors)


def test_parses_glove_text_format():
    dim = infer_dim(SAMPLE)
    vocab, matrix = load_glove_text(SAMPLE)
    assert dim == 50
    assert matrix.shape == (len(vocab), 50)
    assert matrix.dtype == np.float32
    assert "king" in vocab


def test_word2vec_header_and_limit(tmp_path=None):
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "with_header.txt"
        path.write_text("3 2\nx 1.0 0.0\ny 0.0 1.0\nz 1.0 1.0\n", encoding="utf-8")
        vocab, matrix = load_glove_text(path)
        assert vocab == ["x", "y", "z"] and matrix.shape == (3, 2)
        vocab, matrix = load_glove_text(path, limit=2)
        assert vocab == ["x", "y"] and matrix.shape == (2, 2)


def test_topk_order_is_correct():
    index = toy_index()
    results = index.search([1.0, 0.0], k=4)
    assert [r.word for r in results] == ["a", "b", "c", "d"]
    assert [r.rank for r in results] == [1, 2, 3, 4]
    assert results[0].score == 1.0
    assert results[3].score == -1.0
    # scores are monotonically non-increasing
    assert all(a.score >= b.score for a, b in zip(results, results[1:]))


def test_k_is_respected_and_clamped():
    index = toy_index()
    assert len(index.search([1.0, 0.0], k=2)) == 2
    assert len(index.search([1.0, 0.0], k=99)) == 4


def test_metrics_agree_on_the_nearest_point():
    index = toy_index()
    for metric in ("cosine", "dot", "euclidean"):
        top = index.search([1.0, 0.0], k=1, metric=metric)[0]
        assert top.word == "a", metric
    # euclidean scores are negated distances
    assert index.search([1.0, 0.0], k=1, metric="euclidean")[0].score == 0.0


def test_cosine_is_scale_invariant_but_dot_is_not():
    index = toy_index()
    small = index.search([1.0, 0.0], k=4)
    large = index.search([50.0, 0.0], k=4)
    assert [r.word for r in small] == [r.word for r in large]
    assert np.allclose([r.score for r in small], [r.score for r in large], atol=1e-6)
    assert index.search([50.0, 0.0], k=1, metric="dot")[0].score == 50.0


def test_scores_match_a_brute_force_reference():
    index = GloveIndex.load(SAMPLE, cache=False)
    query = index.vector("king")
    matrix = np.asarray(index.vectors, dtype=np.float64)
    q = np.asarray(query, dtype=np.float64)
    reference = (matrix @ q) / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q))
    expected = [index.vocab[i] for i in np.argsort(-reference)[:5]]
    assert [r.word for r in index.search(query, k=5)] == expected
    assert np.allclose([r.score for r in index.search(query, k=5)], np.sort(reference)[::-1][:5], atol=1e-5)


def test_exclusion_and_most_similar():
    index = GloveIndex.load(SAMPLE, cache=False)
    assert index.search(index.vector("king"), k=1)[0].word == "king"
    assert index.most_similar("king", k=1)[0].word != "king"
    excluded = index.search(index.vector("king"), k=3, exclude=["king", "queen"])
    assert not {"king", "queen"} & {r.word for r in excluded}


def test_clustered_neighbours_are_recovered():
    """Synthetic fixture: same-topic words should dominate the top of the list."""
    index = GloveIndex.load(SAMPLE, cache=False)
    royalty = {"king", "queen", "prince", "princess", "monarch", "throne", "crown",
               "royal", "kingdom", "emperor"}
    neighbours = {r.word for r in index.most_similar("king", k=5)}
    assert neighbours <= royalty


def test_raw_query_embedding_roundtrip():
    index = GloveIndex.load(SAMPLE, cache=False)
    raw = index.vector("coffee").tolist()
    assert index.search(raw, k=1)[0].word == "coffee"
    assert index.search(np.asarray(raw, dtype=np.float64), k=1)[0].word == "coffee"


def test_dimension_mismatch_is_rejected():
    index = GloveIndex.load(SAMPLE, cache=False)
    try:
        index.search([0.1, 0.2, 0.3], k=1)
    except ValueError as exc:
        assert "3 dimensions" in str(exc) and "50" in str(exc)
    else:
        raise AssertionError("expected a ValueError for a 3-d query against a 50-d index")


def test_phrase_encoding_is_the_token_mean():
    index = GloveIndex.load(SAMPLE, cache=False)
    expected = (index.vector("coffee") + index.vector("tea")) / 2.0
    assert np.allclose(index.encode("Coffee and TEA!"), expected, atol=1e-6)


def test_out_of_vocabulary_handling():
    index = GloveIndex.load(SAMPLE, cache=False)
    # unknown tokens are skipped ...
    assert np.allclose(index.encode("coffee zzzzq"), index.vector("coffee"), atol=1e-6)
    # ... unless strict, and an all-unknown query always raises
    for call in (lambda: index.encode("coffee zzzzq", strict=True), lambda: index.encode("zzzzq")):
        try:
            call()
        except KeyError:
            pass
        else:
            raise AssertionError("expected KeyError for out-of-vocabulary input")


def test_cache_roundtrip_is_lossless():
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        cache = Path(tmp) / "cache"
        built = GloveIndex.load(SAMPLE, cache=True, cache_dir=cache)
        assert (cache / "vectors.npy").is_file()
        reloaded = GloveIndex.load(SAMPLE, cache=True, cache_dir=cache)  # cache hit
        assert reloaded.vocab == built.vocab
        assert np.array_equal(np.asarray(reloaded.vectors), np.asarray(built.vectors))
        assert [r.word for r in reloaded.search(built.vector("king"), k=5)] == [
            r.word for r in built.search(built.vector("king"), k=5)
        ]


def test_vector_literal_and_file_parsing():
    import tempfile

    assert np.allclose(parse_vector_literal("1, 2 ,3"), [1, 2, 3])
    assert np.allclose(parse_vector_literal("[1.5 -2.5]"), [1.5, -2.5])
    with tempfile.TemporaryDirectory() as tmp:
        as_list = Path(tmp) / "a.json"
        as_list.write_text("[0.5, 1.5]", encoding="utf-8")
        assert np.allclose(read_vector_file(str(as_list)), [0.5, 1.5])
        as_obj = Path(tmp) / "b.json"
        as_obj.write_text(json.dumps({"embedding": [1.0, 2.0]}), encoding="utf-8")
        assert np.allclose(read_vector_file(str(as_obj)), [1.0, 2.0])
        as_txt = Path(tmp) / "c.txt"
        as_txt.write_text("1 2 3\n", encoding="utf-8")
        assert np.allclose(read_vector_file(str(as_txt)), [1, 2, 3])


def test_tokenizer():
    assert tokenize("The King's Crown, 2014!") == ["the", "king's", "crown", "2014"]


def test_cli_json_output(capsys=None):
    proc = subprocess.run(
        [sys.executable, "-m", "glove_retrieval", "--vectors", str(SAMPLE),
         "--text", "king", "-k", "3", "--json", "--no-cache"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["dim"] == 50 and payload["metric"] == "cosine"
    assert payload["results"][0]["word"] == "king"
    assert len(payload["results"]) == 3


def test_cli_accepts_a_raw_query_embedding():
    index = GloveIndex.load(SAMPLE, cache=False)
    literal = ",".join(f"{v:.6f}" for v in index.vector("guitar"))
    proc = subprocess.run(
        [sys.executable, "-m", "glove_retrieval", "--vectors", str(SAMPLE),
         "--vector", literal, "-k", "1", "--json", "--no-cache"],
        cwd=ROOT, capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["results"][0]["word"] == "guitar"


def test_cli_missing_vector_file_exits_nonzero():
    assert main(["--vectors", str(SAMPLE), "--text", "king", "--no-cache"]) == 0
    assert main(["--vectors", "does/not/exist.txt", "--text", "king"]) == 2


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    print(f"\n{failures} failure(s)")
    raise SystemExit(1 if failures else 0)
