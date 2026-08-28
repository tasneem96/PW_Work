# PW_Work — GloVe top-k retrieval

Give it a **query embedding** (or a word/phrase), get back the **top-k nearest
vectors** from **GloVe 6B / 50d**.

Two interchangeable backends behind one interface:

| backend | flag | use when |
| --- | --- | --- |
| NumPy brute force | `--vectors` | you have a GloVe text file; exact, ~5 ms |
| FAISS | `--faiss` | you already have a FAISS database, or want an ANN index |

## Setup

```bash
pip install -r requirements.txt      # numpy
pip install faiss-cpu                # only for the FAISS backend
./scripts/download_glove.sh          # fetches glove.6B.zip, keeps glove.6B.50d.txt in ./data
```

`download_glove.sh 100d` (or `200d`/`300d`) grabs a different dimension; the
code works with any of them.

> The download needs network access to `nlp.stanford.edu` (or the Hugging Face
> mirror). If those hosts are blocked, download `glove.6B.zip` elsewhere and
> point `--vectors` at the extracted file.

## CLI

```bash
# a raw query embedding: 50 comma- or space-separated floats
python -m glove_retrieval --vector "0.418,0.24968,-0.41242,..." -k 10

# a query embedding from a JSON file — [..] or {"embedding": [..]} — or stdin
python -m glove_retrieval --vector-file query.json -k 10 --json
cat query.json | python -m glove_retrieval --vector-file - -k 10

# let the tool build the embedding for you (single word, or mean of a phrase)
python -m glove_retrieval --text king -k 10 --exclude-query
python -m glove_retrieval --text "a cup of coffee" -k 5

# analogies: king - man + woman
python -m glove_retrieval --analogy king -man woman -k 5
```

| flag | meaning |
| --- | --- |
| `--vectors PATH` | GloVe file or cache dir (default `data/glove.6B.50d.txt`, or `$GLOVE_VECTORS`) |
| `-k, --top-k N` | number of results (default 10) |
| `--metric` | `cosine` (default), `dot`, or `euclidean` |
| `--limit N` | load only the N most frequent tokens — GloVe files are frequency-sorted |
| `--exclude WORD...` | drop specific words from the results |
| `--exclude-query` | with `--text`, drop the query's own tokens |
| `--json` | machine-readable output |
| `--no-cache` | skip the `.npy` cache |

Output:

```
rank  word     cosine
   1  queen    +0.783924
   2  prince   +0.766591
   ...
```

With `--metric euclidean` the column is a distance (smaller is better); the
other metrics are similarities (larger is better).

## Querying an existing FAISS database

Point `--faiss` at the index file. Everything else — `--vector`, `--vector-file`,
`-k`, `--exclude` — works exactly as it does on the NumPy backend.

```bash
python -m glove_retrieval --faiss vectors.faiss --vector-file query.json -k 10
python -m glove_retrieval --faiss vectors.faiss --describe          # what is in there?
python -m glove_retrieval --faiss ivf.faiss --labels vocab.txt --text king --nprobe 32
```

```python
from glove_retrieval import FaissIndex

index = FaissIndex.open("vectors.faiss")          # labels sidecar auto-detected
index.search(my_embedding, k=10)
index.search_batch(many_embeddings, k=10)         # one FAISS call for N queries
```

### Labels

FAISS stores vectors and integer ids, never strings, so the id → word mapping
comes from a sidecar. It is auto-detected next to the index
(`<index>.labels.txt`, `vocab.txt`, …) or set with `--labels`. Accepted:

| format | shape |
| --- | --- |
| `.txt` | one label per line; line number = id |
| `.tsv` / `.csv` | `id<sep>label`, or one label per line |
| `.json` | `["the", "of", ...]` or `{"0": "the", "7": "and"}` |
| `.npy` | array of strings |

With no labels the results carry the raw FAISS ids. A dict/`.tsv` mapping is
what you want for an `IndexIDMap` with non-contiguous ids.

### Cosine vs. dot product

This is the easiest thing to get silently wrong. FAISS has no cosine metric —
cosine is an **inner-product index over unit-norm vectors**, and the query has
to be normalized too, or you get a dot product ranking instead.

The backend reconstructs a sample of stored vectors and checks whether they are
unit-norm, then normalizes the query to match. `--describe` shows what it
concluded:

```json
{ "metric": "inner_product", "storage_normalized": true,
  "normalize_query": true, "effective_metric": "cosine" }
```

Override with `--normalize` / `--no-normalize`. Forcing `--normalize` on an
index whose vectors are *not* unit-norm warns and still reports the metric as
`dot`, because normalizing one side gives `||v||·cos(v, q)` — a different
ranking from cosine, and calling it cosine would be a lie. `--metric` is
rejected outright when it contradicts the index.

Detection uses the **median** norm, not an exact test: quantized indexes (PQ,
SQ) reconstruct approximately, and a cosine-built `IVF,PQ` index comes back with
norms spread over ~0.87–1.11. An exact test would read that as un-normalized and
quietly downgrade every query to a dot product. Indexes that cannot reconstruct
at all get a warning rather than a guess.

### Building an index

```bash
python scripts/build_faiss_index.py data/glove.6B.50d.txt data/glove.50d.faiss
python scripts/build_faiss_index.py data/glove.6B.50d.txt out.faiss --factory IVF1024,Flat
```

Measured on 400k × 50d, k=10, clustered synthetic data (a fairer stand-in for
real embeddings than gaussian noise):

| index | query | recall@10 |
| --- | --- | --- |
| NumPy exact | 5.0 ms | 1.000 |
| `Flat` (FAISS exact) | 6.3 ms | 1.000 |
| `IVF1024,Flat` `nprobe=16` | 0.24 ms | 1.000 |
| `IVF1024,Flat` `nprobe=1` | 0.09 ms | 0.970 |
| `HNSW32` | 0.11 ms | 0.954 |

At GloVe 6B scale an exact scan is already fast enough, so `Flat` is the safe
default; ANN pays off when you have millions of vectors or a high query rate.
An IVF index left at `nprobe=1` prints a one-time note, since that scans ~0.1%
of the data and silently costs recall.

## Python API

```python
from glove_retrieval import GloveIndex

index = GloveIndex.load("data/glove.6B.50d.txt")   # builds a .npy cache on first run
print(len(index), index.dim)                        # 400000 50

# 1. you already have a 50-d query embedding
for hit in index.search(my_embedding, k=10):
    print(hit.rank, hit.word, hit.score)

# 2. or build one from text
index.search(index.encode("morning coffee"), k=5)

# convenience wrappers
index.most_similar("king", k=10)                    # excludes 'king' itself
index.analogy(["king", "woman"], ["man"], k=5)
index.scores(my_embedding)                          # raw similarity for all 400k rows
```

`search()` accepts a list, a NumPy array of any float dtype, or a string; it
raises `ValueError` if the query's dimension does not match the index. Query
arrays are never modified in place. `FaissIndex` exposes the same methods, so
swapping backends is a one-line change.

Scores are always "higher is better": cosine and dot as-is, euclidean as a
**negated** distance. The CLI prints euclidean back as a positive distance.

## Caching

The first load parses the 171 MB text file (~20 s) and writes
`data/glove.6B.50d.txt.cache/` containing `vectors.npy` + `vocab.txt`.
Later loads memory-map that cache and start instantly. The cache is keyed on
the source file's path, size and mtime, so it rebuilds itself if the vectors
change. `--limit N` gets its own cache directory.

## Layout

```
glove_retrieval/
  loader.py        GloVe text parsing (.txt/.gz/.zip, word2vec headers) + .npy cache
  index.py         GloveIndex: encode, scores, search, most_similar, analogy
  faiss_backend.py FaissIndex: open/build a FAISS db, labels, metric detection
  cli.py           argparse front end for both backends
scripts/
  download_glove.sh        fetch + extract glove.6B.<dim>.txt
  build_faiss_index.py     GloVe text file -> .faiss + labels sidecar
  make_sample_vectors.py   regenerate the offline test fixture
data/
  sample.synthetic.50d.txt 80 synthetic 50-d vectors in 8 topical clusters
tests/
  test_retrieval.py        NumPy backend
  test_faiss.py            FAISS backend (skipped if faiss is missing)
```

## Tests

```bash
python -m pytest tests -q      # 52 tests; test_faiss.py skips without faiss
```

`test_faiss.py` pins the parts that are easy to get wrong: FAISS `Flat` results
must match the brute-force backend score for score, `-1` padding must never
surface as a result, exclusions must over-fetch rather than shrink the result
set, `nprobe` must actually move recall, and `IndexIDMap` must not be handed to
`reconstruct_n` (which aborts the process rather than raising).

The tests run offline against `data/sample.synthetic.50d.txt`. **Those are not
GloVe vectors** — they are deterministic pseudo-random vectors grouped into
topical clusters (royalty, animals, food, tech, weather, family, transport,
music) so nearest-neighbour retrieval has a checkable right answer without a
822 MB download. Real queries should use the downloaded `glove.6B.50d.txt`.
