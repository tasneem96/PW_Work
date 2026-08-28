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
pip install faiss-cpu h5py           # FAISS backend / ann-benchmarks datasets
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

## ANN-Benchmarks datasets (glove-25-angular)

`glove_retrieval/ann_benchmark.py` handles the HDF5 files from
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks) — `train`, `test`,
`neighbors`, `distances` — with an HNSW index.

```python
from glove_retrieval.ann_benchmark import faiss_index, recall_at_k

D, I = faiss_index(xb, xq, k=100)          # build HNSW + search, returns faiss (D, I)
print(recall_at_k(I, ground_truth, k=10))
```

```bash
python scripts/bench_hnsw.py data/glove-25-angular.hdf5 -k 10
```

```
built HNSW M=32 efConstruction=200 in 104s

 efSearch   recall@10         QPS   ms/query
       16      0.9668      88,039      0.011
       64      0.9995      40,662      0.025
      128      1.0000      24,010      0.042
      256      1.0000      14,232      0.070
```

(1,183,514 × 25 on 4 cores — glove-25-angular's exact shape, clustered synthetic
vectors. The graph is built once and `efSearch` swept, since the recall/speed
trade-off is a query-time knob.)

### Three things that quietly break this benchmark

**`-angular` means cosine.** The ground truth in `glove-25-angular.hdf5` is
cosine-based, and GloVe row norms vary by an order of magnitude, so an index
built on raw L2 or raw inner product disagrees with `neighbors` no matter how
well you tune HNSW. `faiss_index` L2-normalizes both sides and uses
`METRIC_INNER_PRODUCT` — for unit vectors `||a-b||² = 2-2cos`, so L2 and IP
rank identically, but IP makes the returned `D` cosine similarity directly. On
the test fixture: **recall@10 of 0.98 normalized vs. 0.27 not.** That is the
single biggest failure mode here, and it looks like a broken index rather than a
metric mistake.

**`faiss.normalize_L2` works in place.** Called on `xb` straight from h5py it
silently rewrites your database array, so any later exact-search comparison is
against different data. Every normalization here goes through a copy.

**HNSW's default `efSearch` is 16.** Asking for `k=100` with a candidate list of
16 cannot return 100 good neighbours. `faiss_index` defaults `ef_search` to
`max(2k, 64)` and never lets it fall below `k`.

### Distance conventions

Recall is computed from **ids**, so it does not depend on how the file defines
distance. If you do want to compare `D` against `distances`, the convention has
varied between dataset versions, so it is measured rather than assumed:

```python
from glove_retrieval.ann_benchmark import detect_convention_for_results
detect_convention_for_results(I, D, ground_truth, ground_truth_distances)
# -> '1 - cosine' | 'euclidean_on_unit' | 'arccos' | 'cosine_similarity' | None
```

Compare only where the returned id equals the exact id — an ANN result and the
ground truth do not line up cell for cell, and an elementwise comparison is
comparing distances to *different* neighbours (it matches nothing, which reads
as "unknown convention").

### Tracing one query through the graph

faiss reports only counters for a search (`hnsw_stats.ndis`, `nhops`) — it never
hands back the nodes it visited. `glove_retrieval/hnsw_trace.py` walks the same
graph with the same algorithm in Python, so the path is inspectable:

```python
from glove_retrieval.hnsw_trace import search_with_trace, verify_against_faiss

D, I, trace = search_with_trace(index, query, k=10, ef_search=64)
print(trace.summary())
print(trace.greedy_path)          # [entry, hop, hop, ...] down the upper layers
print(trace.path_at_level(2))     # the walk on one level
print(trace.expansions[0].node, trace.expansions[0].discovered)
```

```
entry point : 12586
greedy path : 12586 -> 16298 -> 7779   (2 hops down levels 2..1)
layer 0     : 65 nodes expanded, 1453 visited, 1466 distances computed
efSearch=64  k=10
```

```bash
python scripts/visualize_query_trace_3d.py --query-id 0 -k 10 -o trace.png
```

Being a re-implementation, it is worth nothing unless it matches:

```python
verify_against_faiss(index, query, k=10, ef_search=64)
# {'ids_match': True, 'distances_match': True, 'ndis_match': True, ...}
```

`ndis_match` is the strict one — two traversals that differ anywhere will not
agree on the number of distance computations. It holds exactly for both metrics
across `ef ∈ {16, 64, 256}` and `k ∈ {1, 10, 100}`. Two details were needed to
get there:

- faiss **does not count the entry point's own distance** in `ndis`. Counting it
  leaves you off by exactly one on every query.
- the loop stops when the nearest unexplored candidate is worse than the whole
  `ef`-sized frontier. Stopping on the *k*-sized result set instead still looks
  plausible but returns the wrong ids; stopping on "candidates below `d0`" never
  fires at all, since `d0` is the minimum — that one silently walks the entire
  graph (`ndis` 19865 vs. faiss's 1451).

Note `index.search` on an L2 index returns **squared** distances; the replay
matches that rather than taking a root.

### Offline fixture

`scripts/make_sample_ann_dataset.py` writes a small file with the same keys,
dtypes and cosine ground truth, with deliberately varied row norms so a missing
normalization fails loudly. The tests build it themselves; `.hdf5` files are
gitignored.

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
  ann_benchmark.py HNSW over ann-benchmarks HDF5 (faiss_index, recall_at_k)
  hnsw_trace.py    replay a search step by step: entry point, hops, expansions
  cli.py           argparse front end for both backends
scripts/
  download_glove.sh        fetch + extract glove.6B.<dim>.txt
  build_faiss_index.py     GloVe text file -> .faiss + labels sidecar
  bench_hnsw.py            recall/QPS sweep over an ann-benchmarks dataset
  make_sample_ann_dataset.py  small stand-in for glove-25-angular.hdf5
  visualize_hnsw_node.py      one node's layers, 2D (networkx)
  visualize_hnsw_node_3d.py   one node's layers, stacked in 3D (pyvista)
  visualize_query_trace_3d.py one query's path through the graph (pyvista)
  make_sample_vectors.py   regenerate the offline test fixture
data/
  sample.synthetic.50d.txt 80 synthetic 50-d vectors in 8 topical clusters
tests/
  test_retrieval.py        NumPy backend
  test_faiss.py            FAISS backend (skipped if faiss is missing)
  test_ann_benchmark.py    HNSW, recall, angular handling
  test_hnsw_trace.py       traversal replay vs. faiss (ids, distances, ndis)
```

## Tests

```bash
python -m pytest tests -q      # 94 tests; faiss/h5py tests skip if missing
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
