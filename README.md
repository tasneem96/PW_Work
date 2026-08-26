# PW_Work — GloVe top-k retrieval

Give it a **query embedding** (or a word/phrase), get back the **top-k nearest
vectors** from **GloVe 6B / 50d**.

Search is an exact brute-force scan: one dense mat-vec over the 400k x 50
matrix, which runs in ~5 ms and needs no approximate index.

## Setup

```bash
pip install -r requirements.txt      # numpy only
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
raises `ValueError` if the query's dimension does not match the index.

## Caching

The first load parses the 171 MB text file (~20 s) and writes
`data/glove.6B.50d.txt.cache/` containing `vectors.npy` + `vocab.txt`.
Later loads memory-map that cache and start instantly. The cache is keyed on
the source file's path, size and mtime, so it rebuilds itself if the vectors
change. `--limit N` gets its own cache directory.

## Layout

```
glove_retrieval/
  loader.py   GloVe text parsing (.txt/.gz/.zip, word2vec headers) + .npy cache
  index.py    GloveIndex: encode, scores, search, most_similar, analogy
  cli.py      argparse front end
scripts/
  download_glove.sh        fetch + extract glove.6B.<dim>.txt
  make_sample_vectors.py   regenerate the offline test fixture
data/
  sample.synthetic.50d.txt 80 synthetic 50-d vectors in 8 topical clusters
tests/test_retrieval.py
```

## Tests

```bash
python -m pytest tests -q      # or: python tests/test_retrieval.py
```

The tests run offline against `data/sample.synthetic.50d.txt`. **Those are not
GloVe vectors** — they are deterministic pseudo-random vectors grouped into
topical clusters (royalty, animals, food, tech, weather, family, transport,
music) so nearest-neighbour retrieval has a checkable right answer without a
822 MB download. Real queries should use the downloaded `glove.6B.50d.txt`.
