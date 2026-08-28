#!/usr/bin/env python3
"""Build a FAISS database from GloVe vectors.

    python scripts/build_faiss_index.py data/glove.6B.50d.txt data/glove.50d.faiss
    python scripts/build_faiss_index.py data/glove.6B.50d.txt out.faiss \
        --factory IVF1024,Flat --metric cosine

Writes the index plus a ``<index>.labels.txt`` sidecar holding the id -> word
mapping (FAISS itself stores only integer ids).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glove_retrieval.faiss_backend import FaissIndex  # noqa: E402
from glove_retrieval.loader import load_glove_text  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("vectors", help="GloVe text file (.txt/.gz/.zip)")
    parser.add_argument("out", help="output .faiss path")
    parser.add_argument(
        "--factory",
        default="Flat",
        help="faiss index_factory string, e.g. Flat, IVF1024,Flat, HNSW32 (default: Flat)",
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        choices=("cosine", "dot", "euclidean"),
        help="cosine normalizes vectors into an inner-product index (default: cosine)",
    )
    parser.add_argument("--limit", type=int, default=None, help="index only the N most frequent tokens")
    parser.add_argument("--train-size", type=int, default=200_000, help="training sample for IVF/PQ")
    args = parser.parse_args()

    start = time.perf_counter()
    vocab, matrix = load_glove_text(args.vectors, limit=args.limit)
    print(f"parsed {len(vocab):,} x {matrix.shape[1]}d in {time.perf_counter() - start:.1f}s")

    start = time.perf_counter()
    index = FaissIndex.build(
        vocab, matrix, metric=args.metric, factory=args.factory, train_size=args.train_size
    )
    print(f"built {args.factory} ({args.metric}) in {time.perf_counter() - start:.1f}s")

    path = index.save(args.out)
    print(f"wrote {path} + {path.with_suffix(path.suffix + '.labels.txt')}")
    print(f"try:   python -m glove_retrieval --faiss {path} --text king -k 10")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
