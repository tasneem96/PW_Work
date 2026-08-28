#!/usr/bin/env python3
"""Run HNSW against an ANN-Benchmarks dataset and report recall vs. speed.

    python scripts/bench_hnsw.py data/glove-25-angular.hdf5
    python scripts/bench_hnsw.py data/glove-25-angular.hdf5 -k 10 -M 32 --ef 16 64 256

Builds the graph once, then sweeps efSearch -- the recall/QPS trade-off is made
at query time, so rebuilding per point would waste minutes for nothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glove_retrieval.ann_benchmark import (  # noqa: E402
    build_hnsw,
    detect_convention_for_results,
    load_dataset,
    normalized,
    recall_at_k,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", default="data/glove-25-angular.hdf5")
    parser.add_argument("-k", type=int, default=10, help="neighbours to retrieve (default: 10)")
    parser.add_argument("-M", type=int, default=32, help="HNSW graph degree (default: 32)")
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument(
        "--ef", type=int, nargs="+", default=[16, 32, 64, 128, 256], help="efSearch values to sweep"
    )
    parser.add_argument("--metric", default=None, help="override the file's distance attribute")
    parser.add_argument("--queries", type=int, default=None, help="use only the first N queries")
    args = parser.parse_args()

    try:
        data = load_dataset(args.dataset)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    xb, xq = data["train"], data["test"]
    gt = data["neighbors"]
    if args.queries:
        xq, gt = xq[: args.queries], gt[: args.queries]
        if "distances" in data:
            data["distances"] = data["distances"][: args.queries]
    metric = args.metric or str(data["metric"])

    print(f"{args.dataset}: train {xb.shape}, test {xq.shape}, neighbors {gt.shape}, metric={metric}")
    if args.k > gt.shape[1]:
        print(f"error: ground truth has only {gt.shape[1]} neighbours per query", file=sys.stderr)
        return 2

    start = time.perf_counter()
    index = build_hnsw(xb, M=args.M, ef_construction=args.ef_construction, metric=metric)
    print(f"built HNSW M={args.M} efConstruction={args.ef_construction} in {time.perf_counter() - start:.1f}s")

    queries = normalized(xq) if metric in ("angular", "cosine") else np.ascontiguousarray(xq, dtype=np.float32)

    print(f"\n{'efSearch':>9}  {'recall@' + str(args.k):>10}  {'QPS':>10}  {'ms/query':>9}")
    first = None
    for ef in sorted(args.ef):
        index.hnsw.efSearch = max(ef, args.k)
        index.search(queries[:1], args.k)  # warm up
        start = time.perf_counter()
        D, I = index.search(queries, args.k)
        elapsed = time.perf_counter() - start
        if first is None:
            first = (D, I)
        print(
            f"{index.hnsw.efSearch:>9}  {recall_at_k(I, gt, args.k):>10.4f}  "
            f"{len(queries) / elapsed:>10,.0f}  {elapsed / len(queries) * 1000:>9.3f}"
        )

    if "distances" in data and metric in ("angular", "cosine"):
        convention = detect_convention_for_results(first[1], first[0], gt, data["distances"])
        print(
            f"\nfile's 'distances' match: {convention or 'none of the known conventions'}"
            "\n(recall above is computed from ids, so it does not depend on this)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
