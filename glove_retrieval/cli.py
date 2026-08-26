"""Command line interface: give it a query embedding, get the top-k neighbours."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np

from .index import METRICS, GloveIndex, SearchResult

DEFAULT_VECTORS = os.environ.get("GLOVE_VECTORS", "data/glove.6B.50d.txt")


def parse_vector_literal(text: str) -> np.ndarray:
    """Parse ``0.1,0.2,...`` or ``[0.1, 0.2, ...]`` or whitespace-separated floats."""
    text = text.strip().lstrip("[").rstrip("]")
    parts = [p for p in text.replace(",", " ").split() if p]
    if not parts:
        raise argparse.ArgumentTypeError("empty query vector")
    try:
        return np.asarray([float(p) for p in parts], dtype=np.float32)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"not a numeric vector: {exc}") from None


def read_vector_file(path: str) -> np.ndarray:
    """Read a query embedding from JSON (list, or ``{"embedding": [...]}``) or text."""
    raw = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    stripped = raw.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            for key in ("embedding", "vector", "query", "values", "data"):
                if key in payload:
                    payload = payload[key]
                    break
            else:
                raise ValueError(
                    "JSON object has no 'embedding'/'vector'/'query'/'values' key"
                )
        return np.asarray(payload, dtype=np.float32).reshape(-1)
    return parse_vector_literal(stripped)


def normalize_argv(argv: Sequence[str]) -> List[str]:
    """Attach a leading-minus vector value to its flag.

    ``--vector -0.12,0.4`` would otherwise be read by argparse as an unknown
    option, because the value is not parseable as a plain negative number.
    """
    flags = {"--vector", "-v"}
    out: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in flags and i + 1 < len(argv) and argv[i + 1].startswith("-"):
            out.append(f"{token}={argv[i + 1]}")
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="glove-topk",
        description="Retrieve the top-k nearest GloVe 6B/50d vectors for a query embedding.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  glove-topk --text 'king'\n"
            "  glove-topk --text 'a cup of coffee' -k 5\n"
            "  glove-topk --vector '0.12,-0.4,...' -k 10 --metric cosine\n"
            "  glove-topk --vector-file query.json --json\n"
            "  cat query.json | glove-topk --vector-file -\n"
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", "-t", help="embed this word/phrase and search with it")
    source.add_argument("--vector", "-v", help="query embedding as comma/space separated floats")
    source.add_argument(
        "--vector-file", "-f", help="read the query embedding from a JSON/text file ('-' for stdin)"
    )
    source.add_argument(
        "--analogy",
        nargs="+",
        metavar="TERM",
        help="analogy query, e.g. --analogy king -man woman (prefix '-' to subtract)",
    )

    parser.add_argument(
        "--vectors",
        default=DEFAULT_VECTORS,
        help=f"GloVe file or cache directory (default: %(default)s, or $GLOVE_VECTORS)",
    )
    parser.add_argument("-k", "--top-k", type=int, default=10, help="how many results (default: 10)")
    parser.add_argument(
        "--metric", choices=METRICS, default="cosine", help="similarity metric (default: cosine)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="load only the N most frequent tokens (smaller index, faster startup)",
    )
    parser.add_argument("--no-cache", action="store_true", help="skip the .npy cache")
    parser.add_argument(
        "--exclude", nargs="*", default=[], metavar="WORD", help="words to omit from the results"
    )
    parser.add_argument(
        "--exclude-query",
        action="store_true",
        help="with --text, drop the query's own tokens from the results",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    return parser


def _format_table(results: Sequence[SearchResult], metric: str) -> str:
    if not results:
        return "(no results)"
    width = max(len(r.word) for r in results)
    label = "distance" if metric == "euclidean" else metric
    lines = [f"{'rank':>4}  {'word'.ljust(width)}  {label}"]
    for r in results:
        score = -r.score if metric == "euclidean" else r.score
        lines.append(f"{r.rank:>4}  {r.word.ljust(width)}  {score:+.6f}")
    return "\n".join(lines)


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(normalize_argv(argv))

    try:
        index = GloveIndex.load(args.vectors, limit=args.limit, cache=not args.no_cache)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    exclude = list(args.exclude)
    try:
        if args.text is not None:
            query = index.encode(args.text)
            if args.exclude_query:
                from .index import tokenize

                exclude.extend(tokenize(args.text))
        elif args.vector is not None:
            query = parse_vector_literal(args.vector)
        elif args.vector_file is not None:
            query = read_vector_file(args.vector_file)
        else:
            positive = [t for t in args.analogy if not t.startswith("-")]
            negative = [t[1:] for t in args.analogy if t.startswith("-")]
            results = index.analogy(positive, negative, k=args.top_k)
            query = None
        if query is not None:
            results = index.search(query, k=args.top_k, metric=args.metric, exclude=exclude)
    except (KeyError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                {
                    "vectors": str(args.vectors),
                    "dim": index.dim,
                    "vocab_size": len(index),
                    "metric": args.metric,
                    "k": args.top_k,
                    "results": [
                        {"rank": r.rank, "word": r.word, "score": r.score, "index": r.index}
                        for r in results
                    ],
                },
                indent=2,
            )
        )
    else:
        print(_format_table(results, args.metric))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
