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
        description=(
            "Retrieve the top-k nearest vectors for a query embedding, from a GloVe "
            "text file or an existing FAISS database."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  glove-topk --text 'king'\n"
            "  glove-topk --text 'a cup of coffee' -k 5\n"
            "  glove-topk --vector '0.12,-0.4,...' -k 10 --metric cosine\n"
            "  glove-topk --vector-file query.json --json\n"
            "  cat query.json | glove-topk --vector-file -\n"
            "\n"
            "against an existing FAISS database:\n"
            "  glove-topk --faiss vectors.faiss --vector-file query.json -k 10\n"
            "  glove-topk --faiss ivf.faiss --labels vocab.txt --text king --nprobe 32\n"
            "  glove-topk --faiss vectors.faiss --describe\n"
        ),
    )
    source = parser.add_mutually_exclusive_group()
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
        help="GloVe file or cache directory (default: %(default)s, or $GLOVE_VECTORS)",
    )
    parser.add_argument(
        "--faiss",
        metavar="PATH",
        default=os.environ.get("GLOVE_FAISS_INDEX"),
        help="query this FAISS index instead of --vectors (or $GLOVE_FAISS_INDEX)",
    )
    parser.add_argument(
        "--labels",
        metavar="PATH",
        help="id -> word sidecar for --faiss (.txt/.tsv/.json/.npy); auto-detected if omitted",
    )
    parser.add_argument(
        "--nprobe", type=int, default=None, metavar="N", help="IVF probe count (recall vs. speed)"
    )
    normalization = parser.add_mutually_exclusive_group()
    normalization.add_argument(
        "--normalize",
        dest="normalize",
        action="store_true",
        default=None,
        help="L2-normalize the query embedding (cosine index); auto-detected by default",
    )
    normalization.add_argument(
        "--no-normalize", dest="normalize", action="store_false", help="never normalize the query"
    )
    parser.add_argument(
        "--describe", action="store_true", help="print index metadata and exit"
    )
    parser.add_argument("-k", "--top-k", type=int, default=10, help="how many results (default: 10)")
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default=None,
        help="similarity metric for --vectors (default: cosine); a FAISS index's own metric wins",
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


def _open_index(args):
    """Open whichever backend the flags select. Returns (index, kind)."""
    if args.faiss:
        from .faiss_backend import FaissIndex

        return (
            FaissIndex.open(
                args.faiss,
                labels=args.labels,
                normalize=args.normalize,
                nprobe=args.nprobe,
            ),
            "faiss",
        )
    if args.labels or args.nprobe is not None:
        raise ValueError("--labels/--nprobe only apply to --faiss")
    return GloveIndex.load(args.vectors, limit=args.limit, cache=not args.no_cache), "glove"


def main(argv: List[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(normalize_argv(argv))

    if not args.describe and not any(
        (args.text, args.vector, args.vector_file, args.analogy)
    ):
        print(
            "error: one of --text/--vector/--vector-file/--analogy is required "
            "(or --describe)",
            file=sys.stderr,
        )
        return 2

    try:
        index, kind = _open_index(args)
    except ImportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if kind == "faiss":
        if args.metric is not None and args.metric != index.metric:
            print(
                f"error: --metric {args.metric} conflicts with the index, which is "
                f"{index.metric} ({describe_metric(index)}). Rebuild the index, or "
                "use --normalize/--no-normalize.",
                file=sys.stderr,
            )
            return 2
        metric = index.metric
    else:
        metric = args.metric or "cosine"

    if args.describe:
        info = index.describe() if kind == "faiss" else {
            "type": "GloveIndex",
            "source": str(args.vectors),
            "dim": index.dim,
            "ntotal": len(index),
            "metric": metric,
        }
        print(json.dumps(info, indent=2))
        return 0

    exclude = list(args.exclude)
    try:
        query = None
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

        if query is not None:
            if kind == "faiss":
                results = index.search(query, k=args.top_k, exclude=exclude)
            else:
                results = index.search(query, k=args.top_k, metric=metric, exclude=exclude)
    except (KeyError, ValueError, OSError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = {
            "backend": kind,
            "source": str(args.faiss if kind == "faiss" else args.vectors),
            "dim": index.dim,
            "vocab_size": len(index),
            "metric": metric,
            "k": args.top_k,
            "results": [
                {"rank": r.rank, "word": r.word, "score": r.score, "index": r.index}
                for r in results
            ],
        }
        if kind == "faiss" and index.nprobe is not None:
            payload["nprobe"] = index.nprobe
        print(json.dumps(payload, indent=2))
    else:
        print(_format_table(results, metric))
    return 0


def describe_metric(index) -> str:
    """Explain a FAISS index's effective metric, for error messages."""
    if not index.is_inner_product:
        return "L2 index"
    return (
        "inner-product index over unit-norm vectors"
        if index.normalize
        else "inner-product index, query not normalized"
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
