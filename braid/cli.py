"""Command-line entry point: ``python -m braid <group> <command>``.

Groups mirror the phases:

``protocol``  validate, show, freeze, hash (Phase 0)
``phase0``    gate
``phase1``    sweep, gate, parity, trace
``run``       protocol validate -> phase 0 gate -> sweep -> phase 1 gate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .datasets import load_dataset
from .gates import phase0_gate, phase1_gate
from .hnsw.native import parity_report
from .hnsw.reference import build_index, search
from .hnsw.trace import ExposurePolicy, TraceLevel, knowledge_fraction
from .paths import DEFAULT_PROTOCOL
from .protocol import ProtocolError, freeze_protocol, load_protocol
from .rng import derive_int
from .splits import make_split
from .sweep import run_clean_sweep


def _add_protocol_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--protocol",
        default=str(DEFAULT_PROTOCOL),
        help=f"path to the protocol file (default: {DEFAULT_PROTOCOL})",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="braid", description=__doc__)
    parser.add_argument("--version", action="version", version=f"braid {__version__}")
    groups = parser.add_subparsers(dest="group", required=True)

    protocol = groups.add_parser("protocol", help="Phase 0 protocol file operations")
    protocol_cmds = protocol.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("validate", "validate the protocol schema and verify its freeze hash"),
        ("show", "print the protocol as canonical JSON"),
        ("hash", "print the protocol content hash"),
        ("freeze", "stamp a draft protocol as frozen and write its hash"),
    ):
        sub = protocol_cmds.add_parser(name, help=help_text)
        _add_protocol_arg(sub)
        if name == "freeze":
            sub.add_argument("--force", action="store_true", help="re-freeze an already frozen file")
        if name in ("validate", "show", "hash"):
            sub.add_argument(
                "--allow-draft", action="store_true", help="do not require status=frozen"
            )

    phase0 = groups.add_parser("phase0", help="Phase 0: protocol and claim freeze")
    phase0_cmds = phase0.add_subparsers(dest="command", required=True)
    gate0 = phase0_cmds.add_parser("gate", help="run the Phase 0 exit gate")
    _add_protocol_arg(gate0)
    gate0.add_argument("--json", action="store_true", help="print the gate result as JSON")

    phase1 = groups.add_parser("phase1", help="Phase 1: baseline system and instrumentation")
    phase1_cmds = phase1.add_subparsers(dest="command", required=True)

    sweep = phase1_cmds.add_parser("sweep", help="run the clean M x efSearch sweep")
    _add_protocol_arg(sweep)
    sweep.add_argument("--profile", default="smoke", help="run profile (default: smoke)")
    sweep.add_argument("--dataset", action="append", dest="datasets", help="restrict to a dataset id")
    sweep.add_argument("--n", type=int, default=None, help="subset the corpus size")
    sweep.add_argument("--queries", type=int, default=None, help="subset the query count")
    sweep.add_argument("--out", default=None, help="output directory")
    sweep.add_argument("--trace-sample", type=int, default=8, help="queries traced in full detail")
    sweep.add_argument("--no-parity", action="store_true", help="skip the hnswlib parity check")
    sweep.add_argument("--quiet", action="store_true")

    gate1 = phase1_cmds.add_parser("gate", help="run the Phase 1 exit gate")
    _add_protocol_arg(gate1)
    gate1.add_argument("--summary", default=None, help="sweep summary.json (default: newest)")
    gate1.add_argument(
        "--require-claim-bearing",
        action="store_true",
        help="fail unless the sweep used a claim-bearing profile",
    )
    gate1.add_argument("--json", action="store_true")

    parity = phase1_cmds.add_parser("parity", help="hnswlib recall-parity check for one dataset")
    _add_protocol_arg(parity)
    parity.add_argument("--dataset", required=True)
    parity.add_argument("--numeric-type", default="fp32")
    parity.add_argument("--M", type=int, default=16)
    parity.add_argument("--n", type=int, default=3000)
    parity.add_argument("--queries", type=int, default=64)
    parity.add_argument("--k", type=int, default=10)

    trace = phase1_cmds.add_parser("trace", help="print one full-detail search trace")
    _add_protocol_arg(trace)
    trace.add_argument("--dataset", required=True)
    trace.add_argument("--numeric-type", default="fp32")
    trace.add_argument("--M", type=int, default=16)
    trace.add_argument("--ef-search", type=int, default=50)
    trace.add_argument("--k", type=int, default=10)
    trace.add_argument("--n", type=int, default=3000)
    trace.add_argument("--query", type=int, default=0, help="index within Qcal")
    trace.add_argument("--events", type=int, default=40, help="how many events to print")

    run = groups.add_parser("run", help="protocol validate, Phase 0 gate, sweep, Phase 1 gate")
    _add_protocol_arg(run)
    run.add_argument("--profile", default="smoke")
    run.add_argument("--no-parity", action="store_true")
    run.add_argument("--quiet", action="store_true")
    return parser


def _cmd_protocol(args: argparse.Namespace) -> int:
    require_frozen = not getattr(args, "allow_draft", False)
    if args.command == "freeze":
        protocol = freeze_protocol(args.protocol, force=args.force)
        print(f"frozen {protocol.protocol_id} at {protocol.doc['frozen_at']}")
        print(f"protocol_hash {protocol.content_hash}")
        return 0
    try:
        protocol = load_protocol(args.protocol, require_frozen=require_frozen, validate=False)
    except ProtocolError as exc:
        print(f"protocol error: {exc}", file=sys.stderr)
        return 1
    if args.command == "hash":
        print(protocol.content_hash)
        return 0
    if args.command == "show":
        print(json.dumps(protocol.doc, indent=2, sort_keys=True))
        return 0
    problems = protocol.problems()
    if problems:
        print(f"{protocol.protocol_id}: {len(problems)} problem(s)")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print(f"{protocol.protocol_id}: valid, status={protocol.status}, hash={protocol.content_hash}")
    print(f"  claims: {', '.join(protocol.claim_families)}")
    print(f"  datasets: {', '.join(protocol.dataset_ids)}")
    print(f"  M grid: {protocol.m_grid}   efSearch grid: {protocol.ef_search_grid}")
    print(f"  profiles: {', '.join(protocol.profile_names)}")
    return 0


def _cmd_phase0(args: argparse.Namespace) -> int:
    result = phase0_gate(args.protocol)
    path = result.write()
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True) if args.json else result.report())
    if not args.json:
        print(f"  written to {path}")
    return 0 if result.passed else 1


def _cmd_phase1(args: argparse.Namespace) -> int:
    if args.command == "sweep":
        summary = run_clean_sweep(
            load_protocol(args.protocol),
            profile_name=args.profile,
            out_dir=args.out,
            dataset_ids=args.datasets,
            n_override=args.n,
            n_queries_override=args.queries,
            trace_sample=args.trace_sample,
            native_parity=not args.no_parity,
            verbose=not args.quiet,
        )
        print(f"cells: {summary['cell_count']} / {summary['expected_cell_count']}")
        print(f"summary: {summary['artifacts']['summary']}")
        return 0
    if args.command == "gate":
        result = phase1_gate(
            args.summary,
            protocol_path=args.protocol,
            require_claim_bearing_profile=args.require_claim_bearing,
        )
        path = result.write()
        print(json.dumps(result.as_dict(), indent=2, sort_keys=True) if args.json else result.report())
        if not args.json:
            print(f"  written to {path}")
        return 0 if result.passed else 1

    protocol = load_protocol(args.protocol)
    dataset = load_dataset(
        protocol, args.dataset, numeric_type=args.numeric_type, n=args.n, n_queries=None
    )
    split = make_split(protocol, dataset.dataset_id, dataset.n_queries)
    params = protocol.hnsw_params(
        M=args.M,
        seed=derive_int(
            protocol.root_seed, "build", dataset.dataset_id, args.numeric_type, f"M={args.M}"
        ),
    )

    if args.command == "parity":
        queries = dataset.queries[split.cal_ids][: args.queries]
        report = parity_report(
            dataset.store,
            queries,
            params,
            ef_grid=protocol.ef_search_grid,
            k=args.k,
            recall_tolerance=float(
                protocol.doc["hnsw"].get("parity_tolerance", {}).get("tolerance", 0.05)
            ),
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("passed", False) else 1

    if args.command == "trace":
        graph = build_index(dataset.store, params)
        query = dataset.queries[split.cal_ids][args.query]
        result = search(
            graph,
            dataset.store,
            query,
            k=args.k,
            ef_search=args.ef_search,
            trace_level=TraceLevel.FULL,
            exposure=ExposurePolicy("threat_model"),
            query_index=int(split.cal_ids[args.query]),
        )
        trace = result.trace
        assert trace is not None
        print(json.dumps(trace.as_dict(), indent=2, sort_keys=True))
        print(f"L(q) size: {len(trace.local_pool())}  |A|/N: "
              f"{knowledge_fraction(len(trace.local_pool()), dataset.n):.4f}")
        print(f"exposed local edges: {len(trace.exposed_edges())}")
        for event in trace.events[: args.events]:
            print(f"  {event.seq:>5} {event.kind:<22} layer={event.layer} node={event.node}")
        return 0
    raise ValueError(f"unknown phase1 command {args.command!r}")


def _cmd_run(args: argparse.Namespace) -> int:
    print("== protocol ==")
    code = _cmd_protocol(argparse.Namespace(command="validate", protocol=args.protocol, allow_draft=False))
    if code:
        return code
    print("\n== Phase 0 gate ==")
    gate0 = phase0_gate(args.protocol)
    gate0.write()
    print(gate0.report())
    if not gate0.passed:
        return 1
    print("\n== Phase 1 clean sweep ==")
    summary = run_clean_sweep(
        load_protocol(args.protocol),
        profile_name=args.profile,
        native_parity=not args.no_parity,
        verbose=not args.quiet,
    )
    print("\n== Phase 1 gate ==")
    gate1 = phase1_gate(Path(summary["artifacts"]["summary"]), protocol_path=args.protocol)
    gate1.write()
    print(gate1.report())
    return 0 if gate1.passed else 1


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.group == "protocol":
        return _cmd_protocol(args)
    if args.group == "phase0":
        return _cmd_phase0(args)
    if args.group == "phase1":
        return _cmd_phase1(args)
    if args.group == "run":
        return _cmd_run(args)
    raise ValueError(f"unknown group {args.group!r}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
