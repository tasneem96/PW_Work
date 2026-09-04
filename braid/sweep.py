"""Phase 1: the complete clean M x efSearch sweep.

One run produces everything Phase 1's exit gate needs, on calibration queries
only:

* recall@k for every (dataset, numeric type, M, efSearch, k) cell in the active
  profile, against independently cross-checked exact answers;
* the four work quantities of Section 16.3, as distributions;
* e_clean(rho) for every frozen recall target, with right-censoring recorded,
  so that a later Delta_ef is measured against a pre-attack number;
* repeated clean runs, to show the work counters are stable rather than noisy;
* deterministic rebuilds, to show G(D) is a function of the frozen seed;
* full-detail traces on a query sample, to check that every local edge in the
  attacker's view came from an instrumented event, and to report the knowledge
  fraction |A| / N that Section 17 asks for;
* the clean/stale identity check with D' = D;
* recall parity against hnswlib.

Held-out queries are never touched here. Phase 1 is a selection-side phase
under the frozen leakage policy, so opening Qtest would be logged as a
violation and would fail the Phase 0 gate.

One deliberate shortcut: for a given efSearch the search is run once at
k = max(top_k) and truncated for smaller k. The protocol validator enforces
efSearch >= max(top_k), and HNSW's layer-0 search depends on k only through
``ef = max(ef_search, k)``, so truncation is exact rather than an
approximation. It removes a factor of |top_k| from the sweep cost.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import metrics
from .audit import AuditLog, default_log
from .datasets import Dataset, DatasetUnavailable, load_dataset
from .exact import cross_check_exact, exact_topk
from .hnsw.conditions import evaluate_conditions, identity_check
from .hnsw.native import native_version, parity_report
from .hnsw.reference import build_index, search_many
from .hnsw.trace import (
    ExposurePolicy,
    TraceLevel,
    knowledge_fraction,
    merge_local_view,
)
from .paths import SWEEP_DIR, ensure_dirs
from .protocol import Protocol, load_protocol
from .rng import derive_int
from .splits import make_split


@dataclass
class SweepPaths:
    run_dir: Path
    cells: Path
    summary: Path
    traces: Path


def _run_paths(out_dir: Path | None, run_id: str) -> SweepPaths:
    base = Path(out_dir) if out_dir is not None else SWEEP_DIR / run_id
    base.mkdir(parents=True, exist_ok=True)
    return SweepPaths(
        run_dir=base,
        cells=base / "cells.jsonl",
        summary=base / "summary.json",
        traces=base / "trace_sample.json",
    )


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "hnswlib": native_version(),
    }


def _provenance_check(traces) -> dict[str, Any]:
    """Every exposed local edge must name the instrumented event that exposed it.

    This is the concrete form of Phase 1's "every recorded local edge traceable
    to an instrumented event" gate. It also verifies the other direction of the
    exposure rule: L(q) may only contain nodes that a distance-evaluation event
    actually touched.
    """
    checked_edges = 0
    orphan_edges: list[dict[str, Any]] = []
    mismatched_events: list[dict[str, Any]] = []
    untraceable_nodes: list[dict[str, Any]] = []

    for trace in traces:
        if trace.level != TraceLevel.FULL:
            return {
                "checked": False,
                "reason": "provenance can only be checked on FULL traces",
            }
        by_seq = {event.seq: event for event in trace.events}
        for layer, u, v, seq in trace.exposed_edges():
            checked_edges += 1
            event = by_seq.get(seq)
            if event is None:
                orphan_edges.append({"query": trace.query_index, "layer": layer, "u": u, "v": v, "seq": seq})
                continue
            if (
                event.kind != "neighbor_list_exposed"
                or event.node != u
                or event.layer != layer
                or v not in event.detail.get("neighbors", [])
            ):
                mismatched_events.append(
                    {"query": trace.query_index, "layer": layer, "u": u, "v": v, "seq": seq, "kind": event.kind}
                )
        evaluated: set[int] = set()
        for event in trace.events:
            if event.kind == "distance_eval":
                evaluated.update(int(n) for n in event.detail.get("nodes", []))
        missing = set(trace.local_nodes) - evaluated
        if missing:
            untraceable_nodes.append({"query": trace.query_index, "nodes": sorted(missing)})

    return {
        "checked": True,
        "n_traces": len(list(traces)),
        "edges_checked": checked_edges,
        "orphan_edges": orphan_edges[:10],
        "orphan_edge_count": len(orphan_edges),
        "mismatched_events": mismatched_events[:10],
        "mismatched_event_count": len(mismatched_events),
        "untraceable_local_nodes": untraceable_nodes[:10],
        "untraceable_local_node_count": len(untraceable_nodes),
        "passed": not (orphan_edges or mismatched_events or untraceable_nodes),
    }


def _counter_signature(traces) -> list[tuple[int, ...]]:
    """The deterministic part of a run's work counters, per query.

    Latency is excluded on purpose: wall clock is not expected to repeat, and
    pretending otherwise would turn a stability check into a flaky one.
    """
    return [
        (
            t.query_index,
            t.distance_evals,
            t.expansions,
            t.unique_visited,
            t.counter("neighbor_lists_exposed"),
            t.counter("result_prunes"),
            tuple(sorted(t.hops_per_layer.items())),
            tuple(reason for _, reason in t.stop_events),
            tuple(t.result_ids),
        )
        for t in traces
    ]


def run_clean_sweep(
    protocol: Protocol | None = None,
    *,
    profile_name: str = "full",
    out_dir: Path | str | None = None,
    dataset_ids: Sequence[str] | None = None,
    n_override: int | None = None,
    n_queries_override: int | None = None,
    trace_sample: int = 8,
    exact_check_queries: int = 48,
    native_parity: bool = True,
    log: AuditLog | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the clean sweep for one profile and write its artifacts."""
    protocol = protocol or load_protocol()
    profile = protocol.profile(profile_name)
    log = log or default_log()
    ensure_dirs()
    run_id = f"{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{profile_name}-{protocol.content_hash[:8]}"
    paths = _run_paths(Path(out_dir) if out_dir else None, run_id)

    convention = protocol.convention
    k_grid = sorted(profile.top_k_grid)
    k_max = max(k_grid)
    requested = list(dataset_ids) if dataset_ids else list(profile.dataset_ids)
    # A profile may shrink the corpus and query count; it may never grow them.
    n_override = n_override if n_override is not None else profile.max_n
    n_queries_override = (
        n_queries_override if n_queries_override is not None else profile.max_queries
    )

    summary: dict[str, Any] = {
        "run_id": run_id,
        "provenance": protocol.provenance(profile_name),
        "profile": profile.as_dict(),
        "environment": _environment(),
        "convention": convention,
        "phase": 1,
        "queries_used": "calibration_only",
        "datasets": {},
        "skipped_datasets": {},
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    trace_dump: dict[str, Any] = {}
    cells_written = 0

    with paths.cells.open("w", encoding="utf-8") as cells_file:
        for dataset_id in requested:
            for numeric_type in profile.numeric_types:
                key = f"{dataset_id}/{numeric_type}"
                try:
                    dataset = load_dataset(
                        protocol,
                        dataset_id,
                        numeric_type=numeric_type,
                        n=n_override,
                        n_queries=n_queries_override,
                    )
                except DatasetUnavailable as exc:
                    summary["skipped_datasets"][key] = str(exc).splitlines()[0]
                    if verbose:
                        print(f"[skip] {key}: not available here", flush=True)
                    continue
                if verbose:
                    print(f"[dataset] {key}: n={dataset.n} d={dataset.dim}", flush=True)
                summary["datasets"][key] = _sweep_one(
                    protocol=protocol,
                    profile_name=profile_name,
                    dataset=dataset,
                    numeric_type=numeric_type,
                    convention=convention,
                    k_grid=k_grid,
                    k_max=k_max,
                    ef_grid=profile.ef_search_grid,
                    m_grid=profile.m_grid,
                    build_repeats=profile.build_repeats,
                    trace_sample=trace_sample,
                    exact_check_queries=exact_check_queries,
                    native_parity=native_parity,
                    cells_file=cells_file,
                    trace_dump=trace_dump,
                    log=log,
                    verbose=verbose,
                )
                cells_written += summary["datasets"][key]["cell_count"]

    summary["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    summary["cell_count"] = cells_written
    summary["expected_cell_count"] = (
        len(summary["datasets"]) * len(profile.m_grid) * len(profile.ef_search_grid) * len(k_grid)
    )
    summary["grid_complete"] = bool(summary["cell_count"] == summary["expected_cell_count"])
    summary["artifacts"] = {
        "cells": str(paths.cells),
        "summary": str(paths.summary),
        "trace_sample": str(paths.traces),
    }
    paths.summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    paths.traces.write_text(json.dumps(trace_dump, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if verbose:
        print(f"[done] {paths.summary}", flush=True)
    return summary


def _sweep_one(
    *,
    protocol: Protocol,
    profile_name: str,
    dataset: Dataset,
    numeric_type: str,
    convention: str,
    k_grid: Sequence[int],
    k_max: int,
    ef_grid: Sequence[int],
    m_grid: Sequence[int],
    build_repeats: int,
    trace_sample: int,
    exact_check_queries: int,
    native_parity: bool,
    cells_file,
    trace_dump: dict[str, Any],
    log: AuditLog,
    verbose: bool,
) -> dict[str, Any]:
    dataset_id = dataset.dataset_id
    split = make_split(protocol, dataset_id, dataset.n_queries, log=log)
    cal_queries = dataset.queries[split.cal_ids]
    cal_ids = [int(i) for i in split.cal_ids]

    exact_cross_check = cross_check_exact(
        cal_queries, dataset.store, k_max, convention, max_queries=exact_check_queries
    )
    exact = exact_topk(cal_queries, dataset.store, k_max, convention)
    truth = {k: [set(int(i) for i in row[:k]) for row in exact.ids] for k in k_grid}
    truth_top1 = [int(i) for i in exact.nearest]

    out: dict[str, Any] = {
        "dataset": dataset.fingerprint(),
        "split": split.fingerprint(),
        "exact_cross_check": exact_cross_check,
        "exact_tie_rate": float(exact.tie_mask().mean()),
        "per_M": {},
        "cell_count": 0,
    }

    for m_index, m in enumerate(m_grid):
        params = protocol.hnsw_params(
            M=int(m),
            seed=derive_int(protocol.root_seed, "build", dataset_id, numeric_type, f"M={m}"),
        )
        build_started = time.perf_counter_ns()
        graph = build_index(dataset.store, params)
        build_ns = time.perf_counter_ns() - build_started
        structural_problems = graph.validate()
        if verbose:
            print(
                f"  [M={m}] built in {build_ns / 1e9:.1f}s, {graph.edge_count()} edges, "
                f"{len(structural_problems)} structural problems",
                flush=True,
            )

        # G(D) must be a function of the frozen seed, not of run order.
        rebuild_check: dict[str, Any] = {"checked": False}
        if m_index == 0:
            rebuilt = build_index(dataset.store, params)
            rebuild_check = {
                "checked": True,
                "same_structure_hash": bool(rebuilt.structure_hash() == graph.structure_hash()),
                "same_edge_count": bool(rebuilt.edge_count() == graph.edge_count()),
            }
            rebuild_check["passed"] = bool(
                rebuild_check["same_structure_hash"] and rebuild_check["same_edge_count"]
            )

        recall_by_ef: dict[int, dict[int, float]] = {k: {} for k in k_grid}
        counter_stability: dict[str, Any] = {"repeats": int(build_repeats), "unstable_cells": []}

        for ef in ef_grid:
            ids, _distances, traces = search_many(
                graph,
                dataset.store,
                cal_queries,
                k=k_max,
                ef_search=int(ef),
                trace_level=TraceLevel.COUNTERS,
                query_ids=cal_ids,
            )
            signature = _counter_signature(traces)
            stable = True
            for _repeat in range(max(0, int(build_repeats) - 1)):
                _ids2, _d2, traces2 = search_many(
                    graph,
                    dataset.store,
                    cal_queries,
                    k=k_max,
                    ef_search=int(ef),
                    trace_level=TraceLevel.COUNTERS,
                    query_ids=cal_ids,
                )
                if _counter_signature(traces2) != signature:
                    stable = False
            if not stable:
                counter_stability["unstable_cells"].append({"M": int(m), "ef_search": int(ef)})

            work = metrics.work_summary(traces)
            for k in k_grid:
                recall = metrics.recall_at_k(truth[k], ids[:, :k], k)
                recall_by_ef[k][int(ef)] = recall
                row = {
                    "run_profile": profile_name,
                    "protocol_hash": protocol.content_hash,
                    "dataset_id": dataset_id,
                    "numeric_type": numeric_type,
                    "n": dataset.n,
                    "dim": dataset.dim,
                    "convention": convention,
                    "condition": "hnsw_clean",
                    "M": int(m),
                    "ef_construction": params.ef_construction,
                    "ef_search": int(ef),
                    "k": int(k),
                    "n_queries": int(cal_queries.shape[0]),
                    "query_split": "cal",
                    "recall_at_k": recall,
                    "top1_hit_rate": metrics.top1_hit_rate(truth_top1, ids[:, :k]),
                    "correct_in_topk_rate": metrics.contains_rate(truth_top1, ids, k),
                    "work": work,
                    "counters_stable_across_repeats": bool(stable),
                }
                cells_file.write(json.dumps(row, sort_keys=True) + "\n")
                out["cell_count"] += 1

        counter_stability["passed"] = not counter_stability["unstable_cells"]

        # Full-detail traces on a sample: edge provenance and the local view A.
        sample_n = min(int(trace_sample), int(cal_queries.shape[0]))
        sample_ef = int(sorted(ef_grid)[len(list(ef_grid)) // 2])
        _sids, _sdist, sample_traces = search_many(
            graph,
            dataset.store,
            cal_queries[:sample_n],
            k=k_max,
            ef_search=sample_ef,
            trace_level=TraceLevel.FULL,
            exposure=ExposurePolicy("threat_model"),
            query_ids=cal_ids[:sample_n],
        )
        provenance = _provenance_check(sample_traces)
        local_view = merge_local_view(sample_traces)
        local_view_summary = {
            "n_traces": local_view["n_traces"],
            "pool_size": local_view["pool_size"],
            "local_edge_count": local_view["local_edge_count"],
            "knowledge_fraction": knowledge_fraction(local_view["pool_size"], dataset.n),
            "ef_search": sample_ef,
        }
        trace_dump[f"{dataset_id}/{numeric_type}/M={m}"] = {
            "ef_search": sample_ef,
            "traces": [t.as_dict(include_events=False) for t in sample_traces],
            "local_view": local_view_summary,
            "example_events": [e.as_dict() for e in sample_traces[0].events[:40]]
            if sample_traces
            else [],
        }

        # D' = D: the stale path must reproduce the clean path exactly.
        conditions = evaluate_conditions(
            clean_store=dataset.store,
            queries=cal_queries[:sample_n],
            params=params,
            k=k_max,
            ef_search=sample_ef,
            graph_clean=graph,
            graph_rebuilt=graph,
            trace_level=TraceLevel.COUNTERS,
            query_ids=cal_ids[:sample_n],
        )
        identity = identity_check(conditions)
        identity["rebuilt_matches_clean_ids"] = bool(
            np.array_equal(conditions["hnsw_clean"].ids, conditions["hnsw_rebuilt"].ids)
        )
        identity["exact_agrees_on_top1"] = float(
            np.mean(
                conditions["exact"].ids[:, 0] == np.array(truth_top1[:sample_n], dtype=np.int64)
            )
        )

        parity = {"available": False, "reason": "not requested"}
        if native_parity:
            parity = parity_report(
                dataset.store,
                cal_queries[: min(64, cal_queries.shape[0])],
                params,
                ef_grid=ef_grid,
                k=k_max,
                recall_tolerance=float(
                    protocol.doc["hnsw"].get("parity_tolerance", {}).get("tolerance", 0.05)
                ),
                reference_graph=graph,
            )

        out["per_M"][str(int(m))] = {
            "params": params.as_dict(),
            "build_ns": int(build_ns),
            "graph": graph.stats(),
            "structural_problems": structural_problems,
            "rebuild_determinism": rebuild_check,
            "counter_stability": counter_stability,
            "edge_provenance": provenance,
            "local_view": local_view_summary,
            "condition_identity": identity,
            "native_parity": parity,
            "recall_by_ef": {str(k): recall_by_ef[k] for k in k_grid},
            "e_clean": {
                str(k): [
                    metrics.ef_at_recall(recall_by_ef[k], rho, ef_grid)
                    for rho in protocol.recall_targets
                ]
                for k in k_grid
            },
        }
    return out
