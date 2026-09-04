"""Search-trace instrumentation: the attacker's local view and the operator's counters.

This module is the whole point of Phase 1. Two different consumers read it:

* the threat model (Sections 4 and 4.1) needs L(q), the set of vectors whose
  distances were evaluated, and N_local(u), the neighbour lists that the search
  actually exposed. Nothing else about the graph may leak into attack code;
* the work-amplification measurement (Section 16.3) needs expanded nodes,
  unique visited nodes, distance evaluations, and latency, kept apart from each
  other rather than merged into one "cost" number.

Every recorded local edge carries the sequence number of the event that
exposed it, which is exactly what Phase 1's exit gate checks: no edge may
appear in the attacker's view unless an instrumented event put it there.

Latency recorded here is Python-level wall clock. It is adequate for detecting
gross regressions and useless as the paper's latency claim; Section 16.3's
timing discipline requires the native implementation on controlled hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Iterable

import numpy as np


class TraceLevel(IntEnum):
    """How much is recorded.

    ``NONE``
        nothing; used for warm-up runs.
    ``COUNTERS``
        work counters, stop reasons, L(q) and the local neighbour lists, but no
        per-event log. This is the sweep default.
    ``FULL``
        every event with a sequence number, so edge provenance is checkable.
    """

    NONE = 0
    COUNTERS = 1
    FULL = 2


#: Event kinds. Anything the attacker or the operator may read must originate
#: in one of these; there is no other channel out of the search.
EVENT_KINDS = (
    "query_start",
    "entry_point",
    "layer_enter",
    "layer_exit",
    "greedy_hop",
    "neighbor_list_exposed",
    "distance_eval",
    "candidate_push",
    "candidate_pop",
    "visited_add",
    "result_prune",
    "stop",
    "query_end",
)

#: Reasons a layer search stops. Recorded separately because "ran out of
#: candidates" and "best candidate is worse than the current worst result" are
#: different failure modes under corruption.
STOP_REASONS = ("candidate_worse_than_furthest", "candidate_queue_empty", "greedy_local_minimum")


@dataclass
class ExposurePolicy:
    """What the trace is allowed to reveal.

    ``threat_model`` is the gray-box setting of Section 5.1: only nodes and
    edges touched during the search. ``white_box`` exists solely for the global
    upper-bound baseline of Section 17 and is recorded in every artifact so a
    white-box number can never be reported as a gray-box one.
    """

    mode: str = "threat_model"

    def __post_init__(self) -> None:
        if self.mode not in ("threat_model", "white_box"):
            raise ValueError(f"unknown exposure mode {self.mode!r}")

    @property
    def local_only(self) -> bool:
        return self.mode == "threat_model"


@dataclass(frozen=True)
class TraceEvent:
    seq: int
    kind: str
    layer: int | None = None
    node: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "kind": self.kind,
            "layer": self.layer,
            "node": self.node,
            **({"detail": self.detail} if self.detail else {}),
        }


@dataclass
class QueryTrace:
    """One query's instrumented search."""

    query_index: int
    level: TraceLevel
    exposure: ExposurePolicy
    entry_node: int | None = None
    entry_layer: int | None = None
    result_ids: tuple[int, ...] = ()
    result_distances: tuple[float, ...] = ()
    latency_ns: int = 0
    counters: dict[str, int] = field(default_factory=dict)
    hops_per_layer: dict[int, int] = field(default_factory=dict)
    stop_events: list[tuple[int, str]] = field(default_factory=list)
    events: list[TraceEvent] = field(default_factory=list)
    #: L(q): nodes whose distance was evaluated, in first-touch order.
    local_nodes: list[int] = field(default_factory=list)
    #: N_local(u) per layer, with the exposing event sequence per edge.
    local_edges: dict[tuple[int, int], list[tuple[int, int]]] = field(default_factory=dict)

    # -- attacker-visible view ----------------------------------------------
    def local_pool(self) -> set[int]:
        """L(q) from Section 4."""
        return set(self.local_nodes)

    def wrong_candidates(self, correct_id: int) -> set[int]:
        """R(q) = L(q) \\ {i*(q)} from Section 4."""
        return self.local_pool() - {int(correct_id)}

    def neighbors(self, node: int, layer: int) -> list[int]:
        """N_local(u) at one layer, as exposed by this query."""
        return [v for v, _ in self.local_edges.get((int(layer), int(node)), [])]

    def exposed_edges(self) -> list[tuple[int, int, int, int]]:
        """(layer, u, v, exposing_event_seq) for every exposed local edge."""
        return [
            (layer, u, v, seq)
            for (layer, u), items in self.local_edges.items()
            for v, seq in items
        ]

    # -- operator-visible view ----------------------------------------------
    def counter(self, name: str) -> int:
        return int(self.counters.get(name, 0))

    @property
    def distance_evals(self) -> int:
        return self.counter("distance_evals")

    @property
    def expansions(self) -> int:
        return self.counter("expansions")

    @property
    def unique_visited(self) -> int:
        return self.counter("unique_visited")

    def work_summary(self) -> dict[str, int | float]:
        return {
            "distance_evals": self.distance_evals,
            "expansions": self.expansions,
            "unique_visited": self.unique_visited,
            "neighbor_lists_exposed": self.counter("neighbor_lists_exposed"),
            "result_prunes": self.counter("result_prunes"),
            "greedy_hops": sum(self.hops_per_layer.values()),
            "latency_ns": int(self.latency_ns),
        }

    def as_dict(self, *, include_events: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "query_index": self.query_index,
            "trace_level": int(self.level),
            "exposure_mode": self.exposure.mode,
            "entry_node": self.entry_node,
            "entry_layer": self.entry_layer,
            "result_ids": list(self.result_ids),
            "result_distances": [float(d) for d in self.result_distances],
            "work": self.work_summary(),
            "hops_per_layer": {str(k): v for k, v in sorted(self.hops_per_layer.items())},
            "stop_events": [{"seq": s, "reason": r} for s, r in self.stop_events],
            "local_pool_size": len(self.local_nodes),
            "local_edge_count": len(self.exposed_edges()),
        }
        if include_events:
            out["events"] = [e.as_dict() for e in self.events]
        return out


class TraceRecorder:
    """Collects one query's trace. One recorder per query, never shared."""

    def __init__(
        self,
        query_index: int = -1,
        level: TraceLevel = TraceLevel.COUNTERS,
        exposure: ExposurePolicy | None = None,
    ) -> None:
        self.level = TraceLevel(level)
        self.trace = QueryTrace(
            query_index=int(query_index),
            level=self.level,
            exposure=exposure or ExposurePolicy(),
        )
        self._seq = 0
        self._seen: set[int] = set()

    # -- low level -----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.level != TraceLevel.NONE

    def event(self, kind: str, *, layer: int | None = None, node: int | None = None, **detail: Any) -> int:
        """Record an event and return its sequence number.

        The sequence number is allocated at every level except ``NONE`` so that
        edge provenance still refers to a real ordering when the event bodies
        are not being kept.
        """
        if not self.enabled:
            return -1
        if kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {kind!r}")
        self._seq += 1
        if self.level == TraceLevel.FULL:
            self.trace.events.append(
                TraceEvent(seq=self._seq, kind=kind, layer=layer, node=node, detail=dict(detail))
            )
        return self._seq

    def count(self, name: str, amount: int = 1) -> None:
        if not self.enabled:
            return
        self.trace.counters[name] = self.trace.counters.get(name, 0) + int(amount)

    # -- the four instrumented channels -------------------------------------
    def note_distance_evals(
        self, nodes: Iterable[int], distances: Iterable[float], *, layer: int, context: str
    ) -> int:
        """Distance evaluations: this is what defines L(q)."""
        node_list = [int(v) for v in nodes]
        if not self.enabled:
            return -1
        seq = self.event(
            "distance_eval", layer=layer, node=None, context=context, nodes=node_list
        )
        self.count("distance_evals", len(node_list))
        self.count(f"distance_evals::{context}", len(node_list))
        for node in node_list:
            if node not in self._seen:
                self._seen.add(node)
                self.trace.local_nodes.append(node)
                self.count("unique_visited")
                self.event("visited_add", layer=layer, node=node)
        return seq

    def note_neighbor_list(self, node: int, layer: int, neighbors: Iterable[int]) -> int:
        """Neighbour-list exposure: this is what defines N_local(u)."""
        neighbor_list = [int(v) for v in neighbors]
        if not self.enabled:
            return -1
        seq = self.event(
            "neighbor_list_exposed", layer=layer, node=int(node), neighbors=neighbor_list
        )
        self.count("neighbor_lists_exposed")
        key = (int(layer), int(node))
        bucket = self.trace.local_edges.setdefault(key, [])
        known = {v for v, _ in bucket}
        for v in neighbor_list:
            if v not in known:
                bucket.append((v, seq))
                known.add(v)
        return seq

    def note_expansion(self, node: int, layer: int, distance: float) -> int:
        self.count("expansions")
        return self.event("candidate_pop", layer=layer, node=int(node), distance=float(distance))

    def note_candidate_push(self, node: int, layer: int, distance: float) -> int:
        self.count("candidate_pushes")
        return self.event("candidate_push", layer=layer, node=int(node), distance=float(distance))

    def note_prune(self, node: int, layer: int, distance: float) -> int:
        self.count("result_prunes")
        return self.event("result_prune", layer=layer, node=int(node), distance=float(distance))

    def note_greedy_hop(self, frm: int, to: int, layer: int) -> int:
        self.trace.hops_per_layer[int(layer)] = self.trace.hops_per_layer.get(int(layer), 0) + 1
        self.count("greedy_hops")
        return self.event("greedy_hop", layer=layer, node=int(to), previous=int(frm))

    def note_stop(self, reason: str, *, layer: int, node: int | None = None, **detail: Any) -> int:
        if reason not in STOP_REASONS:
            raise ValueError(f"unknown stop reason {reason!r}")
        seq = self.event("stop", layer=layer, node=node, reason=reason, **detail)
        self.count(f"stop::{reason}")
        if self.enabled:
            self.trace.stop_events.append((seq, reason))
        return seq

    def note_entry(self, node: int, layer: int, distance: float) -> int:
        self.trace.entry_node = int(node)
        self.trace.entry_layer = int(layer)
        return self.event("entry_point", layer=layer, node=int(node), distance=float(distance))

    # -- finishing -----------------------------------------------------------
    def finish(self, ids: np.ndarray, distances: np.ndarray, latency_ns: int) -> QueryTrace:
        self.trace.result_ids = tuple(int(i) for i in np.asarray(ids).reshape(-1))
        self.trace.result_distances = tuple(
            float(d) for d in np.asarray(distances, dtype=np.float64).reshape(-1)
        )
        self.trace.latency_ns = int(latency_ns)
        self.event("query_end", n_results=len(self.trace.result_ids))
        return self.trace


NULL_RECORDER = TraceRecorder(level=TraceLevel.NONE)


def merge_local_view(traces: Iterable[QueryTrace]) -> dict[str, Any]:
    """The pooled gray-box view A = union over Qcal of L(q) (Section 4).

    Returns the pool, the pooled local neighbour lists, and the knowledge
    fraction |A| / N, which Section 17 requires to be reported next to attack
    strength rather than mentioned once in prose.
    """
    pool: set[int] = set()
    edges: dict[tuple[int, int], set[int]] = {}
    n_traces = 0
    for trace in traces:
        n_traces += 1
        pool |= trace.local_pool()
        for (layer, u), items in trace.local_edges.items():
            edges.setdefault((layer, u), set()).update(v for v, _ in items)
    return {
        "n_traces": n_traces,
        "pool": sorted(pool),
        "pool_size": len(pool),
        "local_edges": {f"{layer}:{u}": sorted(vs) for (layer, u), vs in edges.items()},
        "local_edge_count": sum(len(vs) for vs in edges.values()),
    }


def knowledge_fraction(pool_size: int, n: int) -> float:
    """|A| / N (Section 4)."""
    return float(pool_size) / float(n) if n else 0.0
