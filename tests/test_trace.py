"""Instrumentation: the local view and the work counters must be trustworthy."""

from __future__ import annotations

import pytest

from braid.hnsw.reference import build_index, search, search_many
from braid.hnsw.trace import (
    EVENT_KINDS,
    ExposurePolicy,
    TraceLevel,
    knowledge_fraction,
    merge_local_view,
)
from braid.sweep import _counter_signature, _provenance_check


@pytest.fixture(scope="module")
def graph_and_traces(small_dataset, small_params):
    graph = build_index(small_dataset.store, small_params)
    _ids, _d, traces = search_many(
        graph,
        small_dataset.store,
        small_dataset.queries[:6],
        k=10,
        ef_search=50,
        trace_level=TraceLevel.FULL,
        exposure=ExposurePolicy("threat_model"),
    )
    return graph, traces


def test_local_pool_is_exactly_what_distance_evaluation_touched(graph_and_traces):
    _graph, traces = graph_and_traces
    for trace in traces:
        evaluated = set()
        for event in trace.events:
            if event.kind == "distance_eval":
                evaluated.update(int(n) for n in event.detail["nodes"])
        assert trace.local_pool() == evaluated
        assert trace.counter("unique_visited") == len(trace.local_nodes)


def test_every_local_edge_names_the_event_that_exposed_it(graph_and_traces):
    _graph, traces = graph_and_traces
    report = _provenance_check(traces)
    assert report["checked"] and report["passed"], report
    assert report["edges_checked"] > 0


def test_exposed_edges_are_real_graph_edges(graph_and_traces):
    graph, traces = graph_and_traces
    for trace in traces:
        for layer, u, v, _seq in trace.exposed_edges():
            assert v in graph.neighbors(u, layer)


def test_the_local_view_is_a_small_fraction_of_the_database(graph_and_traces, small_dataset):
    _graph, traces = graph_and_traces
    view = merge_local_view(traces)
    fraction = knowledge_fraction(view["pool_size"], small_dataset.n)
    assert 0.0 < fraction < 1.0
    assert view["local_edge_count"] > 0


def test_wrong_candidate_set_excludes_the_correct_node(graph_and_traces):
    _graph, traces = graph_and_traces
    trace = traces[0]
    correct = trace.result_ids[0]
    assert correct not in trace.wrong_candidates(correct)
    assert trace.wrong_candidates(correct) | {correct} == trace.local_pool()


def test_work_counters_are_internally_consistent(graph_and_traces):
    _graph, traces = graph_and_traces
    for trace in traces:
        work = trace.work_summary()
        assert work["expansions"] >= 1
        assert work["distance_evals"] >= work["unique_visited"]
        assert work["neighbor_lists_exposed"] >= work["expansions"]
        assert work["latency_ns"] > 0
        assert trace.entry_node is not None


def test_stopping_events_are_recorded_with_declared_reasons(graph_and_traces):
    _graph, traces = graph_and_traces
    reasons = {reason for trace in traces for _seq, reason in trace.stop_events}
    assert reasons
    assert reasons <= {
        "candidate_worse_than_furthest",
        "candidate_queue_empty",
        "greedy_local_minimum",
    }


def test_counters_are_identical_across_repeated_runs(small_dataset, small_params):
    graph = build_index(small_dataset.store, small_params)
    signatures = []
    for _ in range(3):
        _ids, _d, traces = search_many(
            graph, small_dataset.store, small_dataset.queries[:8], k=10, ef_search=50
        )
        signatures.append(_counter_signature(traces))
    assert signatures[0] == signatures[1] == signatures[2]


def test_counter_level_records_the_local_view_without_event_bodies(small_dataset, small_params):
    graph = build_index(small_dataset.store, small_params)
    result = search(
        graph,
        small_dataset.store,
        small_dataset.queries[0],
        k=10,
        ef_search=50,
        trace_level=TraceLevel.COUNTERS,
    )
    trace = result.trace
    assert trace is not None
    assert trace.events == []
    assert trace.local_pool()
    assert trace.exposed_edges()
    # provenance is only checkable on full traces, and says so
    assert _provenance_check([trace])["checked"] is False


def test_disabled_tracing_produces_no_trace(small_dataset, small_params):
    graph = build_index(small_dataset.store, small_params)
    result = search(
        graph,
        small_dataset.store,
        small_dataset.queries[0],
        k=5,
        ef_search=20,
        trace_level=TraceLevel.NONE,
    )
    assert result.trace is None


def test_event_kinds_are_closed(graph_and_traces):
    _graph, traces = graph_and_traces
    for trace in traces:
        for event in trace.events:
            assert event.kind in EVENT_KINDS
