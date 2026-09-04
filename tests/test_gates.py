"""The exit gates themselves: they must fail when the discipline is broken."""

from __future__ import annotations

import copy
import json

import pytest

from braid.gates import phase0_gate, phase1_gate
from braid.protocol import freeze_protocol


def _named(result, name):
    return next(check for check in result.checks if check.name == name)


def test_phase0_gate_passes_on_the_repository_protocol(temp_log):
    result = phase0_gate(log=temp_log)
    assert result.passed, [c.name for c in result.blocking_failures]
    assert _named(result, "four_claim_families_declared_separately").passed
    assert _named(result, "splits_deterministic_disjoint_and_sealed").passed


def test_phase0_gate_rejects_a_tampered_protocol(protocol, tmp_path, temp_log):
    doc = copy.deepcopy(protocol.doc)
    doc["budgets"]["K_grid"] = [1, 2, 4]  # a quiet budget change
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    result = phase0_gate(path, log=temp_log)
    assert not result.passed
    assert not _named(result, "protocol_loads_and_hash_verifies").passed


def test_phase0_gate_requires_the_changelog_to_record_the_active_hash(
    protocol, tmp_path, temp_log
):
    doc = copy.deepcopy(protocol.doc)
    doc["status"] = "draft"
    doc.pop("protocol_hash", None)
    doc.pop("frozen_at", None)
    path = tmp_path / "unlogged.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    freeze_protocol(path)  # a new freeze timestamp gives a new hash
    result = phase0_gate(path, log=temp_log)
    assert not result.passed
    assert not _named(result, "protocol_changelog_records_this_hash").passed


def test_phase0_gate_catches_a_test_split_access_from_a_selection_phase(temp_log):
    temp_log.append(
        "test_split_unseal",
        dataset_id="syn-clusters-d64",
        phase=3,
        reason="peeking while tuning",
        permitted=True,
        allowed_phases=[8, 9, 10],
        n_test=500,
    )
    result = phase0_gate(log=temp_log)
    assert not result.passed
    assert not _named(result, "no_test_split_access_from_selection_phases").passed


def test_phase0_gate_flags_refused_attempts_as_advisory(temp_log):
    temp_log.append(
        "test_split_unseal",
        dataset_id="syn-clusters-d64",
        phase=2,
        reason="refused",
        permitted=False,
        allowed_phases=[8, 9, 10],
        n_test=500,
    )
    result = phase0_gate(log=temp_log)
    assert result.passed  # advisory, not blocking
    assert not _named(result, "no_refused_unseal_attempts").passed


def test_phase1_gate_needs_a_sweep_summary(tmp_path, temp_log):
    result = phase1_gate(tmp_path / "missing.json", log=temp_log)
    assert not result.passed
    assert not _named(result, "clean_sweep_summary_present").passed


@pytest.fixture(scope="module")
def small_sweep(tmp_path_factory):
    from braid.protocol import load_protocol
    from braid.sweep import run_clean_sweep

    out = tmp_path_factory.mktemp("sweep")
    summary = run_clean_sweep(
        load_protocol(),
        profile_name="smoke",
        out_dir=out,
        n_override=400,
        n_queries_override=24,
        trace_sample=4,
        verbose=False,
    )
    return out, summary


def test_sweep_writes_a_complete_grid_with_provenance(small_sweep):
    out, summary = small_sweep
    assert summary["grid_complete"]
    assert summary["cell_count"] == summary["expected_cell_count"]
    assert summary["provenance"]["protocol_status"] == "frozen"
    assert summary["provenance"]["claim_bearing"] is False
    assert summary["queries_used"] == "calibration_only"
    rows = [json.loads(line) for line in (out / "cells.jsonl").read_text().splitlines()]
    assert rows and all(row["query_split"] == "cal" for row in rows)
    assert {row["k"] for row in rows} == {1, 10}


def test_phase1_gate_passes_on_that_sweep(small_sweep, temp_log):
    out, _summary = small_sweep
    result = phase1_gate(out / "summary.json", log=temp_log)
    assert result.passed, [c.name for c in result.blocking_failures]
    assert _named(result, "every_local_edge_traceable_to_an_event").passed
    assert _named(result, "work_counters_stable_across_repeats").passed
    assert _named(result, "exact_answers_independently_checked").passed


def test_phase1_gate_is_advisory_about_a_smoke_profile(small_sweep, temp_log):
    out, _summary = small_sweep
    result = phase1_gate(out / "summary.json", log=temp_log)
    assert not _named(result, "profile_is_claim_bearing").passed
    strict = phase1_gate(
        out / "summary.json", require_claim_bearing_profile=True, log=temp_log
    )
    assert not strict.passed


@pytest.mark.parametrize(
    "path_in_summary,check_name",
    [
        (("counter_stability", "passed"), "work_counters_stable_across_repeats"),
        (("edge_provenance", "passed"), "every_local_edge_traceable_to_an_event"),
        (("rebuild_determinism", "passed"), "graph_build_reproducible_from_seed"),
        (("condition_identity", "passed"), "stale_path_identical_to_clean_when_D_prime_equals_D"),
        (("native_parity", "passed"), "native_recall_parity_within_tolerance"),
    ],
)
def test_phase1_gate_fails_when_a_recorded_check_failed(
    small_sweep, tmp_path, temp_log, path_in_summary, check_name
):
    out, _summary = small_sweep
    summary = json.loads((out / "summary.json").read_text())
    for entry in summary["datasets"].values():
        for cell in entry["per_M"].values():
            section, field = path_in_summary
            cell[section][field] = False
    doctored = tmp_path / "doctored.json"
    doctored.write_text(json.dumps(summary), encoding="utf-8")
    result = phase1_gate(doctored, log=temp_log)
    assert not result.passed
    assert not _named(result, check_name).passed


def test_phase1_gate_rejects_a_sweep_from_another_protocol(small_sweep, tmp_path, temp_log):
    out, _summary = small_sweep
    summary = json.loads((out / "summary.json").read_text())
    summary["provenance"]["protocol_hash"] = "0" * 64
    stale = tmp_path / "stale.json"
    stale.write_text(json.dumps(summary), encoding="utf-8")
    result = phase1_gate(stale, log=temp_log)
    assert not result.passed
    assert not _named(result, "sweep_ran_under_current_protocol_hash").passed
