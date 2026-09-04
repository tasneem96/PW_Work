"""Exit gates for Phase 0 and Phase 1.

A phase is not finished because its code runs. It is finished when its exit
gate passes, and the gate is machine-checked so that "we froze the protocol"
and "the counters are stable" are claims about artifacts rather than about
intentions. Each check is either ``blocking`` (the phase cannot be declared
complete) or ``advisory`` (worth knowing, does not block; a missing optional
dependency is the usual case).

Phase 0 gate: no hyperparameter, target rule, eligibility rule, or success
metric can be tuned using Qtest.
Phase 1 gate: exact-search answers independently checked, stable work counters
across repeated clean runs, and every recorded local edge traceable to an
instrumented event.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .audit import AuditLog, default_log
from .datasets import DatasetUnavailable, load_dataset
from .paths import DOC_DIR, GATE_DIR, SWEEP_DIR, ensure_dirs
from .protocol import (
    REQUIRED_CLAIM_FAMILIES,
    Protocol,
    ProtocolError,
    load_protocol,
)
from .splits import make_split

BLOCKING = "blocking"
ADVISORY = "advisory"


@dataclass
class Check:
    name: str
    passed: bool
    severity: str = BLOCKING
    detail: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": bool(self.passed),
            "severity": self.severity,
            "detail": self.detail,
        }


@dataclass
class GateResult:
    phase: int
    checks: list[Check] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def add(self, name: str, passed: bool, *, severity: str = BLOCKING, detail: Any = None) -> None:
        self.checks.append(Check(name=name, passed=passed, severity=severity, detail=detail))

    @property
    def blocking_failures(self) -> list[Check]:
        return [c for c in self.checks if c.severity == BLOCKING and not c.passed]

    @property
    def advisories(self) -> list[Check]:
        return [c for c in self.checks if c.severity == ADVISORY and not c.passed]

    @property
    def passed(self) -> bool:
        return not self.blocking_failures

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "passed": self.passed,
            "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "context": self.context,
            "n_checks": len(self.checks),
            "blocking_failures": [c.name for c in self.blocking_failures],
            "advisories": [c.name for c in self.advisories],
            "checks": [c.as_dict() for c in self.checks],
        }

    def write(self, path: Path | None = None) -> Path:
        ensure_dirs()
        target = path or GATE_DIR / f"phase{self.phase}.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    def report(self) -> str:
        lines = [f"Phase {self.phase} exit gate: {'PASS' if self.passed else 'FAIL'}"]
        for check in self.checks:
            if check.passed:
                mark = "pass"
            else:
                mark = "FAIL" if check.severity == BLOCKING else "warn"
            lines.append(f"  [{mark}] {check.name}")
            if not check.passed and check.detail is not None:
                detail = json.dumps(check.detail, sort_keys=True)
                lines.append(f"         {detail[:500]}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# Phase 0
# --------------------------------------------------------------------------

def phase0_gate(
    protocol_path: Path | str | None = None,
    *,
    log: AuditLog | None = None,
) -> GateResult:
    """Check that the protocol is frozen and that nothing was tuned on Qtest."""
    log = log or default_log()
    result = GateResult(phase=0)

    try:
        protocol = load_protocol(protocol_path, require_frozen=True, validate=False)
        result.add("protocol_loads_and_hash_verifies", True, detail={"hash": protocol.content_hash})
    except ProtocolError as exc:
        result.add("protocol_loads_and_hash_verifies", False, detail=str(exc))
        return result

    result.context = {
        "protocol_id": protocol.protocol_id,
        "protocol_hash": protocol.content_hash,
        "frozen_at": protocol.doc.get("frozen_at"),
        "protocol_path": str(protocol.path),
    }

    problems = protocol.problems()
    result.add("protocol_schema_valid", not problems, detail=problems or None)
    result.add("protocol_status_frozen", protocol.frozen, detail={"status": protocol.status})

    # the four claim families, stated separately
    families = protocol.claim_families
    missing_families = [f for f in REQUIRED_CLAIM_FAMILIES if f not in families]
    result.add(
        "four_claim_families_declared_separately",
        not missing_families,
        detail={"declared": families, "missing": missing_families},
    )
    incomplete = [
        c["id"]
        for c in protocol.doc["claims"]
        if not (c.get("primary_metric") and c.get("exit_gate") and c.get("statement"))
    ]
    result.add(
        "every_claim_has_metric_and_exit_gate", not incomplete, detail={"incomplete": incomplete}
    )

    # frozen declarations that later phases are forbidden to touch
    declared = {
        "datasets": bool(protocol.doc["datasets"]),
        "embedding_models": bool(protocol.doc["embedding_models"]),
        "numeric_types": bool(protocol.numeric_types),
        "hnsw_implementation": bool(protocol.doc["system"].get("hnsw_implementation")),
        "distance_convention": bool(protocol.convention),
        "M_grid": bool(protocol.m_grid),
        "ef_search_grid": bool(protocol.ef_search_grid),
        "seeds": "root" in protocol.doc["seeds"],
        "split_policy": bool(protocol.doc["splits"]),
        "budget_grids": all(protocol.doc["budgets"].get(k) for k in ("K_grid", "BV_grid", "BF_grid")),
        "finite_value_policy": bool(protocol.doc["bitflip_policy"].get("finite_only")),
        "bit_classes": bool(protocol.doc["bitflip_policy"].get("bit_classes")),
        "target_rules": bool(protocol.doc["targets"].get("rules")),
        "target_eligibility": bool(protocol.doc["targets"].get("eligibility")),
        "recall_targets": bool(protocol.recall_targets),
        "primary_comparisons": bool(protocol.doc["comparisons"].get("conditions")),
    }
    result.add(
        "all_required_declarations_frozen",
        all(declared.values()),
        detail={k: v for k, v in declared.items() if not v} or declared,
    )

    rules = protocol.doc["targets"]["rules"]
    bad_rules = [
        r.get("id")
        for r in rules
        if not (r.get("deterministic") and r.get("frozen_before_selection"))
    ]
    result.add(
        "target_rules_deterministic_and_pre_frozen",
        not bad_rules,
        detail={"rules": [r.get("id") for r in rules], "violations": bad_rules},
    )

    # the split is deterministic, disjoint, and covers every query
    split_details: dict[str, Any] = {}
    split_ok = True
    for dataset_id in protocol.dataset_ids:
        spec = protocol.dataset(dataset_id)
        n_queries = int(spec["n_queries"])
        first = make_split(protocol, dataset_id, n_queries, log=log)
        second = make_split(protocol, dataset_id, n_queries, log=log)
        same = first.fingerprint()["split_hash"] == second.fingerprint()["split_hash"]
        disjoint = first.is_disjoint()
        covering = first.covers(n_queries)
        sealed = first.sealed
        split_details[dataset_id] = {
            "deterministic": same,
            "disjoint": disjoint,
            "covers_all_queries": covering,
            "test_split_sealed": sealed,
            "n_cal": first.n_cal,
            "n_test": first.n_test,
        }
        split_ok = split_ok and same and disjoint and covering and sealed
    result.add("splits_deterministic_disjoint_and_sealed", split_ok, detail=split_details)

    # the negative claim: Qtest was never opened by a selection phase
    selection_phases = set(int(p) for p in protocol.doc["leakage_policy"]["selection_phases"])
    unseals = log.of_kind("test_split_unseal")
    violations = [
        r for r in unseals if int(r["detail"].get("phase", -1)) in selection_phases and r["detail"].get("permitted")
    ]
    refused = [r for r in unseals if not r["detail"].get("permitted")]
    result.add(
        "no_test_split_access_from_selection_phases",
        not violations,
        detail={
            "unseal_events": len(unseals),
            "violations": violations[:10],
            "refused_attempts": len(refused),
            "audit_log": str(log.path),
        },
    )
    result.add(
        "no_refused_unseal_attempts",
        not refused,
        severity=ADVISORY,
        detail={"refused": refused[:10]},
    )

    # datasets must at least be resolvable, or explicitly flagged unavailable
    availability: dict[str, str] = {}
    for dataset_id in protocol.dataset_ids:
        try:
            dataset = load_dataset(protocol, dataset_id, n=min(64, int(protocol.dataset(dataset_id)["n"])), n_queries=8)
            availability[dataset_id] = f"available ({dataset.n} x {dataset.dim})"
        except DatasetUnavailable:
            availability[dataset_id] = "declared, not available in this environment"
        except Exception as exc:  # a broken declaration, unlike a missing download
            availability[dataset_id] = f"ERROR: {type(exc).__name__}: {exc}"
    broken = {k: v for k, v in availability.items() if v.startswith("ERROR")}
    result.add("declared_datasets_resolvable", not broken, detail=availability)

    unavailable = {k: v for k, v in availability.items() if v.startswith("declared, not")}
    result.add(
        "all_declared_datasets_present",
        not unavailable,
        severity=ADVISORY,
        detail={
            "unavailable": unavailable,
            "note": "external corpora are declared for Phase 8; Phase 0/1 runs use the synthetic ones",
        },
    )

    # a frozen protocol needs a changelog entry naming it
    changelog = DOC_DIR / "protocol_changelog.md"
    text = changelog.read_text(encoding="utf-8") if changelog.exists() else ""
    result.add(
        "protocol_changelog_records_this_hash",
        bool(text) and protocol.protocol_id in text and (protocol.content_hash[:12] in text),
        detail={
            "changelog": str(changelog),
            "expected_hash_prefix": protocol.content_hash[:12],
            "exists": changelog.exists(),
        },
    )
    return result


# --------------------------------------------------------------------------
# Phase 1
# --------------------------------------------------------------------------

def latest_sweep_summary(sweep_dir: Path | str | None = None) -> Path | None:
    base = Path(sweep_dir) if sweep_dir is not None else SWEEP_DIR
    if not base.exists():
        return None
    candidates = sorted(base.glob("*/summary.json"))
    if not candidates:
        candidates = sorted(base.glob("summary.json"))
    return candidates[-1] if candidates else None


def phase1_gate(
    summary_path: Path | str | None = None,
    *,
    protocol_path: Path | str | None = None,
    require_claim_bearing_profile: bool = False,
    log: AuditLog | None = None,
) -> GateResult:
    """Check the instrumentation and baseline against a clean-sweep summary."""
    log = log or default_log()
    result = GateResult(phase=1)

    try:
        protocol = load_protocol(protocol_path, require_frozen=True, validate=True)
    except ProtocolError as exc:
        result.add("protocol_frozen_and_valid", False, detail=str(exc))
        return result
    result.add("protocol_frozen_and_valid", True, detail={"hash": protocol.content_hash})

    path = Path(summary_path) if summary_path else latest_sweep_summary()
    if path is None or not Path(path).exists():
        result.add(
            "clean_sweep_summary_present",
            False,
            detail="no sweep summary found; run `python -m braid phase1 sweep` first",
        )
        return result
    summary = json.loads(Path(path).read_text(encoding="utf-8"))
    result.add("clean_sweep_summary_present", True, detail={"path": str(path)})
    result.context = {
        "summary": str(path),
        "run_id": summary.get("run_id"),
        "profile": summary.get("profile", {}).get("name"),
        "claim_bearing": summary.get("provenance", {}).get("claim_bearing"),
        "protocol_hash": protocol.content_hash,
    }

    result.add(
        "sweep_ran_under_current_protocol_hash",
        summary.get("provenance", {}).get("protocol_hash") == protocol.content_hash,
        detail={
            "sweep": summary.get("provenance", {}).get("protocol_hash"),
            "current": protocol.content_hash,
        },
    )
    result.add(
        "sweep_used_calibration_queries_only",
        summary.get("queries_used") == "calibration_only",
        detail={"queries_used": summary.get("queries_used")},
    )
    result.add(
        "profile_is_claim_bearing",
        bool(summary.get("provenance", {}).get("claim_bearing")),
        severity=BLOCKING if require_claim_bearing_profile else ADVISORY,
        detail={
            "profile": summary.get("profile", {}).get("name"),
            "note": "a smoke profile exercises the pipeline; it cannot support a claim",
        },
    )

    datasets = summary.get("datasets", {})
    result.add("at_least_one_dataset_swept", bool(datasets), detail={"datasets": list(datasets)})
    if not datasets:
        return result

    exact_failures: dict[str, Any] = {}
    native_exact_missing: list[str] = []
    structural: dict[str, Any] = {}
    counters: dict[str, Any] = {}
    rebuilds: dict[str, Any] = {}
    provenance: dict[str, Any] = {}
    identity: dict[str, Any] = {}
    parity_failures: dict[str, Any] = {}
    parity_missing: list[str] = []
    ef_missing: dict[str, Any] = {}
    knowledge: dict[str, float] = {}

    for key, entry in datasets.items():
        check = entry.get("exact_cross_check", {})
        if not check.get("passed"):
            exact_failures[key] = check
        comparisons = check.get("comparisons", {})
        if not comparisons.get("hnswlib_brute_force", {}).get("available"):
            native_exact_missing.append(key)

        for m, cell in entry.get("per_M", {}).items():
            tag = f"{key}/M={m}"
            if cell.get("structural_problems"):
                structural[tag] = cell["structural_problems"][:5]
            stability = cell.get("counter_stability", {})
            if not stability.get("passed"):
                counters[tag] = stability
            rebuild = cell.get("rebuild_determinism", {})
            if rebuild.get("checked") and not rebuild.get("passed"):
                rebuilds[tag] = rebuild
            prov = cell.get("edge_provenance", {})
            if not (prov.get("checked") and prov.get("passed")):
                provenance[tag] = prov
            ident = cell.get("condition_identity", {})
            if not (ident.get("checked") and ident.get("passed") and ident.get("rebuilt_matches_clean_ids")):
                identity[tag] = ident
            parity = cell.get("native_parity", {})
            if not parity.get("available"):
                parity_missing.append(tag)
            elif not parity.get("passed"):
                parity_failures[tag] = {
                    "worst_recall_gap": parity.get("worst_recall_gap"),
                    "tolerance": parity.get("recall_tolerance"),
                }
            e_clean = cell.get("e_clean", {})
            expected = len(protocol.recall_targets)
            for k, entries in e_clean.items():
                if len(entries) != expected:
                    ef_missing[f"{tag}/k={k}"] = {"expected": expected, "got": len(entries)}
            view = cell.get("local_view", {})
            if "knowledge_fraction" in view:
                knowledge[tag] = float(view["knowledge_fraction"])

    result.add(
        "exact_answers_independently_checked",
        not exact_failures,
        detail=exact_failures or {"note": "matrix, per-query loop, and hnswlib brute force agree"},
    )
    result.add(
        "third_party_exact_check_available",
        not native_exact_missing,
        severity=ADVISORY,
        detail={"missing_for": native_exact_missing},
    )
    result.add("graph_structure_invariants_hold", not structural, detail=structural or None)
    result.add("work_counters_stable_across_repeats", not counters, detail=counters or None)
    result.add("graph_build_reproducible_from_seed", not rebuilds, detail=rebuilds or None)
    result.add("every_local_edge_traceable_to_an_event", not provenance, detail=provenance or None)
    result.add(
        "stale_path_identical_to_clean_when_D_prime_equals_D",
        not identity,
        detail=identity or None,
    )
    result.add(
        "e_clean_computed_for_every_recall_target",
        not ef_missing,
        detail=ef_missing or {"recall_targets": protocol.recall_targets},
    )
    result.add(
        "grid_complete_for_active_profile",
        bool(summary.get("grid_complete")),
        detail={
            "cells": summary.get("cell_count"),
            "expected": summary.get("expected_cell_count"),
        },
    )
    result.add(
        "native_recall_parity_within_tolerance",
        not parity_failures,
        detail=parity_failures or None,
    )
    result.add(
        "native_parity_available",
        not parity_missing,
        severity=ADVISORY,
        detail={"missing_for": parity_missing},
    )
    result.add(
        "knowledge_fraction_reported",
        bool(knowledge),
        detail={"knowledge_fraction_by_cell": knowledge},
    )

    selection_phases = set(int(p) for p in protocol.doc["leakage_policy"]["selection_phases"])
    unseals = [
        r
        for r in log.of_kind("test_split_unseal")
        if int(r["detail"].get("phase", -1)) in selection_phases
    ]
    result.add(
        "no_test_split_access_during_phase1",
        not unseals,
        detail={"events": unseals[:10]},
    )
    return result


def run_gate(phase: int, **kwargs: Any) -> GateResult:
    if int(phase) == 0:
        return phase0_gate(**kwargs)
    if int(phase) == 1:
        return phase1_gate(**kwargs)
    raise ValueError(
        f"phase {phase} has no gate in this repository; phases 0 and 1 are implemented"
    )
