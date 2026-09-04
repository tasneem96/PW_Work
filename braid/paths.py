"""Canonical repository paths.

Everything written by a run goes under ``results/`` so that a clean checkout
plus a protocol file is enough to reproduce a run from scratch.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"
DOC_DIR = REPO_ROOT / "docs"
RESULTS_DIR = REPO_ROOT / "results"

AUDIT_DIR = RESULTS_DIR / "audit"
SWEEP_DIR = RESULTS_DIR / "sweep"
GATE_DIR = RESULTS_DIR / "gates"
TRACE_DIR = RESULTS_DIR / "traces"

DEFAULT_PROTOCOL = CONFIG_DIR / "protocol_v2.json"
LEAKAGE_AUDIT_LOG = AUDIT_DIR / "leakage_audit.jsonl"


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, AUDIT_DIR, SWEEP_DIR, GATE_DIR, TRACE_DIR):
        path.mkdir(parents=True, exist_ok=True)
