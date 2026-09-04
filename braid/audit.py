"""Append-only audit log.

Phase 0's exit gate is a negative claim: no hyperparameter, target rule,
eligibility rule, or success metric was tuned using Qtest. A negative claim
cannot be proved by reading code, so the test split is sealed at runtime
(:mod:`braid.splits`) and every attempt to open it writes a record here. The
gate then reads the log instead of trusting anybody's memory.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from .paths import LEAKAGE_AUDIT_LOG


@dataclass
class AuditLog:
    path: Path = field(default_factory=lambda: LEAKAGE_AUDIT_LOG)

    def append(self, kind: str, **detail: Any) -> dict[str, Any]:
        record = {
            "ts": time.time(),
            "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "pid": os.getpid(),
            "kind": kind,
            "detail": detail,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return iter(())
        def _iter() -> Iterator[dict[str, Any]]:
            with self.path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        yield json.loads(line)
        return _iter()

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [r for r in self.records() if r.get("kind") == kind]


_DEFAULT = AuditLog()


def default_log() -> AuditLog:
    return _DEFAULT
