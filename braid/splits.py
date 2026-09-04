"""Qcal / Qtest, with the test split sealed at runtime.

Section 5.3 and Phase 0's exit gate both turn on one discipline: bits, target
rules, eligibility rules, surrogate parameters, and success thresholds are
chosen using calibration queries only. Held-out query ids are therefore not
merely "not used"; they are unreachable unless a caller explicitly opens the
seal, naming a phase and a reason, which the audit log records.

The split itself is a deterministic function of the frozen root seed and the
dataset id, so it can be regenerated anywhere without shipping index files.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np

from .audit import AuditLog, default_log
from .protocol import Protocol
from .rng import generator


class LeakageError(RuntimeError):
    """Held-out query ids were requested without opening the seal."""


@dataclass(frozen=True, eq=False)
class QuerySplit:
    """A frozen calibration/test partition of the query ids for one dataset."""

    dataset_id: str
    cal_ids: np.ndarray
    _test_ids: np.ndarray
    root_seed: int
    cal_fraction: float
    test_fraction: float
    sealed: bool = True
    log: AuditLog = field(default_factory=default_log, repr=False, compare=False)

    # -- calibration side (always open) --------------------------------------
    @property
    def n_cal(self) -> int:
        return int(self.cal_ids.size)

    @property
    def n_test(self) -> int:
        """Test *count* is public; the ids are not. Counting cannot tune anything."""
        return int(self._test_ids.size)

    @property
    def n_queries(self) -> int:
        return self.n_cal + self.n_test

    # -- held-out side (sealed) ----------------------------------------------
    def test_ids(self, *, unseal_token: "UnsealToken | None" = None) -> np.ndarray:
        if not self.sealed:
            return self._test_ids.copy()
        if unseal_token is None or not unseal_token.active:
            raise LeakageError(
                f"held-out query ids for {self.dataset_id!r} are sealed. Phase 0's exit gate "
                f"forbids tuning on Qtest; open them with `with split.unseal(phase=..., "
                f"reason=...) as token:` and the access is written to the audit log."
            )
        return self._test_ids.copy()

    @contextmanager
    def unseal(
        self,
        *,
        phase: int,
        reason: str,
        protocol: Protocol | None = None,
    ) -> Iterator["UnsealToken"]:
        """Open the held-out ids, on the record.

        ``phase`` must be one of ``leakage_policy.test_unseal_allowed_phases``
        when a protocol is supplied; every call is logged either way, including
        refused ones, so a refusal cannot be hidden by retrying.
        """
        allowed: list[int] | None = None
        if protocol is not None:
            allowed = [int(p) for p in protocol.doc["leakage_policy"]["test_unseal_allowed_phases"]]
        permitted = allowed is None or int(phase) in allowed
        self.log.append(
            "test_split_unseal",
            dataset_id=self.dataset_id,
            phase=int(phase),
            reason=reason,
            permitted=bool(permitted),
            allowed_phases=allowed,
            n_test=self.n_test,
        )
        if not permitted:
            raise LeakageError(
                f"phase {phase} is a selection phase under the frozen leakage policy and may not "
                f"open Qtest (allowed: {allowed}). The refusal has been logged."
            )
        token = UnsealToken(dataset_id=self.dataset_id, phase=int(phase), reason=reason)
        token.active = True
        try:
            yield token
        finally:
            token.active = False

    # -- integrity -----------------------------------------------------------
    def is_disjoint(self) -> bool:
        return not (set(self.cal_ids.tolist()) & set(self._test_ids.tolist()))

    def covers(self, n_queries: int) -> bool:
        union = set(self.cal_ids.tolist()) | set(self._test_ids.tolist())
        return union == set(range(int(n_queries)))

    def fingerprint(self) -> dict[str, Any]:
        """Describes the split without revealing which ids are held out."""
        import hashlib

        h = hashlib.sha256()
        h.update(self.dataset_id.encode())
        h.update(np.ascontiguousarray(np.sort(self.cal_ids)).tobytes())
        h.update(np.ascontiguousarray(np.sort(self._test_ids)).tobytes())
        return {
            "dataset_id": self.dataset_id,
            "root_seed": self.root_seed,
            "n_cal": self.n_cal,
            "n_test": self.n_test,
            "cal_fraction": self.cal_fraction,
            "test_fraction": self.test_fraction,
            "split_hash": h.hexdigest(),
            "sealed": self.sealed,
        }


@dataclass
class UnsealToken:
    dataset_id: str
    phase: int
    reason: str
    active: bool = False


def make_split(
    protocol: Protocol,
    dataset_id: str,
    n_queries: int,
    *,
    log: AuditLog | None = None,
) -> QuerySplit:
    """Deterministic Qcal/Qtest split for one dataset under the frozen seed."""
    splits = protocol.doc["splits"]
    cal_fraction = float(splits["cal_fraction"])
    test_fraction = float(splits["test_fraction"])
    sealed = bool(protocol.doc["leakage_policy"]["seal_test_split"])

    rng = generator(protocol.root_seed, "split", dataset_id)
    order = rng.permutation(int(n_queries))
    n_cal = int(round(cal_fraction * n_queries))
    n_cal = max(1, min(int(n_queries) - 1, n_cal))
    cal_ids = np.sort(order[:n_cal]).astype(np.int64)
    test_ids = np.sort(order[n_cal:]).astype(np.int64)
    return QuerySplit(
        dataset_id=dataset_id,
        cal_ids=cal_ids,
        _test_ids=test_ids,
        root_seed=protocol.root_seed,
        cal_fraction=cal_fraction,
        test_fraction=test_fraction,
        sealed=sealed,
        log=log or default_log(),
    )
