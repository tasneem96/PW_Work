"""Phase 0: the frozen protocol.

The protocol file is the single declaration of everything that must be fixed
before any attack bit is chosen: datasets, embedding models, numeric types,
HNSW implementation and version, distance convention, M values, the efSearch
grid, seeds, the Qcal/Qtest split, the BV/BF/K grids, the finite-value policy,
the target-selection rules, the recall targets, the primary exact/stale/rebuilt
comparisons, and the four claim families stated separately.

Freezing is enforced, not requested. The file carries a SHA-256 over its own
canonical content; loading recomputes that hash and refuses to hand out a
protocol whose content drifted from what was frozen. Changing a frozen value
means writing a new versioned protocol file and a changelog entry, which leaves
a visible trail instead of a silent edit.

Run profiles exist for cheap smoke runs. A profile may only *subset* frozen
grids, never introduce a value that was not frozen, and every artifact records
which profile produced it. A smoke-profile run cannot support a claim.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .paths import DEFAULT_PROTOCOL
from .similarity import CONVENTIONS
from .vectors import BIT_WIDTH, NUMERIC_TYPES

HASH_FIELD = "protocol_hash"
STATUS_FROZEN = "frozen"
STATUS_DRAFT = "draft"

#: The claim families that Phase 0 requires to be stated separately.
REQUIRED_CLAIM_FAMILIES = (
    "untargeted_retrieval_failure",
    "targeted_route_steering",
    "targeted_retrieval",
    "work_amplification",
)

REQUIRED_CONDITIONS = ("exact", "hnsw_clean", "hnsw_stale", "hnsw_rebuilt")

REQUIRED_TOP_LEVEL = (
    "schema_version",
    "protocol_id",
    "status",
    "claims",
    "system",
    "datasets",
    "embedding_models",
    "hnsw",
    "seeds",
    "splits",
    "budgets",
    "bitflip_policy",
    "targets",
    "recall_targets",
    "comparisons",
    "surrogate",
    "profiles",
    "leakage_policy",
)


class ProtocolError(RuntimeError):
    """The protocol file is missing, invalid, or internally inconsistent."""


class FreezeViolation(ProtocolError):
    """The protocol content does not match the hash it was frozen with."""


def canonical_json(doc: Any) -> str:
    """Stable serialization: sorted keys, no insignificant whitespace."""
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compute_protocol_hash(doc: dict[str, Any]) -> str:
    payload = {k: v for k, v in doc.items() if k != HASH_FIELD}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _is_sorted_unique(values: Sequence[Any]) -> bool:
    return all(a < b for a, b in zip(values, values[1:]))


def _subset(candidate: Iterable[Any], frozen: Iterable[Any]) -> bool:
    frozen_set = list(frozen)
    return all(value in frozen_set for value in candidate)


@dataclass(frozen=True)
class Protocol:
    doc: dict[str, Any]
    path: Path | None = None

    # -- basic accessors -----------------------------------------------------
    @property
    def protocol_id(self) -> str:
        return str(self.doc["protocol_id"])

    @property
    def status(self) -> str:
        return str(self.doc["status"])

    @property
    def frozen(self) -> bool:
        return self.status == STATUS_FROZEN

    @property
    def recorded_hash(self) -> str | None:
        value = self.doc.get(HASH_FIELD)
        return str(value) if value else None

    @property
    def content_hash(self) -> str:
        return compute_protocol_hash(self.doc)

    @property
    def root_seed(self) -> int:
        return int(self.doc["seeds"]["root"])

    @property
    def convention(self) -> str:
        return str(self.doc["system"]["distance"]["convention"])

    @property
    def numeric_types(self) -> list[str]:
        return list(self.doc["system"]["numeric_types"])

    @property
    def m_grid(self) -> list[int]:
        return [int(m) for m in self.doc["hnsw"]["M_grid"]]

    @property
    def ef_search_grid(self) -> list[int]:
        return [int(e) for e in self.doc["hnsw"]["ef_search_grid"]]

    @property
    def ef_construction(self) -> int:
        return int(self.doc["hnsw"]["ef_construction"])

    @property
    def top_k_grid(self) -> list[int]:
        return [int(k) for k in self.doc["hnsw"]["top_k"]]

    @property
    def neighbor_selection(self) -> dict[str, Any]:
        return dict(self.doc["hnsw"]["neighbor_selection"])

    def hnsw_params(self, M: int, seed: int):
        """Build parameters straight from the frozen declaration.

        Every build in the project goes through here, so changing how graphs
        are built requires a protocol version bump rather than a code edit.
        """
        from .hnsw.params import HnswParams

        selection = self.neighbor_selection
        return HnswParams(
            M=int(M),
            ef_construction=self.ef_construction,
            seed=int(seed),
            convention=self.convention,  # type: ignore[arg-type]
            extend_candidates=bool(selection["extend_candidates"]),
            keep_pruned_connections=bool(selection["keep_pruned_connections"]),
        )

    @property
    def recall_targets(self) -> list[float]:
        return [float(r) for r in self.doc["recall_targets"]]

    @property
    def dataset_ids(self) -> list[str]:
        return [str(d["id"]) for d in self.doc["datasets"]]

    def dataset(self, dataset_id: str) -> dict[str, Any]:
        for spec in self.doc["datasets"]:
            if spec["id"] == dataset_id:
                return copy.deepcopy(spec)
        raise ProtocolError(f"dataset {dataset_id!r} is not declared in {self.protocol_id}")

    def claim(self, claim_id: str) -> dict[str, Any]:
        for spec in self.doc["claims"]:
            if spec["id"] == claim_id:
                return copy.deepcopy(spec)
        raise ProtocolError(f"claim {claim_id!r} is not declared in {self.protocol_id}")

    @property
    def claim_families(self) -> list[str]:
        return [str(c["family"]) for c in self.doc["claims"]]

    @property
    def profile_names(self) -> list[str]:
        return sorted(self.doc["profiles"])

    # -- profiles ------------------------------------------------------------
    def profile(self, name: str = "full") -> "RunProfile":
        if name not in self.doc["profiles"]:
            raise ProtocolError(
                f"profile {name!r} is not declared; declared profiles: {self.profile_names}"
            )
        spec = copy.deepcopy(self.doc["profiles"][name])
        return RunProfile(
            name=name,
            claim_bearing=bool(spec.get("claim_bearing", False)),
            description=str(spec.get("description", "")),
            dataset_ids=list(spec.get("datasets", self.dataset_ids)),
            numeric_types=list(spec.get("numeric_types", self.numeric_types)),
            m_grid=[int(m) for m in spec.get("M_grid", self.m_grid)],
            ef_search_grid=[int(e) for e in spec.get("ef_search_grid", self.ef_search_grid)],
            top_k_grid=[int(k) for k in spec.get("top_k", self.top_k_grid)],
            build_repeats=int(spec.get("build_repeats", self.doc["seeds"]["build_repeats"])),
            max_n=spec.get("max_n"),
            max_queries=spec.get("max_queries"),
            protocol_hash=self.content_hash,
        )

    # -- integrity -----------------------------------------------------------
    def verify_hash(self) -> None:
        if not self.frozen:
            return
        recorded = self.recorded_hash
        if not recorded:
            raise FreezeViolation(
                f"protocol {self.protocol_id!r} is marked frozen but carries no {HASH_FIELD}"
            )
        actual = self.content_hash
        if recorded != actual:
            raise FreezeViolation(
                f"protocol {self.protocol_id!r} content hash {actual} does not match the frozen "
                f"hash {recorded}; a frozen protocol must not be edited in place. Write a new "
                f"versioned protocol file and record the change in docs/protocol_changelog.md."
            )

    def problems(self) -> list[str]:
        """All schema and consistency problems, as human-readable strings."""
        doc = self.doc
        out: list[str] = []

        for field in REQUIRED_TOP_LEVEL:
            if field not in doc:
                out.append(f"missing top-level section {field!r}")
        if out:
            return out

        # claims: the four families must be stated separately
        families = self.claim_families
        for family in REQUIRED_CLAIM_FAMILIES:
            if family not in families:
                out.append(f"no claim declared for required family {family!r}")
        seen: set[str] = set()
        for spec in doc["claims"]:
            for field in ("id", "family", "statement", "primary_metric", "exit_gate", "phase"):
                if field not in spec:
                    out.append(f"claim {spec.get('id', '<unnamed>')!r} is missing field {field!r}")
            cid = str(spec.get("id"))
            if cid in seen:
                out.append(f"duplicate claim id {cid!r}")
            seen.add(cid)

        # system
        system = doc["system"]
        if system["distance"]["convention"] not in CONVENTIONS:
            out.append(f"distance convention {system['distance']['convention']!r} is not supported")
        for numeric_type in system["numeric_types"]:
            if numeric_type not in NUMERIC_TYPES:
                out.append(f"numeric type {numeric_type!r} is not supported")
        impl = system.get("hnsw_implementation", {})
        for role in ("primary", "cross_check"):
            entry = impl.get(role)
            if not entry or "name" not in entry or "version" not in entry:
                out.append(f"hnsw_implementation.{role} must declare a name and a version")

        # datasets
        if not doc["datasets"]:
            out.append("no datasets declared")
        for spec in doc["datasets"]:
            for field in ("id", "kind", "n", "dim", "n_queries", "numeric_types", "available"):
                if field not in spec:
                    out.append(f"dataset {spec.get('id', '<unnamed>')!r} is missing field {field!r}")
            if spec.get("kind") == "synthetic" and "generator" not in spec:
                out.append(f"synthetic dataset {spec.get('id')!r} must declare a generator")
            if spec.get("kind") == "external" and "loader" not in spec:
                out.append(f"external dataset {spec.get('id')!r} must declare a loader")
            if not _subset(spec.get("numeric_types", []), system["numeric_types"]):
                out.append(f"dataset {spec.get('id')!r} declares an undeclared numeric type")

        # hnsw grids
        hnsw = doc["hnsw"]
        for key in ("M_grid", "ef_search_grid", "top_k"):
            values = hnsw.get(key) or []
            if not values:
                out.append(f"hnsw.{key} must be a non-empty list")
            elif not _is_sorted_unique(values):
                out.append(f"hnsw.{key} must be sorted ascending with no duplicates: {values}")
        if int(hnsw.get("ef_construction", 0)) <= 0:
            out.append("hnsw.ef_construction must be positive")
        if max(hnsw.get("top_k", [1])) > min(hnsw.get("ef_search_grid", [1])):
            out.append("every efSearch value must be >= max(top_k); otherwise recall is capped by ef")
        parity = hnsw.get("parity_tolerance")
        if not parity:
            out.append(
                "hnsw.parity_tolerance must be declared: the reference implementation is only "
                "credible while it tracks the deployed one"
            )
        else:
            for field in ("tolerance", "build_seeds", "queries", "judgement"):
                if field not in parity:
                    out.append(f"hnsw.parity_tolerance must declare {field!r}")
            if int(parity.get("build_seeds", 0)) < 2:
                out.append(
                    "hnsw.parity_tolerance.build_seeds must be >= 2: a single build per side "
                    "cannot separate an implementation gap from build-seed variance"
                )
        selection = hnsw.get("neighbor_selection")
        if not selection:
            out.append(
                "hnsw.neighbor_selection must be declared: the Algorithm 4 flags change graph "
                "degree and therefore how hard the index is to search"
            )
        else:
            for field in ("extend_candidates", "keep_pruned_connections"):
                if field not in selection:
                    out.append(f"hnsw.neighbor_selection must declare {field!r}")

        # seeds and splits
        seeds = doc["seeds"]
        if "root" not in seeds:
            out.append("seeds.root must be declared")
        if int(seeds.get("build_repeats", 0)) < 2:
            out.append("seeds.build_repeats must be >= 2 so counter stability can be measured")
        splits = doc["splits"]
        cal, test = float(splits.get("cal_fraction", 0)), float(splits.get("test_fraction", 0))
        if not (0 < cal < 1) or not (0 < test < 1):
            out.append("splits.cal_fraction and splits.test_fraction must lie in (0, 1)")
        if abs(cal + test - 1.0) > 1e-9:
            out.append(f"splits fractions must sum to 1, got {cal} + {test}")

        # budgets
        budgets = doc["budgets"]
        for key in ("K_grid", "BV_grid", "BF_grid"):
            values = budgets.get(key) or []
            if not values:
                out.append(f"budgets.{key} must be a non-empty list")
            elif not _is_sorted_unique(values):
                out.append(f"budgets.{key} must be sorted ascending with no duplicates: {values}")
            elif min(values) < 1:
                out.append(f"budgets.{key} values must be >= 1")
        if budgets.get("K_grid") and budgets.get("BV_grid") and budgets.get("BF_grid"):
            # A bit set must be reachable under the nesting constraints (2)-(4):
            # at most BV vectors, BF features each, and one flip per bit position.
            widest = max(
                (BIT_WIDTH[nt] for nt in system["numeric_types"] if nt in BIT_WIDTH),
                default=32,
            )
            capacity = max(budgets["BV_grid"]) * max(budgets["BF_grid"]) * widest
            if max(budgets["K_grid"]) > capacity:
                out.append(
                    f"max(K_grid) = {max(budgets['K_grid'])} exceeds the {capacity} bits that "
                    f"max(BV_grid) x max(BF_grid) x {widest}-bit features can hold"
                )

        # bit-flip policy
        policy = doc["bitflip_policy"]
        modes = policy.get("modes") or []
        for required in ("finite_only", "unrestricted_ieee754"):
            if required not in modes:
                out.append(f"bitflip_policy.modes must include {required!r} as a distinct condition")
        for required in ("sign", "exponent", "mantissa"):
            if required not in (policy.get("bit_classes") or []):
                out.append(f"bitflip_policy.bit_classes must include {required!r}")
        finite = policy.get("finite_only") or {}
        for field in ("reject_nan", "reject_inf", "max_abs_coordinate_rule"):
            if field not in finite:
                out.append(f"bitflip_policy.finite_only must declare {field!r}")

        # targets
        targets = doc["targets"]
        rules = targets.get("rules") or []
        rule_ids = [str(r.get("id")) for r in rules]
        for required in ("universal_fixed", "local_reachable"):
            if required not in rule_ids:
                out.append(f"targets.rules must declare the {required!r} rule")
        for rule in rules:
            for field in ("id", "description", "deterministic", "frozen_before_selection", "scope"):
                if field not in rule:
                    out.append(f"target rule {rule.get('id')!r} is missing field {field!r}")
            if not rule.get("deterministic", False):
                out.append(f"target rule {rule.get('id')!r} must be deterministic")
            if not rule.get("frozen_before_selection", False):
                out.append(
                    f"target rule {rule.get('id')!r} must be frozen before bit selection"
                )
        if "eligibility" not in targets:
            out.append("targets.eligibility must be declared (clean-return and reachability rules)")
        if not (targets.get("lambda_grid") or []):
            out.append("targets.lambda_grid must be declared (lambda = 0 must be included)")
        elif 0 not in [float(x) for x in targets["lambda_grid"]]:
            out.append("targets.lambda_grid must include lambda = 0 (pure targeting)")

        # recall targets and comparisons
        for rho in doc["recall_targets"]:
            if not 0 < float(rho) <= 1:
                out.append(f"recall target {rho} must lie in (0, 1]")
        conditions = doc["comparisons"].get("conditions") or []
        for required in REQUIRED_CONDITIONS:
            if required not in conditions:
                out.append(f"comparisons.conditions must include {required!r}")
        if "primary_metric" not in doc["comparisons"]:
            out.append("comparisons.primary_metric must be declared")

        # surrogate parameter grids stay calibration-only
        surrogate = doc["surrogate"]
        for key in ("gamma_grid", "H_grid", "tau_grid"):
            if not (surrogate.get(key) or []):
                out.append(f"surrogate.{key} must be declared")
        if not surrogate.get("calibration_only", False):
            out.append("surrogate.calibration_only must be true: gamma/H/tau are swept on Qcal only")
        if surrogate.get("H_is_ef_search", True) is not False:
            out.append("surrogate.H_is_ef_search must be false: H counts surrogate steps, not ef")

        # profiles must subset the frozen grids
        for name, spec in doc["profiles"].items():
            if not _subset(spec.get("datasets", []), self.dataset_ids):
                out.append(f"profile {name!r} references an undeclared dataset")
            if not _subset(spec.get("M_grid", []), self.m_grid):
                out.append(f"profile {name!r} widens M_grid beyond the frozen grid")
            if not _subset(spec.get("ef_search_grid", []), self.ef_search_grid):
                out.append(f"profile {name!r} widens ef_search_grid beyond the frozen grid")
            if not _subset(spec.get("numeric_types", []), self.numeric_types):
                out.append(f"profile {name!r} widens numeric_types beyond the frozen set")
            if not _subset(spec.get("top_k", []), self.top_k_grid):
                out.append(f"profile {name!r} widens top_k beyond the frozen grid")
        if "full" not in doc["profiles"]:
            out.append("a 'full' profile must be declared")
        elif not doc["profiles"]["full"].get("claim_bearing", False):
            out.append("the 'full' profile must be marked claim_bearing")

        # leakage policy
        leak = doc["leakage_policy"]
        for field in ("selection_phases", "test_unseal_allowed_phases", "seal_test_split"):
            if field not in leak:
                out.append(f"leakage_policy must declare {field!r}")
        if not leak.get("seal_test_split", False):
            out.append("leakage_policy.seal_test_split must be true")
        overlap = set(leak.get("selection_phases", [])) & set(
            leak.get("test_unseal_allowed_phases", [])
        )
        if overlap:
            out.append(
                f"phases {sorted(overlap)} are both selection phases and allowed to open Qtest"
            )
        return out

    def validate(self) -> "Protocol":
        problems = self.problems()
        if problems:
            joined = "\n  - ".join(problems)
            raise ProtocolError(f"protocol {self.protocol_id!r} is invalid:\n  - {joined}")
        return self

    # -- provenance ----------------------------------------------------------
    def provenance(self, profile: str | None = None) -> dict[str, Any]:
        """The stamp every artifact carries so a result can never float free."""
        stamp = {
            "protocol_id": self.protocol_id,
            "protocol_hash": self.content_hash,
            "protocol_status": self.status,
            "frozen_at": self.doc.get("frozen_at"),
            "root_seed": self.root_seed,
        }
        if profile is not None:
            stamp["profile"] = profile
            stamp["claim_bearing"] = self.profile(profile).claim_bearing
        return stamp


@dataclass(frozen=True)
class RunProfile:
    """A declared subset of the frozen grids, used to keep runs affordable."""

    name: str
    claim_bearing: bool
    description: str
    dataset_ids: list[str]
    numeric_types: list[str]
    m_grid: list[int]
    ef_search_grid: list[int]
    top_k_grid: list[int]
    build_repeats: int
    max_n: int | None
    max_queries: int | None
    protocol_hash: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "claim_bearing": self.claim_bearing,
            "datasets": list(self.dataset_ids),
            "numeric_types": list(self.numeric_types),
            "M_grid": list(self.m_grid),
            "ef_search_grid": list(self.ef_search_grid),
            "top_k": list(self.top_k_grid),
            "build_repeats": self.build_repeats,
            "max_n": self.max_n,
            "max_queries": self.max_queries,
            "protocol_hash": self.protocol_hash,
        }


def load_protocol(
    path: str | Path | None = None,
    *,
    require_frozen: bool = True,
    validate: bool = True,
) -> Protocol:
    path = Path(path) if path is not None else DEFAULT_PROTOCOL
    if not path.exists():
        raise ProtocolError(f"protocol file not found: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    protocol = Protocol(doc=doc, path=path)
    protocol.verify_hash()
    if require_frozen and not protocol.frozen:
        raise FreezeViolation(
            f"protocol {protocol.protocol_id!r} has status {protocol.status!r}; Phase 0 requires a "
            f"frozen protocol before any Phase >= 1 run. Run `python -m braid protocol freeze`."
        )
    if validate:
        protocol.validate()
    return protocol


def freeze_protocol(path: str | Path, *, force: bool = False) -> Protocol:
    """Stamp a draft protocol as frozen and write its content hash."""
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    draft = Protocol(doc=doc, path=path)
    draft.validate()
    if doc.get("status") == STATUS_FROZEN and not force:
        existing = Protocol(doc=doc, path=path)
        existing.verify_hash()
        return existing
    doc["status"] = STATUS_FROZEN
    doc["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc[HASH_FIELD] = compute_protocol_hash(doc)
    ordered = {key: doc[key] for key in sorted(doc)}
    path.write_text(json.dumps(ordered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return load_protocol(path)
