"""HNSW parameters, fixed by the frozen protocol."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from ..similarity import Convention, check_convention


@dataclass(frozen=True)
class HnswParams:
    """Build-time and structural parameters.

    ``ef_search`` is not here on purpose: it is a query-time setting, and
    Section 5.3 makes the point that the attacker never chooses it. Keeping it
    out of the build parameters makes it impossible to accidentally bake a
    search budget into a graph.

    The neighbour-selection defaults follow hnswlib rather than the paper's
    Algorithm 4 flags. hnswlib's ``getNeighborsByHeuristic2`` does not keep
    pruned connections, so its layer-0 lists can fall below M; keeping them
    (the paper's ``keepPrunedConnections = true``) raises mean layer-0 degree
    from about M to about 1.6M on our corpora and makes the index materially
    easier to search than the deployed one. That is an available ablation, not
    the default, because a robustness result on an easier-to-search index would
    overstate the deployed system's resilience.
    """

    M: int = 16
    ef_construction: int = 200
    max_M0: int | None = None
    level_multiplier: float | None = None
    seed: int = 0
    convention: Convention = "cosine"
    extend_candidates: bool = False
    keep_pruned_connections: bool = False

    def __post_init__(self) -> None:
        check_convention(self.convention)
        if self.M < 2:
            raise ValueError(f"M must be >= 2, got {self.M}")
        if self.ef_construction < 1:
            raise ValueError(f"ef_construction must be >= 1, got {self.ef_construction}")

    @property
    def m0(self) -> int:
        """Layer-0 degree cap; hnswlib's default rule is 2M."""
        return int(self.max_M0) if self.max_M0 is not None else 2 * int(self.M)

    @property
    def mL(self) -> float:
        """Level-assignment multiplier; Malkov & Yashunin recommend 1/ln(M)."""
        if self.level_multiplier is not None:
            return float(self.level_multiplier)
        return 1.0 / math.log(float(self.M))

    def max_degree(self, layer: int) -> int:
        return self.m0 if layer == 0 else int(self.M)

    def as_dict(self) -> dict[str, Any]:
        return {
            "M": int(self.M),
            "ef_construction": int(self.ef_construction),
            "max_M0": self.m0,
            "level_multiplier": self.mL,
            "seed": int(self.seed),
            "convention": self.convention,
            "extend_candidates": bool(self.extend_candidates),
            "keep_pruned_connections": bool(self.keep_pruned_connections),
        }
