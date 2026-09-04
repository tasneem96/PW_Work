"""The distance oracle: the only place a distance is ever computed.

Routing every distance through one object has two consequences that Phase 1
depends on. First, the distance-evaluation counter cannot drift from reality,
because there is no second code path that could compute a distance without
telling the recorder. Second, the stale condition of Section 15 becomes a
one-line change: the graph stays as it is and the oracle is handed a different
vector store, so node distances are computed from corrupted vectors on a graph
that still encodes the clean geometry.
"""

from __future__ import annotations

import numpy as np

from .. import similarity as sim
from ..similarity import Convention
from ..vectors import VectorStore
from .trace import TraceRecorder


class DistanceOracle:
    """Distances from one query (or one stored vector) to a set of node ids."""

    def __init__(self, store: VectorStore, convention: Convention = "cosine") -> None:
        sim.check_convention(convention)
        self.store = store
        self.convention = convention
        self._vectors = store.as_f32()
        if convention == "cosine":
            self._normalized = self._vectors / store.norms()[:, None]
            self._sq_norms = None
        else:
            self._normalized = None
            self._sq_norms = np.einsum("ij,ij->i", self._vectors, self._vectors)

    # -- vector access -------------------------------------------------------
    @property
    def n(self) -> int:
        return self.store.n

    def vector(self, node: int) -> np.ndarray:
        return self._vectors[int(node)]

    def prepare_query(self, query: np.ndarray) -> np.ndarray:
        """Query-side preprocessing, done once per query rather than per hop."""
        q = sim.as_f32(query).reshape(-1)
        if self.convention == "cosine":
            return q / max(float(np.linalg.norm(q)), float(sim.EPS_NORM))
        return q

    def prepared_row(self, node: int) -> np.ndarray:
        """A stored vector in the same preprocessed form as a prepared query.

        Build-time insertion treats the new element as the query, so this
        avoids re-normalizing a row the oracle already normalized once.
        """
        if self.convention == "cosine":
            return self._normalized[int(node)]
        return self._vectors[int(node)]

    def pairwise(self, nodes) -> np.ndarray:
        """Full distance matrix among ``nodes``, in one call.

        The neighbour-selection heuristic (Algorithm 4) asks "is this candidate
        closer to the base element than to any already-kept neighbour?" for
        every candidate. Computing the candidate-by-candidate matrix once turns
        that inner loop into array indexing, which is the difference between a
        build measured in minutes and one measured in seconds. These are
        element-to-element build distances and are never counted as query work.
        """
        ids = np.asarray(nodes, dtype=np.int64)
        if ids.size == 0:
            return np.empty((0, 0), dtype=np.float32)
        if self.convention == "cosine":
            block = self._normalized[ids]
            return (np.float32(1.0) - block @ block.T).astype(np.float32, copy=False)
        block = self._vectors[ids]
        sq = self._sq_norms[ids]
        out = sq[:, None] + sq[None, :] - 2.0 * (block @ block.T)
        return np.maximum(out, 0.0).astype(np.float32, copy=False)

    # -- the counted call ----------------------------------------------------
    def distances(
        self,
        prepared_query: np.ndarray,
        nodes,
        *,
        recorder: TraceRecorder | None = None,
        layer: int = 0,
        context: str = "expansion",
    ) -> np.ndarray:
        """d(q, e_v) for every v in ``nodes``, recorded as evaluations of L(q)."""
        ids = np.asarray(nodes, dtype=np.int64)
        if ids.size == 0:
            return np.empty(0, dtype=np.float32)
        if self.convention == "cosine":
            out = np.float32(1.0) - (self._normalized[ids] @ prepared_query)
        else:
            diff = self._vectors[ids] - prepared_query
            out = np.einsum("ij,ij->i", diff, diff)
        out = out.astype(np.float32, copy=False)
        if recorder is not None and recorder.enabled:
            recorder.note_distance_evals(ids, out, layer=layer, context=context)
        return out

    def uncounted_distances(self, prepared_query: np.ndarray, nodes) -> np.ndarray:
        """Distances that are not part of a query's search work.

        Used by the build-time neighbour-selection heuristic, whose element-to
        -element comparisons are build cost, not query cost. Mixing the two
        would inflate the work counters that Section 16.3 reports.
        """
        return self.distances(prepared_query, nodes, recorder=None)
