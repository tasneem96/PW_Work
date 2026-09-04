"""Cross-checks against hnswlib, the deployed implementation.

The instrumented reference implementation is what Phases 3-7 optimize against,
so its behaviour has to be tied to a real system or the whole project measures
an artifact of its own code. Two checks do that here.

*Recall parity.* Both implementations are built over the same corpus with the
same M and ef_construction and evaluated on the same queries across the frozen
efSearch grid. Recall curves are compared against the same exact answers. They
will not be identical (level assignment, tie-breaking, and pruning order all
differ), so parity is a declared tolerance, reported per grid cell, not a
boolean hidden in a log line.

*Graph statistics.* hnswlib's serialized index is parsed to recover its
adjacency, which gives degree distributions and edge counts to compare with the
reference graph.

Two hnswlib properties matter for the threat model and are recorded rather than
worked around:

1. ``space="cosine"`` normalizes stored copies at insert time, so bit flips in a
   deployed hnswlib index would land on normalized data, not on the raw stored
   coordinates this project attacks. We therefore normalize explicitly and use
   the inner-product space, which is numerically the same ranking while keeping
   the raw-storage convention of the protocol.
2. ``BFIndex(space="cosine")`` does not normalize at all (see
   :func:`braid.exact.exact_topk_native`).
"""

from __future__ import annotations

import struct
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .. import similarity as sim
from ..exact import exact_topk
from ..metrics import recall_at_k
from ..vectors import VectorStore
from .params import HnswParams
from .reference import HnswGraph, build_index, search_many


class NativeUnavailable(RuntimeError):
    """hnswlib is not installed in this environment."""


def _require_hnswlib():
    try:
        import hnswlib
    except Exception as exc:  # pragma: no cover - optional dependency
        raise NativeUnavailable(
            "hnswlib is not installed; install the 'native' extra to run parity checks"
        ) from exc
    return hnswlib


def native_version() -> str | None:
    try:
        import hnswlib

        return getattr(hnswlib, "__version__", "unknown")
    except Exception:  # pragma: no cover - optional dependency
        return None


def _native_space(convention: str) -> str:
    return "ip" if convention == "cosine" else "l2"


def _native_items(store: VectorStore, convention: str) -> np.ndarray:
    if convention == "cosine":
        return sim.normalize(store.as_f32())
    return store.as_f32()


def build_native(store: VectorStore, params: HnswParams):
    """Build an hnswlib index with the same structural parameters."""
    hnswlib = _require_hnswlib()
    index = hnswlib.Index(space=_native_space(params.convention), dim=store.dim)
    index.init_index(
        max_elements=store.n,
        M=int(params.M),
        ef_construction=int(params.ef_construction),
        random_seed=int(params.seed) + 1,
    )
    index.add_items(_native_items(store, params.convention), np.arange(store.n), num_threads=1)
    return index


def native_search(
    index, store: VectorStore, queries: np.ndarray, *, k: int, ef_search: int, convention: str
) -> tuple[np.ndarray, np.ndarray]:
    q = sim.as_f32(queries)
    if q.ndim == 1:
        q = q[None, :]
    if convention == "cosine":
        q = sim.normalize(q)
    index.set_ef(max(int(ef_search), int(k)))
    labels, distances = index.knn_query(q, k=int(k), num_threads=1)
    return np.asarray(labels, dtype=np.int64), np.asarray(distances, dtype=np.float32)


# --------------------------------------------------------------------------
# serialized-index parsing
# --------------------------------------------------------------------------

@dataclass
class NativeGraph:
    """Adjacency recovered from an hnswlib index file."""

    entry_point: int
    max_level: int
    M: int
    max_M0: int
    ef_construction: int
    element_levels: np.ndarray
    links: list[dict[int, list[int]]]

    def edge_count(self, layer: int | None = None) -> int:
        layers = range(len(self.links)) if layer is None else [int(layer)]
        return sum(len(v) for lc in layers for v in self.links[lc].values())

    def stats(self) -> dict[str, Any]:
        per_layer = []
        for layer, table in enumerate(self.links):
            degrees = [len(v) for v in table.values()] or [0]
            per_layer.append(
                {
                    "layer": layer,
                    "nodes": len(table),
                    "edges": int(sum(degrees)),
                    "mean_degree": float(np.mean(degrees)),
                    "max_degree": int(np.max(degrees)),
                }
            )
        return {
            "entry_point": self.entry_point,
            "max_level": self.max_level,
            "layers": len(self.links),
            "edges": self.edge_count(),
            "per_layer": per_layer,
        }


def parse_native_index(path: str | Path, dim: int) -> NativeGraph:
    """Parse hnswlib's ``save_index`` output into an adjacency structure.

    The layout follows ``HierarchicalNSW::saveIndex`` in hnswlib 0.8.0: a
    fixed header, then the layer-0 block (link list, vector, label per
    element), then one variable-length upper-layer link block per element.
    Offsets are read from the header rather than assumed.
    """
    raw = Path(path).read_bytes()
    cursor = 0

    def take(fmt: str):
        nonlocal cursor
        size = struct.calcsize(fmt)
        values = struct.unpack_from(fmt, raw, cursor)
        cursor += size
        return values[0] if len(values) == 1 else values

    offset_level0 = take("<Q")
    max_elements = take("<Q")
    cur_element_count = take("<Q")
    size_data_per_element = take("<Q")
    label_offset = take("<Q")
    offset_data = take("<Q")
    max_level = take("<i")
    entry_point = take("<I")
    max_m = take("<Q")
    max_m0 = take("<Q")
    m = take("<Q")
    _mult = take("<d")
    ef_construction = take("<Q")
    del offset_level0, max_elements, label_offset

    expected_data = int(size_data_per_element) - int(offset_data)
    if expected_data < dim * 4:
        raise ValueError(
            f"hnswlib index element size {size_data_per_element} is too small for dim {dim}"
        )

    level0_start = cursor
    level0 = np.frombuffer(
        raw, dtype=np.uint8, count=int(cur_element_count) * int(size_data_per_element), offset=level0_start
    ).reshape(int(cur_element_count), int(size_data_per_element))
    cursor = level0_start + int(cur_element_count) * int(size_data_per_element)

    links: list[dict[int, list[int]]] = [{} for _ in range(int(max_level) + 1)]
    for node in range(int(cur_element_count)):
        count = int(np.frombuffer(level0[node, :4].tobytes(), dtype=np.uint32)[0])
        neighbors = np.frombuffer(
            level0[node, 4 : 4 + 4 * int(max_m0)].tobytes(), dtype=np.uint32, count=int(max_m0)
        )
        links[0][node] = [int(v) for v in neighbors[:count]]

    size_links_per_element = int(max_m) * 4 + 4
    element_levels = np.zeros(int(cur_element_count), dtype=np.int64)
    for node in range(int(cur_element_count)):
        (link_list_size,) = struct.unpack_from("<I", raw, cursor)
        cursor += 4
        if link_list_size == 0:
            continue
        levels = int(link_list_size) // size_links_per_element
        element_levels[node] = levels
        block = raw[cursor : cursor + int(link_list_size)]
        cursor += int(link_list_size)
        for level in range(levels):
            base = level * size_links_per_element
            count = int(struct.unpack_from("<I", block, base)[0])
            neighbors = np.frombuffer(block, dtype=np.uint32, count=int(max_m), offset=base + 4)
            links[level + 1][node] = [int(v) for v in neighbors[:count]]

    return NativeGraph(
        entry_point=int(entry_point),
        max_level=int(max_level),
        M=int(m),
        max_M0=int(max_m0),
        ef_construction=int(ef_construction),
        element_levels=element_levels,
        links=links,
    )


def native_graph_of(index, dim: int) -> NativeGraph:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.bin"
        index.save_index(str(path))
        return parse_native_index(path, dim)


# --------------------------------------------------------------------------
# parity report
# --------------------------------------------------------------------------

def parity_report(
    store: VectorStore,
    queries: np.ndarray,
    params: HnswParams,
    *,
    ef_grid: Sequence[int],
    k: int = 10,
    recall_tolerance: float = 0.05,
    build_seeds: int = 3,
    reference_graph: HnswGraph | None = None,
) -> dict[str, Any]:
    """Compare reference-implementation recall with hnswlib's, per efSearch.

    Parity is judged on the gap between seed-averaged recalls, and the per-seed
    spread is reported next to it. A single build per implementation is not
    enough to judge parity: on the isotropic 128-d corpus at M = 8, four build
    seeds moved reference recall@10 at efSearch 10 across 0.577 to 0.709 and
    hnswlib's across 0.577 to 0.667, so a one-seed comparison can show a 0.05
    "divergence" that is only which graph each side happened to build. Level
    assignment and insertion-time ordering differ between the two
    implementations by construction, so their graphs are never the same
    instance and only their distributions are comparable.
    """
    if native_version() is None:
        return {"available": False, "reason": "hnswlib not installed"}

    exact = exact_topk(queries, store, k, params.convention)
    truth = [set(int(i) for i in row) for row in exact.ids]
    seeds = [int(params.seed) + offset for offset in range(max(1, int(build_seeds)))]

    per_seed: list[dict[str, Any]] = []
    reference_recall: dict[int, list[float]] = {int(ef): [] for ef in ef_grid}
    native_recall: dict[int, list[float]] = {int(ef): [] for ef in ef_grid}
    overlap: dict[int, list[float]] = {int(ef): [] for ef in ef_grid}
    native_stats = None
    parse_error = None

    for index, seed in enumerate(seeds):
        seed_params = replace(params, seed=seed)
        graph = (
            reference_graph
            if index == 0 and reference_graph is not None
            else build_index(store, seed_params)
        )
        native = build_native(store, seed_params)
        if index == 0:
            try:
                native_stats = native_graph_of(native, store.dim).stats()
            except Exception as exc:  # pragma: no cover - format drift is possible
                parse_error = f"{type(exc).__name__}: {exc}"

        cells = []
        for ef in ef_grid:
            ref_ids, _, _ = search_many(graph, store, queries, k=k, ef_search=int(ef))
            nat_ids, _ = native_search(
                native, store, queries, k=k, ef_search=int(ef), convention=params.convention
            )
            ref = recall_at_k(truth, ref_ids, k)
            nat = recall_at_k(truth, nat_ids, k)
            shared = float(
                np.mean([
                    len(set(ref_ids[i].tolist()) & set(nat_ids[i].tolist())) / float(k)
                    for i in range(ref_ids.shape[0])
                ])
            )
            reference_recall[int(ef)].append(ref)
            native_recall[int(ef)].append(nat)
            overlap[int(ef)].append(shared)
            cells.append(
                {
                    "ef_search": int(ef),
                    "reference_recall": ref,
                    "native_recall": nat,
                    "recall_gap": abs(ref - nat),
                    "mean_result_overlap": shared,
                }
            )
        per_seed.append(
            {
                "seed": seed,
                "reference_layer0_mean_degree": graph.stats()["per_layer"][0]["mean_degree"],
                "cells": cells,
            }
        )

    aggregated = []
    worst_gap = 0.0
    for ef in ef_grid:
        ref_values = np.asarray(reference_recall[int(ef)], dtype=np.float64)
        nat_values = np.asarray(native_recall[int(ef)], dtype=np.float64)
        gap = float(abs(ref_values.mean() - nat_values.mean()))
        worst_gap = max(worst_gap, gap)
        aggregated.append(
            {
                "ef_search": int(ef),
                "reference_recall_mean": float(ref_values.mean()),
                "reference_recall_range": [float(ref_values.min()), float(ref_values.max())],
                "native_recall_mean": float(nat_values.mean()),
                "native_recall_range": [float(nat_values.min()), float(nat_values.max())],
                "mean_recall_gap": gap,
                "seed_spread_reference": float(ref_values.max() - ref_values.min()),
                "seed_spread_native": float(nat_values.max() - nat_values.min()),
                "mean_result_overlap": float(np.mean(overlap[int(ef)])),
                "within_tolerance": bool(gap <= recall_tolerance),
                "gap_below_seed_spread": bool(
                    gap <= max(float(ref_values.max() - ref_values.min()),
                               float(nat_values.max() - nat_values.min()))
                ),
            }
        )

    reference_stats = (
        reference_graph.stats() if reference_graph is not None else build_index(store, params).stats()
    )
    return {
        "available": True,
        "hnswlib_version": native_version(),
        "params": params.as_dict(),
        "k": int(k),
        "n_queries": int(np.atleast_2d(queries).shape[0]),
        "build_seeds": seeds,
        "recall_tolerance": float(recall_tolerance),
        "worst_mean_recall_gap": float(worst_gap),
        "passed": bool(worst_gap <= recall_tolerance),
        "cells": aggregated,
        "per_seed": per_seed,
        "reference_graph_stats": reference_stats,
        "native_graph_stats": native_stats,
        "native_graph_parse_error": parse_error,
    }
