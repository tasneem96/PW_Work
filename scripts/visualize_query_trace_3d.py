"""Show one query's path through the HNSW graph, in 3D (pyvista).

Layer index is the z axis. The greedy descent walks the sparse upper layers
down to layer 0, where the beam search expands nodes until it settles on the
top-k.

    python scripts/visualize_query_trace_3d.py --query-id 0 -k 10
    python scripts/visualize_query_trace_3d.py --query-id 0 -o trace.png
"""

import argparse
import sys
from pathlib import Path

import faiss
import numpy as np
import pyvista as pv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from glove_retrieval.hnsw_trace import search_with_trace, verify_against_faiss  # noqa: E402

LAYER_GAP = 1.5
COLOR_PATH = "#d62728"      # greedy descent
COLOR_EXPANDED = "#4c78a8"  # nodes popped at layer 0
COLOR_TOPK = "#e8a33d"      # the answer
COLOR_QUERY = "#2ca25f"


def project(index, node_ids, query):
    """PCA of the involved vectors; the query rides the same components."""
    vectors = np.stack([index.reconstruct(int(i)) for i in node_ids]).astype(np.float64)
    mean = vectors.mean(0)
    _, _, vt = np.linalg.svd(vectors - mean, full_matrices=False)
    axes = vt[:2].T
    xy = (vectors - mean) @ axes
    scale = np.abs(xy).max() or 1.0
    pos = {int(i): p / scale for i, p in zip(node_ids, xy)}
    q_xy = ((np.asarray(query, dtype=np.float64).reshape(-1) - mean) @ axes) / scale
    return pos, q_xy


def plot_trace(index, query, k=10, ef_search=64, max_expanded=60, out=None):
    D, I, trace = search_with_trace(index, query, k=k, ef_search=ef_search)
    top_k = [int(i) for i in I[0] if i >= 0]

    expanded = trace.expanded_nodes[:max_expanded]
    path_nodes = [n for L in range(trace.max_level, -1, -1) for n in trace.path_at_level(L)]
    drawn = sorted(set(expanded) | set(top_k) | set(path_nodes))
    pos, q_xy = project(index, drawn, query)
    drawn_set = set(drawn)

    plotter = pv.Plotter(off_screen=out is not None, window_size=(1300, 950))
    plotter.set_background("white")

    def xyz(node, level):
        return np.array([*pos[int(node)], level * LAYER_GAP])

    # --- layer planes -------------------------------------------------
    for level in range(trace.max_level + 1):
        plotter.add_mesh(
            pv.Plane(center=(0, 0, level * LAYER_GAP), direction=(0, 0, 1),
                     i_size=2.8, j_size=2.8),
            color="#888888", opacity=0.05,
        )
        plotter.add_point_labels(
            np.array([[1.5, 1.5, level * LAYER_GAP]]), [f"layer {level}"],
            font_size=12, text_color="#666666", shape=None, always_visible=True,
        )

    # --- greedy descent through the upper layers ----------------------
    walk = []
    for level in range(trace.max_level, -1, -1):
        seq = trace.path_at_level(level)
        walk.extend(xyz(n, level) for n in seq)
        if level > 0 and seq:
            walk.append(xyz(seq[-1], level - 1))   # drop to the layer below
    for a, b in zip(walk, walk[1:]):
        if not np.allclose(a, b):
            plotter.add_mesh(pv.Tube(pointa=a, pointb=b, radius=0.012), color=COLOR_PATH)
    if walk:
        plotter.add_points(np.array(walk), color=COLOR_PATH, point_size=17,
                           render_points_as_spheres=True)
        plotter.add_point_labels(
            np.array([walk[0]]), [f"entry {trace.entry_point}"],
            font_size=12, text_color=COLOR_PATH, shape=None, always_visible=True,
        )

    # --- layer 0: what the beam search touched ------------------------
    edges = []
    for expansion in trace.expansions[:max_expanded]:
        for neighbor in expansion.discovered:
            if neighbor in drawn_set:
                edges.append((expansion.node, neighbor))
    if edges:
        members = sorted({n for e in edges for n in e})
        offset = {n: i for i, n in enumerate(members)}
        mesh = pv.PolyData(np.array([xyz(n, 0) for n in members]))
        mesh.lines = np.hstack([[2, offset[a], offset[b]] for a, b in edges])
        plotter.add_mesh(mesh, color=COLOR_EXPANDED, opacity=0.35, line_width=1)

    others = [n for n in expanded if n not in top_k]
    if others:
        plotter.add_points(np.array([xyz(n, 0) for n in others]), color=COLOR_EXPANDED,
                           point_size=10, render_points_as_spheres=True)

    top_points = np.array([xyz(n, 0) for n in top_k])
    plotter.add_points(top_points, color=COLOR_TOPK, point_size=22,
                       render_points_as_spheres=True)
    plotter.add_point_labels(
        top_points, [f"{n}  ({d:.3f})" for n, d in zip(top_k, D[0])],
        font_size=10, text_color="#7a5312", shape=None, always_visible=True,
    )

    # --- the query itself ---------------------------------------------
    query_point = np.array([[*q_xy, -0.55 * LAYER_GAP]])
    plotter.add_points(query_point, color=COLOR_QUERY, point_size=26,
                       render_points_as_spheres=True)
    plotter.add_point_labels(query_point, ["query"], font_size=13,
                             text_color=COLOR_QUERY, shape=None, always_visible=True)
    for node in top_k[:3]:
        plotter.add_mesh(pv.Line(query_point[0], xyz(node, 0)), color=COLOR_QUERY,
                         line_width=2, opacity=0.4)

    plotter.add_text(
        f"entry {trace.entry_point} -> {len(trace.hops)} greedy hops -> "
        f"{len(trace.expansions)} expansions at layer 0 "
        f"({trace.ndis} distances, efSearch={trace.ef_search}, k={k})",
        font_size=10, color="#222222",
    )
    plotter.camera_position = [(5.0, -5.4, 3.0 + trace.max_level * LAYER_GAP),
                               (0, 0, trace.max_level * LAYER_GAP / 2.5), (0, 0, 1)]
    # keep the view direction but fit everything -- the entry point sits on the
    # top layer and is easily clipped when max_level is large
    plotter.reset_camera()

    if out:
        plotter.screenshot(out)
        plotter.close()
        print("saved", out)
    else:
        plotter.show()
    return D, I, trace


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default="data/glove-25-hnsw.faiss")
    parser.add_argument("--dataset", default="data/glove-25-angular.hdf5")
    parser.add_argument("--query-id", type=int, default=0, help="row of the file's `test` set")
    parser.add_argument("-k", type=int, default=10)
    parser.add_argument("--ef", type=int, default=64)
    parser.add_argument("--max-expanded", type=int, default=60)
    parser.add_argument("-o", "--out", default=None)
    args = parser.parse_args()

    import h5py

    index = faiss.read_index(args.index)
    with h5py.File(args.dataset, "r") as f:
        query = np.ascontiguousarray(f["test"][args.query_id], dtype=np.float32)
    faiss.normalize_L2(query.reshape(1, -1))   # angular dataset

    check = verify_against_faiss(index, query, k=args.k, ef_search=args.ef)
    print("matches faiss:", {key: check[key] for key in ("ids_match", "distances_match", "ndis_match")})
    D, I, trace = plot_trace(index, query, k=args.k, ef_search=args.ef,
                             max_expanded=args.max_expanded, out=args.out)
    print()
    print(trace.summary())
    print("\ntop-k:")
    for rank, (node, score) in enumerate(zip(I[0], D[0]), start=1):
        print(f"  {rank:>3}. node {node:<8} cosine {score:.4f}")
