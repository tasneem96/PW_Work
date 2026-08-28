"""Draw the HNSW neighbourhood of one node as stacked 3D layers (pyvista).

Layer index becomes the z axis, so the graph is shown the way HNSW is usually
drawn: layer 0 at the bottom, the sparse upper layers above it, and the focus
node threaded vertically through every layer it belongs to.

    python scripts/visualize_hnsw_node_3d.py            # interactive window
    python scripts/visualize_hnsw_node_3d.py -o out.png # headless screenshot
"""

import argparse

import faiss
import numpy as np
import pyvista as pv

INDEX_PATH = "data/glove-25-hnsw.faiss"
LAYER_GAP = 1.4          # vertical spacing between layers
PALETTE = ("#2f6fdb", "#1fa07a", "#d98a1f", "#8c5bd8", "#c94f7c")


def node_top_layer(index, node_id):
    return index.hnsw.levels.at(node_id) - 1


def get_node_neighbors(index, node_id, layer, neighbors=None):
    # hnsw.neighbors is a MaybeOwnedVectorInt32 in faiss >= 1.9: .at() raises
    # TypeError and it is not subscriptable, so go through vector_to_array.
    hnsw = index.hnsw
    if neighbors is None:
        neighbors = faiss.vector_to_array(hnsw.neighbors)
    begin, end = np.zeros(1, dtype="uint64"), np.zeros(1, dtype="uint64")
    hnsw.neighbor_range(node_id, layer, faiss.swig_ptr(begin), faiss.swig_ptr(end))
    nb = neighbors[int(begin[0]):int(end[0])]
    return nb[nb >= 0].tolist()


def layout_2d(index, node_ids):
    """Project the real vectors to 2D with PCA, so positions mean something.

    A node keeps the same (x, y) on every layer it appears on, which is what
    makes the vertical thread through the stack readable.
    """
    vectors = np.stack([index.reconstruct(int(i)) for i in node_ids]).astype(np.float64)
    centered = vectors - vectors.mean(0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ vt[:2].T
    scale = np.abs(xy).max() or 1.0
    return {int(i): p for i, p in zip(node_ids, xy / scale)}


def plot_node_3d(index, node_id, max_neighbors=10, out=None, labels=True):
    all_neighbors = faiss.vector_to_array(index.hnsw.neighbors)   # read once
    top = node_top_layer(index, node_id)
    layers = {L: get_node_neighbors(index, node_id, L, all_neighbors)[:max_neighbors]
              for L in range(top + 1)}

    node_ids = sorted({node_id} | {n for nbrs in layers.values() for n in nbrs})
    pos = layout_2d(index, node_ids)

    plotter = pv.Plotter(off_screen=out is not None, window_size=(1200, 900))
    plotter.set_background("white")

    for L in range(top + 1):
        z = L * LAYER_GAP
        color = PALETTE[L % len(PALETTE)]
        members = [node_id] + layers[L]
        member_set = set(members)
        xyz = {n: np.array([*pos[n], z]) for n in members}

        # the layer's plane, as a faint slab to sit the graph on
        plotter.add_mesh(
            pv.Plane(center=(0, 0, z), direction=(0, 0, 1), i_size=2.6, j_size=2.6),
            color=color, opacity=0.06, show_edges=False,
        )

        # edges: focus -> neighbours, plus edges among those neighbours
        edges = {(node_id, n) for n in layers[L]}
        for n in layers[L]:
            edges |= {(n, m) for m in get_node_neighbors(index, n, L, all_neighbors)
                      if m in member_set and m != n}
        if edges:
            index_of = {n: i for i, n in enumerate(members)}
            points = np.array([xyz[n] for n in members])
            lines = np.hstack([[2, index_of[a], index_of[b]] for a, b in edges])
            mesh = pv.PolyData(points)
            mesh.lines = lines
            plotter.add_mesh(mesh, color=color, opacity=0.55, line_width=2)

        neighbour_points = np.array([xyz[n] for n in layers[L]])
        if len(neighbour_points):
            plotter.add_points(neighbour_points, color=color, point_size=16,
                               render_points_as_spheres=True)
            if labels:
                plotter.add_point_labels(
                    neighbour_points, [str(n) for n in layers[L]],
                    font_size=11, text_color="#333333", shape=None, always_visible=True,
                )

        plotter.add_points(xyz[node_id].reshape(1, 3), color="#d62728", point_size=26,
                           render_points_as_spheres=True)
        plotter.add_point_labels(
            np.array([[1.35, 1.35, z]]), [f"layer {L}  ({len(layers[L])} nbrs)"],
            font_size=13, text_color=color, shape=None, always_visible=True,
        )

    # the focus node threaded through every layer
    if top > 0:
        base, apex = np.array([*pos[node_id], 0.0]), np.array([*pos[node_id], top * LAYER_GAP])
        plotter.add_mesh(pv.Line(base, apex), color="#d62728", line_width=3, opacity=0.5)

    plotter.add_text(f"HNSW node {node_id} — layers 0..{top}", font_size=13, color="#222222")
    plotter.camera_position = [(4.2, -4.6, 3.4 + top * LAYER_GAP),
                               (0, 0, top * LAYER_GAP / 2), (0, 0, 1)]

    if out:
        plotter.screenshot(out)
        plotter.close()
        print("saved", out)
    else:
        plotter.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", default=INDEX_PATH)
    parser.add_argument("--node", type=int, default=100)
    parser.add_argument("-n", "--max-neighbors", type=int, default=10)
    parser.add_argument("-o", "--out", default=None, help="save a PNG instead of opening a window")
    args = parser.parse_args()
    plot_node_3d(faiss.read_index(args.index), args.node,
                 max_neighbors=args.max_neighbors, out=args.out)
