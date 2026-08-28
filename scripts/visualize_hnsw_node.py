"""Draw the HNSW neighbourhood of one node, layer by layer (networkx + matplotlib)."""

import faiss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

INDEX_PATH = "data/glove-25-hnsw.faiss"


def node_top_layer(index, node_id):
    return index.hnsw.levels.at(node_id) - 1


def get_node_neighbors(index, node_id, layer, neighbors=None):
    # hnsw.neighbors is a MaybeOwnedVectorInt32 in faiss >= 1.9: .at() raises
    # TypeError and it is not subscriptable, so go through vector_to_array.
    # Pass `neighbors` in to avoid re-copying the whole array on every call.
    hnsw = index.hnsw
    if neighbors is None:
        neighbors = faiss.vector_to_array(hnsw.neighbors)
    begin, end = np.zeros(1, dtype="uint64"), np.zeros(1, dtype="uint64")
    hnsw.neighbor_range(node_id, layer, faiss.swig_ptr(begin), faiss.swig_ptr(end))
    nb = neighbors[int(begin[0]):int(end[0])]
    return nb[nb >= 0].tolist()


def plot_node(index, node_id, max_neighbors=12, out="hnsw_node.png"):
    all_neighbors = faiss.vector_to_array(index.hnsw.neighbors)   # read once
    top = node_top_layer(index, node_id)
    layers = {L: get_node_neighbors(index, node_id, L, all_neighbors)[:max_neighbors]
              for L in range(top + 1)}

    # One shared layout so a node sits in the same spot on every layer.
    G = nx.Graph()
    for L, nbrs in layers.items():
        G.add_edges_from((node_id, n) for n in nbrs)
    pos = nx.spring_layout(G, seed=0)

    xy = np.array(list(pos.values()))
    lo, hi = xy.min(0) - 0.15, xy.max(0) + 0.15

    fig, axes = plt.subplots(top + 1, 1, figsize=(6, 4.5 * (top + 1)), squeeze=False)
    for ax, L in zip(axes[:, 0], reversed(range(top + 1))):   # highest layer on top
        nodes = [node_id] + layers[L]
        sub = nx.Graph()
        sub.add_nodes_from(nodes)
        sub.add_edges_from((node_id, n) for n in layers[L])
        # edges among the neighbours themselves, so it looks like the graph it is
        for n in layers[L]:
            sub.add_edges_from((n, m) for m in get_node_neighbors(index, n, L, all_neighbors)
                               if m in set(nodes))
        colors = ["tab:red" if n == node_id else "tab:blue" for n in sub]
        nx.draw_networkx(sub, pos, ax=ax, node_color=colors, node_size=260,
                         font_size=5, font_color="white", edge_color="0.75")
        ax.set_title(f"layer {L} — {len(layers[L])} neighbours")
        # same window on every panel, so the shared layout keeps nodes in place
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
        ax.axis("off")

    fig.suptitle(f"HNSW node {node_id} (layers 0..{top})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out, dpi=150)
    print("saved", out)


if __name__ == "__main__":
    plot_node(faiss.read_index(INDEX_PATH), 100)
