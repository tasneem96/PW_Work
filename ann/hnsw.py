"""Minimal HNSW with work instrumentation.

The graph is built once on clean vectors and is never modified by an attack:
the threat model corrupts stored codes, not index topology. search() therefore
takes the vector table to score against as an argument, so the identical graph
can be traversed with clean or attacked values.
"""

import heapq
import numpy as np


class SearchStats:
    __slots__ = ("n_dist", "expanded", "visited", "result")

    def __init__(self):
        self.n_dist = 0
        self.expanded = []
        self.visited = []
        self.result = []


class InstrumentedHNSW:
    def __init__(self, M=8, ef_construction=64, seed=0):
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.rng = np.random.default_rng(seed)
        self.ml = 1.0 / np.log(M)
        self.layers = []          # layers[l][node] -> neighbour list
        self.node_layer = []
        self.entry = None
        self.vectors = None

    # ---- build -------------------------------------------------------
    def _d(self, q, v, stats=None):
        if stats is not None:
            stats.n_dist += 1
        diff = q - v
        return float(diff @ diff)

    def build(self, X):
        self.vectors = X
        for i in range(len(X)):
            l = int(-np.log(self.rng.random()) * self.ml)
            self.node_layer.append(l)
            prev_top = len(self.layers) - 1
            while len(self.layers) <= l:
                self.layers.append({})
            for lv in range(l + 1):
                self.layers[lv][i] = []
            if self.entry is None:
                self.entry = i
                continue
            self._insert(i, l, prev_top)
            if l > self.node_layer[self.entry]:
                self.entry = i
        return self

    def _insert(self, node, l, prev_top):
        ep = self.entry
        for lv in range(prev_top, l, -1):
            ep = self._greedy(self.vectors[node], ep, lv)
        for lv in range(min(l, prev_top), -1, -1):
            W = self._search_layer(self.vectors[node], [ep], self.ef_construction,
                                   lv, self.vectors, SearchStats())
            cand = sorted(((-nd, i) for nd, i in W))
            cap = self.M0 if lv == 0 else self.M
            nbrs = [i for _, i in cand if i != node and i in self.layers[lv]][:cap]
            self.layers[lv][node] = nbrs
            for nb in nbrs:
                self.layers[lv][nb].append(node)
                if len(self.layers[lv][nb]) > cap:
                    self.layers[lv][nb] = sorted(
                        self.layers[lv][nb],
                        key=lambda z: self._d(self.vectors[nb], self.vectors[z]))[:cap]
            if cand:
                ep = cand[0][1]

    def _greedy(self, q, start, layer, vecs=None, stats=None):
        vecs = self.vectors if vecs is None else vecs
        cur, cur_d = start, self._d(q, vecs[start], stats)
        improved = True
        while improved:
            improved = False
            if stats is not None:
                stats.expanded.append(cur)
            for e in self.layers[layer].get(cur, ()):
                de = self._d(q, vecs[e], stats)
                if de < cur_d:
                    cur, cur_d, improved = e, de, True
        return cur

    # ---- search ------------------------------------------------------
    def _search_layer(self, q, eps, ef, layer, vecs, stats):
        visited = set(eps)
        C, W = [], []
        for ep in eps:
            d = self._d(q, vecs[ep], stats)
            heapq.heappush(C, (d, ep))
            heapq.heappush(W, (-d, ep))
        while C:
            d_c, c = heapq.heappop(C)
            if d_c > -W[0][0]:
                break
            stats.expanded.append(c)
            for e in self.layers[layer].get(c, ()):
                if e in visited:
                    continue
                visited.add(e)
                d_e = self._d(q, vecs[e], stats)
                if d_e < -W[0][0] or len(W) < ef:
                    heapq.heappush(C, (d_e, e))
                    heapq.heappush(W, (-d_e, e))
                    if len(W) > ef:
                        heapq.heappop(W)
        stats.visited.extend(visited)
        return W

    def search(self, q, k, ef, vecs=None):
        vecs = self.vectors if vecs is None else vecs
        stats = SearchStats()
        ep = self.entry
        for lv in range(len(self.layers) - 1, 0, -1):
            ep = self._greedy(q, ep, lv, vecs, stats)
        W = self._search_layer(q, [ep], ef, 0, vecs, stats)
        ranked = sorted(((-nd, i) for nd, i in W))
        stats.result = [i for _, i in ranked]
        return stats.result[:k], stats


def exact_topk(q, X, k):
    return list(np.argsort(((X - q) ** 2).sum(axis=1))[:k])
