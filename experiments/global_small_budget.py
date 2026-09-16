"""Persistent (global) flips, fine budget sweep 1..8, measured on the real
searcher across a query set. One flip set shared by all queries, which is the
honest bit-flip threat model: you corrupt memory once, every query sees it.

Greedy: at each step add the single flip that most inflates mean distance-eval
work across the calibration queries, confirmed on the real search, subject to
exact top-k preservation on every query.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ann.codec import UInt8Codec, N_BITS
from ann.hnsw import InstrumentedHNSW, exact_topk
from ann import config as cfg


def recall(a, e): return len(set(a) & set(e)) / len(e)


def candidate_flips(codec, C, index, Qd, Xq, K, EF, adj, per_q=40):
    """Single-flip pool: for each query, cheapest bits that pull a
    high-unvisited-coverage node toward that query's (d_k, d_ef) band."""
    pool = set()
    for q in Qd:
        d = ((Xq - q) ** 2).sum(axis=1)
        order = np.argsort(d)
        protected = set(order[:K].tolist())
        d_k, d_ef = d[order[K - 1]], d[order[min(EF, len(order)) - 1]]
        _, st = index.search(q, K, EF, Xq)
        visited = set(st.visited)
        scored = []
        for v in range(len(Xq)):
            if v in protected:
                continue
            cov = sum(1 for u in adj[v] if u not in visited)
            if cov and d[v] > d_ef:                      # outside band, could be pulled in
                scored.append((cov, v))
        scored.sort(reverse=True)
        for _, v in scored[:per_q]:
            code = C[v].astype(np.int64)
            val = (code - codec.zero) * codec.scale
            best, best_d = None, d[v]
            for j in range(codec.d):
                r = q[j] - val[j]
                for b in range(N_BITS):
                    a = (int(code[j]) >> b) & 1
                    dv = codec.scale[j] * (1 << b) * (1 - 2 * a)
                    nb = int(code[j]) + (1 << b) * (1 - 2 * a)
                    if not 0 <= nb <= 255:
                        continue
                    d_new = d[v] - r * r + (r - dv) ** 2
                    if d_k < d_new < best_d:              # toward q, not into top-k
                        best, best_d = (v, j, b), d_new
            if best:
                pool.add(best)
    return list(pool)


def mean_work_and_recall(codec, Cw, index, Qd, Xq0, K, EF, ex0s, ann0_dist):
    Xw = codec.decode(Cw)
    infl, exok, recs = [], [], []
    for qi, q in enumerate(Qd):
        annA, stA = index.search(q, K, EF, Xw)
        infl.append(stA.n_dist / ann0_dist[qi] - 1)
        exok.append(exact_topk(q, Xw, K) == ex0s[qi])
        recs.append(recall(annA, ex0s[qi]))
    return np.mean(infl), np.mean(exok), np.mean(recs)


def main():
    rng = np.random.default_rng(0)
    N, D, NQ, K, EF = 8000, 32, 12, cfg.K, cfg.EF_SEARCH
    X = rng.normal(size=(N, D)); Q = rng.normal(size=(NQ, D))
    codec = UInt8Codec(X); C = codec.encode(X); Xq = codec.decode(C)
    print(f"building N={N} M={cfg.M} efC={cfg.EF_CONSTRUCTION} ...", flush=True)
    index = InstrumentedHNSW(M=cfg.M, ef_construction=cfg.EF_CONSTRUCTION, seed=1).build(Xq)
    adj = index.layers[0]
    Qd = [Q[i] for i in range(NQ)]

    ex0s, ann0_dist = [], []
    for q in Qd:
        ann0, st0 = index.search(q, K, EF, Xq)
        ex0s.append(exact_topk(q, Xq, K)); ann0_dist.append(st0.n_dist)

    print("building global candidate pool ...", flush=True)
    cands = candidate_flips(codec, C, index, Qd, Xq, K, EF, adj)
    print(f"pool size {len(cands)}\n", flush=True)

    Cw = C.copy()
    chosen = []
    print(f"{'B':>3} {'work+':>8} {'exact_ok':>9} {'recall':>7}  flip")
    for step in range(8):
        best = None
        for f in cands:
            if f in chosen:
                continue
            v, j, b = f
            Cw[v, j] ^= np.uint8(1 << b)
            w, ok, rc = mean_work_and_recall(codec, Cw, index, Qd, Xq, K, EF, ex0s, ann0_dist)
            Cw[v, j] ^= np.uint8(1 << b)
            if ok >= 0.999 and (best is None or w > best[0]):     # keep exact top-k
                best = (w, ok, rc, f)
        if best is None:
            print("  no exact-preserving flip improves work"); break
        w, ok, rc, f = best
        v, j, b = f
        Cw[v, j] ^= np.uint8(1 << b)
        chosen.append(f)
        print(f"{len(chosen):>3} {w*100:>7.1f}% {ok*100:>8.0f}% {rc:>7.3f}  ({v},{j},b{b})", flush=True)


if __name__ == "__main__":
    main()
