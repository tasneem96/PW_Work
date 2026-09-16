"""Validate the threshold work model: does predicted work track real n_dist?

W_pred = sum_{v in R} a_v * deg(v), a_v = sigmoid((s_v - theta)/T),
theta = soft ef-th largest score over R (R = clean visited set + neighbours),
reachability approximated as 1 on R (one-hop region). Checks two things:
 (1) level: corr(W_pred, real n_dist) across queries on the clean store.
 (2) sensitivity: corr(predicted dW, real d n_dist) over single random flips.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import pearsonr, spearmanr
from ann.codec import UInt8Codec, N_BITS
from ann.hnsw import InstrumentedHNSW
from ann import config as cfg


def region(st, adj):
    R = set(st.visited)
    for u in list(R):
        R.update(adj.get(u, ()))
    return np.array(sorted(R))


def predict(q, Xv, R, adj, ef, T):
    s = -((Xv[R] - q) ** 2).sum(axis=1)              # score = -dist^2
    ssort = np.sort(s)[::-1]
    theta = ssort[min(ef, len(ssort)) - 1]           # ef-th largest (hard proxy for soft)
    a = 1.0 / (1.0 + np.exp(-(s - theta) / T))
    deg = np.array([len(adj.get(int(v), ())) for v in R], dtype=float)
    return float((a * deg).sum())


def main():
    rng = np.random.default_rng(0)
    N, D, NQ, K, EF = 8000, 32, 30, cfg.K, cfg.EF_SEARCH
    X = rng.normal(size=(N, D)); Q = rng.normal(size=(NQ, D))
    codec = UInt8Codec(X); C = codec.encode(X); Xq = codec.decode(C)
    print(f"building N={N} M={cfg.M} efC={cfg.EF_CONSTRUCTION} ...", flush=True)
    index = InstrumentedHNSW(M=cfg.M, ef_construction=cfg.EF_CONSTRUCTION, seed=1).build(Xq)
    adj = index.layers[0]
    T = 0.5 * np.median([np.std(-((Xq - Q[i]) ** 2).sum(axis=1)) for i in range(5)])

    # (1) level correlation across queries
    preds, reals, ctx = [], [], []
    for qi in range(NQ):
        q = Q[qi]
        _, st = index.search(q, K, EF, Xq)
        R = region(st, adj)
        preds.append(predict(q, Xq, R, adj, EF, T)); reals.append(st.n_dist)
        ctx.append((q, R, st.n_dist))
    pr = pearsonr(preds, reals)[0]; sr = spearmanr(preds, reals).statistic
    print(f"\n(1) level: pred vs real n_dist over {NQ} queries")
    print(f"    pearson {pr:.3f}  spearman {sr:.3f}")

    # (2) sensitivity: single random flips, predicted dW vs real d n_dist
    dpred, dreal = [], []
    for qi in range(min(NQ, 12)):
        q, R, base_real = ctx[qi]
        base_pred = predict(q, Xq, R, adj, EF, T)
        for _ in range(8):
            v = int(R[rng.integers(len(R))]); j = int(rng.integers(D)); b = int(rng.integers(N_BITS))
            Cw = codec.flip(C, [(v, j, b)]); Xw = codec.decode(Cw)
            dpred.append(predict(q, Xw, R, adj, EF, T) - base_pred)
            dreal.append(index.search(q, K, EF, Xw)[1].n_dist - base_real)
    dp = spearmanr(dpred, dreal).statistic
    nz = np.sum(np.array(dreal) != 0)
    print(f"\n(2) sensitivity: predicted dW vs real d(n_dist), {len(dpred)} single flips")
    print(f"    spearman {dp:.3f}   ({nz} of {len(dpred)} flips changed real work)")


if __name__ == "__main__":
    main()
