"""Per-query, transient, UNTARGETED delay at small budgets f=1..32.

Matches the framework's model: each query gets its own f-bit budget on its own
trace, flips applied to a copy, one search, restored. No victim, no v_tar; the
objective is only to make THIS query's search do more work. Reports mean
distance-eval inflation across queries at each f, with baselines.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ann.codec import UInt8Codec
from ann.hnsw import InstrumentedHNSW, exact_topk
from ann.attacks import tail_demotion, random_flips, lure_injection
from ann import config as cfg


def recall(a, e): return len(set(a) & set(e)) / len(e)


def main():
    rng = np.random.default_rng(0)
    N, D, NQ, K, EF, B_F = 8000, 32, 15, cfg.K, cfg.EF_SEARCH, 4
    X = rng.normal(size=(N, D)); Q = rng.normal(size=(NQ, D))
    codec = UInt8Codec(X); C = codec.encode(X); Xq = codec.decode(C)
    print(f"building N={N} M={cfg.M} efC={cfg.EF_CONSTRUCTION} ...", flush=True)
    index = InstrumentedHNSW(M=cfg.M, ef_construction=cfg.EF_CONSTRUCTION, seed=1).build(Xq)
    adj = index.layers[0]

    base = []
    for qi in range(NQ):
        ann0, st0 = index.search(Q[qi], K, EF, Xq)
        base.append((ann0, st0, exact_topk(Q[qi], Xq, K)))

    print(f"k={K} ef={EF}, {NQ} queries, per-query transient, B_F={B_F}\n")
    print(f"{'f':>4} {'lure':>7} {'tail':>7} {'rand':>7}  {'lure_rec':>8} {'lure_ex':>7}")
    for f in (1, 2, 3, 4, 5, 8, 16, 32):
        L, T, R, rec, ex = [], [], [], [], []
        for qi in range(NQ):
            q = Q[qi]; ann0, st0, ex0 = base[qi]
            fl = lure_injection(codec, C, q, st0, adj, Xq, K, EF, f, B_F)
            Xa = codec.decode(codec.flip(C, fl))
            annA, stA = index.search(q, K, EF, Xa)
            L.append(stA.n_dist / st0.n_dist - 1)
            rec.append(recall(annA, ex0)); ex.append(exact_topk(q, Xa, K) == ex0)
            ft = tail_demotion(codec, C, q, st0.result, Xq, K, f, B_F)
            T.append(index.search(q, K, EF, codec.decode(codec.flip(C, ft)))[1].n_dist / st0.n_dist - 1)
            frd = random_flips(codec, C, q, st0.result, Xq, K, f, B_F, rng)
            R.append(index.search(q, K, EF, codec.decode(codec.flip(C, frd)))[1].n_dist / st0.n_dist - 1)
        print(f"{f:>4} {np.mean(L)*100:>6.1f}% {np.mean(T)*100:>6.1f}% {np.mean(R)*100:>6.1f}%  "
              f"{np.mean(rec):>8.3f} {np.mean(ex)*100:>6.0f}%", flush=True)


if __name__ == "__main__":
    main()
