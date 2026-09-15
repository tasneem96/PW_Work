"""How much budget does a stored-code delay attack need before it does anything?

premise_test.py showed ~1% work inflation at B=40. This scales the budget to
find the point where the mechanism becomes real, if it ever does.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ann.codec import UInt8Codec
from ann.hnsw import InstrumentedHNSW, exact_topk
from ann.attacks import tail_demotion, random_flips
from ann import config as cfg


def recall(a, e):
    return len(set(a) & set(e)) / len(e)


def main():
    rng = np.random.default_rng(0)
    N, D, NQ, K, B_F = 3000, 32, 12, cfg.K, 4
    X = rng.normal(size=(N, D))
    Q = rng.normal(size=(NQ, D))
    codec = UInt8Codec(X)
    C = codec.encode(X)
    Xq = codec.decode(C)
    index = InstrumentedHNSW(M=cfg.M, ef_construction=cfg.EF_CONSTRUCTION,
                             seed=1).build(Xq)

    print(f"corpus {N}x{D}, k={K}, {B_F} flips/vector max, {NQ} queries")
    print(f"build M={cfg.M} efConstruction={cfg.EF_CONSTRUCTION}, query ef={cfg.EF_SEARCH}")
    for ef in (cfg.EF_SEARCH,):
        print(f"\n--- ef={ef} ---")
        print(f"{'B':>6} {'vecs':>6} {'demote':>9} {'random':>9} "
              f"{'exact ok':>9} {'ann ok':>7} {'rec0':>6} {'recA':>6}")
        for B in (40, 200, 800, 3200):
            dm, rd, ex, an, r0, rA = [], [], [], [], [], []
            for qi in range(NQ):
                q = Q[qi]
                ann0, st0 = index.search(q, K, ef, Xq)
                ex0 = exact_topk(q, Xq, K)
                fd = tail_demotion(codec, C, q, st0.result, Xq, K, B, B_F)
                fr = random_flips(codec, C, q, st0.result, Xq, K, B, B_F, rng)
                XA = codec.decode(codec.flip(C, fd))
                annA, stA = index.search(q, K, ef, XA)
                dm.append(stA.n_dist / st0.n_dist - 1)
                ex.append(exact_topk(q, XA, K) == ex0)
                an.append(set(annA) == set(ann0))
                r0.append(recall(ann0, ex0)); rA.append(recall(annA, ex0))
                XR = codec.decode(codec.flip(C, fr))
                _, stR = index.search(q, K, ef, XR)
                rd.append(stR.n_dist / st0.n_dist - 1)
            print(f"{B:>6} {B//B_F:>6} {np.mean(dm)*100:>8.1f}% {np.mean(rd)*100:>8.1f}% "
                  f"{np.mean(ex)*100:>8.0f}% {np.mean(an)*100:>6.0f}% "
                  f"{np.mean(r0):>6.3f} {np.mean(rA):>6.3f}")


if __name__ == "__main__":
    main()
