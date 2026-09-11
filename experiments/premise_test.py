"""Does a stored-code delay attack inflate HNSW work while the answer holds?

Two effects pull against each other. Degrading a node loosens the stopping
threshold, which adds work. The same degraded node also stops being expanded,
which removes work by cutting off exploration. This measures which wins, and
how that depends on ef/k, before any route-steering machinery is built.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from ann.codec import UInt8Codec
from ann.hnsw import InstrumentedHNSW, exact_topk
from ann.attacks import tail_demotion, random_flips


def recall(approx, exact):
    return len(set(approx) & set(exact)) / len(exact)


def main():
    rng = np.random.default_rng(0)
    N, D, NQ, K = 3000, 32, 25, 10
    B, B_F = 40, 4

    X = rng.normal(size=(N, D))
    Q = rng.normal(size=(NQ, D))

    codec = UInt8Codec(X)
    C = codec.encode(X)
    Xq = codec.decode(C)

    index = InstrumentedHNSW(M=8, ef_construction=64, seed=1).build(Xq)

    print(f"corpus {N}x{D}, k={K}, budget B={B} flips, <={B_F} per vector")
    print(f"{'ef':>5} {'band':>8} {'work0':>8} {'demote':>9} {'random':>9} "
          f"{'exact ok':>9} {'ann ok':>7} {'rec0':>6} {'recA':>6}")

    for ef in (16, 32, 64, 128, 256):
        infl_d, infl_r, ex_ok, ann_ok, r0, rA, bands = [], [], [], [], [], [], []
        for qi in range(NQ):
            q = Q[qi]
            ann0, st0 = index.search(q, K, ef, Xq)
            ex0 = exact_topk(q, Xq, K)

            d = ((Xq - q) ** 2).sum(axis=1)
            order = np.argsort(d)
            bands.append(float(np.sqrt(d[order[min(ef, N) - 1]]) - np.sqrt(d[order[K - 1]])))

            for name, flips in (
                ("d", tail_demotion(codec, C, q, st0.result, Xq, K, B, B_F)),
                ("r", random_flips(codec, C, q, st0.result, Xq, K, B, B_F, rng)),
            ):
                Xa = codec.decode(codec.flip(C, flips))
                annA, stA = index.search(q, K, ef, Xa)
                inf = stA.n_dist / st0.n_dist - 1.0
                if name == "d":
                    infl_d.append(inf)
                    ex_ok.append(exact_topk(q, Xa, K) == ex0)
                    ann_ok.append(set(annA) == set(ann0))
                    r0.append(recall(ann0, ex0))
                    rA.append(recall(annA, ex0))
                else:
                    infl_r.append(inf)

        print(f"{ef:>5} {np.mean(bands):>8.3f} "
              f"{'':>8} {np.mean(infl_d)*100:>8.1f}% {np.mean(infl_r)*100:>8.1f}% "
              f"{np.mean(ex_ok)*100:>8.0f}% {np.mean(ann_ok)*100:>6.0f}% "
              f"{np.mean(r0):>6.3f} {np.mean(rA):>6.3f}")

    print("\nband = distance gap between exact rank k and rank ef "
          "(the interval an attack has to work in)")
    print("demote/random = mean inflation in distance evaluations")


if __name__ == "__main__":
    main()
