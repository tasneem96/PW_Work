"""Does changing geometry to lure the search through new regions beat just
loosening the stopping threshold? Distance-eval inflation at matched recall."""
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
    print(f"building N={N} D={D} M={cfg.M} efC={cfg.EF_CONSTRUCTION} ...", flush=True)
    index = InstrumentedHNSW(M=cfg.M, ef_construction=cfg.EF_CONSTRUCTION, seed=1).build(Xq)
    adj = index.layers[0]

    print(f"k={K} ef={EF}, {NQ} queries, {B_F} flips/vec\n")
    print(f"{'B':>6} {'attack':>10} {'work+':>8} {'exact_ok':>9} {'rec0':>6} {'recA':>6} {'vecs':>5}")
    for B in (40, 200, 800):
        rows = {"lure": [], "tail": [], "rand": []}
        exok = {"lure": [], "tail": []}
        rec = {"lure": [], "tail": []}
        nv = {"lure": [], "tail": []}
        for qi in range(NQ):
            q = Xq_q = Q[qi]
            ann0, st0 = index.search(q, K, EF, Xq)
            ex0 = exact_topk(q, Xq, K)
            attacks = {
                "lure": lure_injection(codec, C, q, st0, adj, Xq, K, EF, B, B_F),
                "tail": tail_demotion(codec, C, q, st0.result, Xq, K, B, B_F),
                "rand": random_flips(codec, C, q, st0.result, Xq, K, B, B_F, rng),
            }
            for name, fl in attacks.items():
                Xa = codec.decode(codec.flip(C, fl))
                annA, stA = index.search(q, K, EF, Xa)
                rows[name].append(stA.n_dist / st0.n_dist - 1)
                if name in exok:
                    exok[name].append(exact_topk(q, Xa, K) == ex0)
                    rec[name].append(recall(annA, ex0))
                    nv[name].append(len({i for i, _, _ in fl}))
        for name in ("lure", "tail", "rand"):
            r0 = np.mean([recall(*[index.search(Q[qi], K, EF, Xq)[0], exact_topk(Q[qi], Xq, K)])
                          for qi in range(NQ)]) if name == "lure" else None
            line = f"{B:>6} {name:>10} {np.mean(rows[name])*100:>7.1f}%"
            if name in exok:
                line += (f" {np.mean(exok[name])*100:>8.0f}% "
                         f"{'':>6} {np.mean(rec[name]):>6.3f} {np.mean(nv[name]):>5.0f}")
            print(line, flush=True)
        print()


if __name__ == "__main__":
    main()
