"""Head-to-head: how to choose which path-pool nodes to flip, at moderate f.

Pool = layer-0 pop order (stats.expanded, dedup), exact top-k removed. Each
node's candidate is its single best demotion bit. Arms differ ONLY in ordering:

  naive_path  : pop order, earliest first (the current baseline)
  cov_path    : by unvisited-neighbour coverage (cheap structure signal, the
                gradient's mechanistic proxy: work-gradient is large where a
                node's expansion reaches unvisited nodes)
  exact_path  : greedy by exact single-flip work inflation on the real searcher

Plus a contrasting pool:
  lure_unvis  : lure injection over UNVISITED nodes, same budget

Per-query, transient, untargeted. Mean distance-eval inflation across queries.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ann.codec import UInt8Codec
from ann.hnsw import InstrumentedHNSW, exact_topk
from ann.attacks import _best_demote_flips, lure_injection
from ann import config as cfg


def recall(a, e): return len(set(a) & set(e)) / len(e)


def dedup(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x); out.append(x)
    return out


def work_of(index, q, K, EF, codec, Cw):
    return index.search(q, K, EF, codec.decode(Cw))[1].n_dist


def main():
    rng = np.random.default_rng(0)
    N, D, NQ, K, EF, POOL = 8000, 32, 8, cfg.K, cfg.EF_SEARCH, 50
    X = rng.normal(size=(N, D)); Q = rng.normal(size=(NQ, D))
    codec = UInt8Codec(X); C = codec.encode(X); Xq = codec.decode(C)
    print(f"building N={N} M={cfg.M} efC={cfg.EF_CONSTRUCTION} ...", flush=True)
    index = InstrumentedHNSW(M=cfg.M, ef_construction=cfg.EF_CONSTRUCTION, seed=1).build(Xq)
    adj = index.layers[0]

    ctx = []
    for qi in range(NQ):
        q = Q[qi]
        ann0, st0 = index.search(q, K, EF, Xq)
        ex0 = exact_topk(q, Xq, K); protk = set(ex0)
        pool = [n for n in dedup(st0.expanded) if n not in protk][:POOL]
        cand = {}
        for n in pool:
            f = _best_demote_flips(codec, C, n, q, 1)
            if f:
                cand[n] = f[0]
        cov = {n: sum(1 for u in adj[n] if u not in set(st0.visited)) for n in pool}
        ctx.append((q, ann0, st0, ex0, protk, pool, cand, cov))

    def measure(qi, flips):
        q, ann0, st0, ex0, *_ = ctx[qi]
        Cw = codec.flip(C, flips)
        annA, stA = index.search(q, K, EF, codec.decode(Cw))
        return stA.n_dist / st0.n_dist - 1, exact_topk(q, Xq, K) == ex0, recall(annA, ex0)

    print(f"k={K} ef={EF}, {NQ} queries, pool<= {POOL}, one bit/vector\n")
    print(f"{'f':>4} {'naive':>7} {'cov':>7} {'exact':>7} {'lure_u':>7}  {'ex_rec':>7} {'ex_ok':>6}")
    for f in (10, 20, 40):
        nai, cov_, exa, lur, exrec, exok = [], [], [], [], [], []
        for qi in range(NQ):
            q, ann0, st0, ex0, protk, pool, cand, cov = ctx[qi]
            order_nodes = [n for n in pool if n in cand]

            nai_fl = [cand[n] for n in order_nodes[:f]]
            nai.append(measure(qi, nai_fl)[0])

            cov_order = sorted(order_nodes, key=lambda n: -cov[n])
            cov_.append(measure(qi, [cand[n] for n in cov_order[:f]])[0])

            # exact greedy, interaction-aware
            chosen, remaining = [], list(order_nodes)
            cur = st0.n_dist
            for _ in range(min(f, len(remaining))):
                best = None
                for n in remaining:
                    w = work_of(index, q, K, EF, codec, codec.flip(C, chosen + [cand[n]]))
                    if best is None or w > best[0]:
                        best = (w, n)
                if best is None or best[0] <= cur:
                    break
                cur = best[0]; chosen.append(cand[best[1]]); remaining.remove(best[1])
            w, ok, rc = measure(qi, chosen)
            exa.append(w); exrec.append(rc); exok.append(ok)

            lfl = lure_injection(codec, C, q, st0, adj, Xq, K, EF, f, 1)
            lur.append(measure(qi, lfl)[0])
        print(f"{f:>4} {np.mean(nai)*100:>6.1f}% {np.mean(cov_)*100:>6.1f}% "
              f"{np.mean(exa)*100:>6.1f}% {np.mean(lur)*100:>6.1f}%  "
              f"{np.mean(exrec):>7.3f} {np.mean(exok)*100:>5.0f}%", flush=True)


if __name__ == "__main__":
    main()
