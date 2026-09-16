"""Delay attacks on the stored codes. Index topology is never touched."""

import numpy as np
from .hnsw import exact_topk

N_BITS = 8


def _best_demote_flips(codec, C, node, q, n_flips):
    """Greedily pick flips on one vector that most increase its L2 distance to q.

    Bits within a dimension interact through the squared term, so gains are
    recomputed after each pick. Per-vector only: under stored-code flips there
    is no cross-vector coupling.
    """
    code = C[node].astype(np.int64).copy()
    pw = (1 << np.arange(N_BITS))[None, :]              # (1, 8)
    chosen = []
    blocked = np.zeros((codec.d, N_BITS), dtype=bool)
    for _ in range(n_flips):
        val = (code - codec.zero) * codec.scale
        r = (q - val)[:, None]                           # (d, 1)
        a = (code[:, None] >> np.arange(N_BITS)[None, :]) & 1
        step = pw * (1 - 2 * a)                          # integer code delta
        nb = code[:, None] + step
        dv = codec.scale[:, None] * step                 # value delta
        gain = (r - dv) ** 2 - r ** 2
        gain[blocked] = -np.inf
        gain[(nb < 0) | (nb > 255)] = -np.inf
        idx = int(np.argmax(gain))
        j, b = divmod(idx, N_BITS)
        if not np.isfinite(gain[j, b]) or gain[j, b] <= 0:
            break
        code[j] += int(step[j, b])
        blocked[j, b] = True
        chosen.append((node, j, b))
    return chosen


def tail_demotion(codec, C, q, clean_result, X_clean, k, B, B_F):
    """Demote the non-answer members of W to loosen the stopping threshold.

    Members of the exact top-k are excluded from the editable pool, so the
    exact answer is preserved by construction: demotion only moves a vector
    further from q, and no answer vector is touched.
    """
    protected = set(exact_topk(q, X_clean, k))
    tail = [n for n in clean_result[k:] if n not in protected]
    flips, spent = [], 0
    for node in tail:
        if spent >= B:
            break
        take = min(B_F, B - spent)
        f = _best_demote_flips(codec, C, node, q, take)
        flips.extend(f)
        spent += len(f)
    return flips


def random_flips(codec, C, q, clean_result, X_clean, k, B, B_F, rng):
    """Control: same budget, same editable pool, uniformly random bits."""
    protected = set(exact_topk(q, X_clean, k))
    pool = [n for n in clean_result[k:] if n not in protected]
    if not pool:
        return []
    flips, per = [], {}
    seen = set()
    while len(flips) < B:
        node = pool[rng.integers(len(pool))]
        if per.get(node, 0) >= B_F:
            if all(per.get(n, 0) >= B_F for n in pool):
                break
            continue
        j, b = int(rng.integers(codec.d)), int(rng.integers(N_BITS))
        if (node, j, b) in seen:
            continue
        seen.add((node, j, b))
        per[node] = per.get(node, 0) + 1
        flips.append((node, j, b))
    return flips


def _score_of(codec, C, node, q):
    v = (C[node].astype(np.float64) - codec.zero) * codec.scale
    return -np.sum((q - v) ** 2)


def _cost_to_band(codec, C, node, q, d_k, d_ef, B_F):
    """Cheapest flips to pull `node` into the (d_k, d_ef) distance band.

    Returns (flips, ok). ok=False if the node cannot be placed strictly inside
    the band within B_F flips without crossing d_k (which would break recall).
    Uses smallest-magnitude helpful bits first for fine control near d_k.
    """
    code = C[node].astype(np.int64).copy()
    chosen = []
    for _ in range(B_F):
        val = (code - codec.zero) * codec.scale
        d = np.sum((q - val) ** 2)
        if d_k < d < d_ef:                      # already inside the band
            return chosen, True
        # want to DECREASE distance (move toward q): pick the bit giving the
        # largest safe distance reduction that does not overshoot past d_k
        best, best_d = None, d
        for j in range(codec.d):
            r = q[j] - val[j]
            for b in range(N_BITS):
                a = (int(code[j]) >> b) & 1
                dv = codec.scale[j] * (1 << b) * (1 - 2 * a)
                nb = int(code[j]) + (1 << b) * (1 - 2 * a)
                if not 0 <= nb <= 255:
                    continue
                d_new = d - r * r + (r - dv) ** 2
                if d_k < d_new < best_d:         # closer, still outside top-k
                    best, best_d = (j, b), d_new
        if best is None:
            break
        j, b = best
        code[j] ^= np.uint8(1 << b)
        chosen.append((node, j, b))
    val = (code - codec.zero) * codec.scale
    d = np.sum((q - val) ** 2)
    return chosen, (d_k < d < d_ef)


def lure_injection(codec, C, q, clean_stats, adj, X_clean, k, ef, B, B_F):
    """Inject lures: pull unvisited nodes with many unvisited neighbours into
    the (rank k, rank ef) band so the real search expands them and pays for
    their neighbourhood. Greedy budgeted maximum coverage over new expansions.

    Exact top-k is preserved because no lure is allowed above d_k, and top-k
    vectors are excluded from the editable pool.
    """
    d_all = ((X_clean - q) ** 2).sum(axis=1)
    order = np.argsort(d_all)
    protected = set(order[:k].tolist())
    d_k = d_all[order[k - 1]]
    d_ef = d_all[order[min(ef, len(order)) - 1]]
    visited = set(clean_stats.visited)

    # candidate lures: not in top-k, ranked by unvisited-neighbour coverage
    cand = []
    for v in range(len(X_clean)):
        if v in protected:
            continue
        new_nbrs = [u for u in adj[v] if u not in visited]
        if new_nbrs:
            cand.append((len(new_nbrs), v, set(new_nbrs)))
    cand.sort(reverse=True)

    flips, spent, covered = [], 0, set()
    for _, v, nbrs in cand:
        if spent >= B:
            break
        gain = len(nbrs - covered)
        if gain == 0:
            continue
        take = min(B_F, B - spent)
        f, ok = _cost_to_band(codec, C, v, q, d_k, d_ef, take)
        if ok and f:
            flips.extend(f)
            spent += len(f)
            covered |= nbrs
    return flips
