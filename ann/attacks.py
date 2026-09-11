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
