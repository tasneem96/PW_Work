"""
Exact minimum bit-flip budget for complete top-k replacement.

Threat model: the adversary flips bits in the STORED document codes only.
The query is untouched. Therefore s(q, x_u) depends only on the flips spent
on item u, the problem is separable across items, and the minimum budget is
obtained exactly by a one-dimensional sweep over a score threshold. No
surrogate objective, no optimizer, no local minima.

Codes are binary (+1/-1 per dimension), score is the inner product.
"""

import numpy as np
from itertools import combinations

INF = float("inf")


def flip_order(q, x):
    """Bit indices ordered by how much flipping them helps, best first.

    Flipping dimension j maps x_j -> -x_j, changing the score by -2*q_j*x_j.
    So bits with q_j*x_j < 0 raise the score by 2|q_j|, and bits with
    q_j*x_j > 0 lower it by 2|q_j|. Within each group, larger |q_j| first.
    """
    prod = q * x
    up = np.nonzero(prod < 0)[0]
    dn = np.nonzero(prod > 0)[0]
    up = up[np.argsort(-np.abs(q[up]))]
    dn = dn[np.argsort(-np.abs(q[dn]))]
    return up, dn


def curve(s0, q, idx, sign, Bmax):
    """Best score reachable with exactly m flips, m = 0..Bmax.

    Monotone, and saturates once the useful bits run out (the adversary
    simply declines to spend the remaining budget on this item).
    """
    g = np.empty(Bmax + 1)
    acc = s0
    for m in range(Bmax + 1):
        if 0 < m <= len(idx):
            acc += sign * 2.0 * abs(q[idx[m - 1]])
        g[m] = acc
    return g


def cost_above(g_up, theta):
    """Fewest flips to push this item's score strictly above theta."""
    hit = np.nonzero(g_up > theta)[0]
    return int(hit[0]) if hit.size else INF


def cost_below(g_dn, theta):
    """Fewest flips to push this item's score strictly below theta."""
    hit = np.nonzero(g_dn < theta)[0]
    return int(hit[0]) if hit.size else INF


def min_budget(q, X, k, Bmax):
    """Exact minimum total flips for T_k^delta disjoint from T_k^0.

    Returns (B_star, theta_star, plan) where plan maps item -> flipped dims.
    """
    s = X @ q
    order = np.argsort(-s)
    T = list(order[:k])
    O = list(order[k:])

    up_idx, dn_idx, G_up, G_dn = {}, {}, {}, {}
    for i in range(len(s)):
        u, d = flip_order(q, X[i])
        up_idx[i], dn_idx[i] = u, d
        G_up[i] = curve(s[i], q, u, +1, Bmax)
        G_dn[i] = curve(s[i], q, d, -1, Bmax)

    # The objective is piecewise constant in theta with breakpoints only at
    # achievable scores, so one probe strictly inside each interval suffices.
    vals = np.unique(np.concatenate([G_up[v] for v in O] + [G_dn[u] for u in T]))
    thetas = list((vals[:-1] + vals[1:]) / 2.0) + [vals[0] - 1.0, vals[-1] + 1.0]

    best = (INF, None, None)
    for th in thetas:
        demote = {u: cost_below(G_dn[u], th) for u in T}
        if any(c == INF for c in demote.values()):
            continue
        lift = sorted(((cost_above(G_up[v], th), v) for v in O))[:k]
        if len(lift) < k or lift[-1][0] == INF:
            continue
        total = sum(demote.values()) + sum(c for c, _ in lift)
        if total < best[0]:
            plan = {u: list(dn_idx[u][:m]) for u, m in demote.items()}
            plan.update({v: list(up_idx[v][:m]) for m, v in lift})
            best = (total, th, plan)

    return best


def apply_plan(X, plan):
    Xa = X.copy()
    for i, bits in plan.items():
        for j in bits:
            Xa[i, j] *= -1
    return Xa


def recall_at_k(q, X_clean, X_attacked, k):
    T0 = set(np.argsort(-(X_clean @ q))[:k])
    Td = set(np.argsort(-(X_attacked @ q))[:k])
    return len(T0 & Td) / k


def brute_force_min(q, X, k, cap):
    """Ground truth by exhaustive search over all flip sets of size <= cap."""
    n, d = X.shape
    slots = [(i, j) for i in range(n) for j in range(d)]
    T0 = set(np.argsort(-(X @ q))[:k])
    for size in range(cap + 1):
        for combo in combinations(slots, size):
            Xa = X.copy()
            for i, j in combo:
                Xa[i, j] *= -1
            if not (set(np.argsort(-(Xa @ q))[:k]) & T0):
                return size
    return INF


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    print("=== correctness: sweep vs exhaustive search ===")
    for trial in range(5):
        q = np.round(rng.normal(size=6), 2)
        X = rng.choice([-1.0, 1.0], size=(6, 6))
        exact = brute_force_min(q, X, k=2, cap=4)
        swept, th, plan = min_budget(q, X, k=2, Bmax=4)
        agree = "ok" if swept == exact else "MISMATCH"
        print(f"  trial {trial}: brute force {exact}   sweep {swept}   {agree}")

    print("\n=== larger instance ===")
    d, n, k = 64, 400, 3
    q = rng.normal(size=d)
    X = rng.choice([-1.0, 1.0], size=(n, d))
    B, theta, plan = min_budget(q, X, k, Bmax=40)
    Xa = apply_plan(X, plan)
    s0, sa = X @ q, Xa @ q
    T0 = np.argsort(-s0)[:k]

    print(f"  corpus {n} x {d}, k={k}")
    print(f"  minimum flips B* = {B}, threshold theta* = {theta:.3f}")
    print(f"  items touched: {len([i for i, b in plan.items() if b])}")
    print(f"  clean top-{k}:    {list(np.argsort(-s0)[:k])}")
    print(f"  attacked top-{k}: {list(np.argsort(-sa)[:k])}")
    print(f"  Recall@{k} = {recall_at_k(q, X, Xa, k)}")

    J = np.sort(sa[[i for i in range(n) if i not in set(T0)]])[::-1][k - 1] - sa[T0].max()
    print(f"  J_geo,full = {J:.4f}  (> 0 certifies complete replacement)")
