# Complete Top-$k$ Replacement

## The objective

Query $q$, score $s(q,\cdot)$. Clean retrieval set $T_k^0$ and outsider set $O_k = V\setminus T_k^0$, both fixed at clean time. $\tilde x_u$ is the representation after perturbation $\delta$.

$$\boxed{\;J^{@k}_{\mathrm{geo,full}}(\delta) \;=\; s^O_{(k)}(\delta)\;-\;\max_{u\in T_k^0} s(q,\tilde x_u)\;}$$

where $s^O_{(k)}(\delta)$ is the $k$-th highest attacked score among outsiders.

$J>0$ means $k$ outsiders each strictly outscore every member of $T_k^0$, so the attacked top-$k$ is drawn entirely from $O_k$:

$$T_k^\delta \cap T_k^0 = \varnothing, \qquad \mathrm{Recall}@k = 0.$$

The comparison is against $\max_{u\in T_k^0}$ rather than any weaker member because all $k$ originals must be evicted, so even the strongest one must be outranked.

## One correction: this is not an iff

Complete replacement implies $s^O_{(k)} \ge \max_u s(q,\tilde x_u)$, not $>$. The correct statement is

$$J^{@k}_{\mathrm{geo,full}} > 0 \;\Longrightarrow\; \mathrm{Recall}@k = 0 \;\Longrightarrow\; J^{@k}_{\mathrm{geo,full}} \ge 0 .$$

On continuous embeddings the gap is measure zero. Under bit flips it is not: flipped codes give lattice-valued scores, exact ties between an outsider and an original occur with positive probability, and a tie is resolved by the index's internal ID ordering, which is not part of $\delta$. Certify attacks under the strict condition and log the tie rate.

## Graded version

$J^{@k}_{\mathrm{geo,full}}$ is a single bit and carries no information while negative. Letting $s^T_{(j)}$ be the $j$-th highest attacked score among the originals,

$$J_{(j)}(\delta) = s^O_{(j)}(\delta) - s^T_{(k-j+1)}(\delta), \qquad j = 1,\dots,k$$

gives $r(\delta) = \max\{j : J_{(j)}>0\}$ outsiders entering the top-$k$, so $\mathrm{Recall}@k = 1 - r/k$. $J_{(j)}$ is non-increasing in $j$ and $J_{(k)} = J^{@k}_{\mathrm{geo,full}}$, so complete replacement is the top rung of this ladder.

Check, $k=3$, attacked originals $(10,5,1)$ and outsiders $(7,6,0)$: rungs are $6,\,1,\,-10$, so $r=2$ and $\mathrm{Recall}@3 = 1/3$. The merged top-3 is $\{10,7,6\}$.

## When this objective is unnecessary

If $\delta$ flips bits in the **stored codes**, then $s(q,\tilde x_u)$ depends only on the flips assigned to $u$. The problem is separable across items, coupled only through the global budget $B$, and complete replacement is exactly decidable without any max-min optimization.

Define per-item costs, $c^-_u(\theta)$ = flips needed to push original $u$ below $\theta$, and $c^+_v(\theta)$ = flips needed to lift outsider $v$ above $\theta$. Both are closed-form (sorted prefix sums of per-bit score deltas for binary or scalar quantization; codebook enumeration per subquantizer for PQ). With $P_k(\theta)$ the $k$ cheapest outsiders to lift:

$$B^\star = \min_\theta\Bigl[\sum_{u\in T_k^0} c^-_u(\theta) + \sum_{v\in P_k(\theta)} c^+_v(\theta)\Bigr]$$

Complete replacement is feasible iff $B^\star \le B$, and the minimizing $\theta$ gives the flip assignment. Both curves are piecewise constant with opposite monotonicity, so this is exact by enumerating breakpoints.

If instead $\delta$ perturbs the **query**, one perturbation moves every score, separability is gone, and $J^{@k}_{\mathrm{geo,full}}$ is the right object. Optimize a hinge over the $k\times k$ pairs against a frozen promoter set rather than the raw order statistic, whose gradient touches only one vector at a time.

## Scope

All of the above describes exhaustive search. Under HNSW or IVF, flips also change traversal and centroid assignment, so $J>0$ is neither necessary nor sufficient for the returned set. Treat it as an oracle bound and measure the index separately.

Untargeted replacement is the cheapest complete replacement, since $P_k(\theta)$ is free to pick the cheapest promoters. Fixing $P$ to an adversary-chosen set has the same constraint structure at weakly higher cost, so $B^\star$ is best reported as a lower bound on any complete-replacement attack.
