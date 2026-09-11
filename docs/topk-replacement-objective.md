# Complete Top-$k$ Replacement under Stored-Code Bit Flips

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

## What this means for the attack: you compute it, you do not optimize it

Because $\delta$ flips bits **only in the stored codes**, $s(q,\tilde x_u)$ depends
only on the flips spent on item $u$. Item $v$'s score does not move when you flip
bits in item $u$. The items are independent and the only thing linking them is the
global budget $B$. That makes complete replacement exactly decidable.

For each item, the best score reachable with $m$ flips is closed form. With binary
codes $x\in\{-1,+1\}^d$ and inner-product scoring, flipping dimension $j$ changes the
score by $-2q_jx_j$, so the useful flips are the $m$ largest $2|q_j|$ of the right
sign and the reachable-score curve is a sorted prefix sum. (Scalar quantization is
the same, with per-bit deltas $q_j\cdot\text{scale}\cdot 2^b$. Product quantization
loses additivity within a subquantizer but stays separable across items: enumerate
the $256$ centroids per subquantizer, then a small knapsack over the $M$ of them.)

From those curves read off, for any threshold $\theta$:

- $c^-_u(\theta)$, the fewest flips to push original $u$ strictly below $\theta$,
- $c^+_v(\theta)$, the fewest flips to push outsider $v$ strictly above $\theta$.

Complete replacement at threshold $\theta$ costs $\sum_{u\in T_k^0}c^-_u(\theta)$ plus
the $k$ smallest $c^+_v(\theta)$, so the minimum budget is

$$\boxed{\;B^\star = \min_\theta\Bigl[\sum_{u\in T_k^0} c^-_u(\theta) + \sum_{v\in P_k(\theta)} c^+_v(\theta)\Bigr]\;}$$

with $P_k(\theta)$ the $k$ cheapest outsiders to lift. Both cost curves are
piecewise constant in $\theta$ with opposite monotonicity, so the minimization is
exact by probing one $\theta$ inside each interval between achievable scores.
The minimizing $\theta$ hands back the flip assignment directly.

The attack is therefore feasible iff $B^\star \le B$, and $B^\star$ is the true
minimum rather than whatever an optimizer happened to reach.

`topk_replacement.py` implements this and checks it against exhaustive search over
all flip sets on small instances. The two agree. On a $400\times 64$ binary corpus
with $k=3$, $B^\star = 4$ flips, and the optimal plan spends all four on demoting
the three originals without promoting anything: the outsiders rise on their own
once the incumbents drop. A max-min formulation would have hidden that structure.

## Scale note

$O_k$ is the whole corpus, so do not build cost curves for every item. The gain
from $B$ flips is bounded by the sum of the $B$ largest $|q_j|$, which is the same
bound for all items, so any outsider whose clean score falls more than that below
$\theta$ can never be lifted and is discarded before any per-item work.

## Scope

All of the above assumes exhaustive search. Under HNSW or IVF, flipped codes also
change graph traversal and centroid assignment, so the returned set is not the
score-ordered top-$k$ and $B^\star$ becomes a lower bound rather than an exact
answer. Report it as an exact result for flat search and measure the ANN index
separately.

Untargeted replacement is the cheapest complete replacement, since $P_k(\theta)$ is
free to pick whichever outsiders are cheapest. Fixing that set to an
adversary-chosen one has identical structure at weakly higher cost, so $B^\star$ is
a lower bound on the cost of any complete-replacement attack, targeted included.
That is its strongest use.
