# Complete Top-$k$ Replacement: Objective, Exact Condition, and Threat-Model Fork

## 1. Setup

Corpus $V$, query $q$, score $s(q,\cdot)$ (inner product or cosine on the stored representation).

- $\tilde x_u$: the representation of item $u$ *after* the adversarial perturbation $\delta$.
- $T_k^0 = \operatorname{top-}k_{u\in V} s(q,x_u)$, the clean retrieval set. Fixed at clean time.
- $O_k = V \setminus T_k^0$, the outsider set. Also fixed at clean time.
- $T_k^\delta = \operatorname{top-}k_{u\in V} s(q,\tilde x_u)$, the attacked retrieval set.

Sorted attacked scores, with the group membership fixed at clean time:

$$a_{(1)} \ge a_{(2)} \ge \cdots \ge a_{(k)}, \qquad a_{(j)} = j\text{-th largest of } \{s(q,\tilde x_u): u \in T_k^0\}$$

$$b_{(1)} \ge b_{(2)} \ge \cdots, \qquad b_{(j)} = j\text{-th largest of } \{s(q,\tilde x_v): v \in O_k\}$$

In the shorthand of the original note, $a_{(1)} = \max_{u\in T_k^0} s(q,\tilde x_u) = s^T_{(1)}(\delta)$ and $b_{(j)} = s^O_{(j)}(\delta)$.

## 2. The untargeted complete-replacement objective

$$\boxed{\;J^{@k}_{\mathrm{geo,full}}(\delta) \;=\; s^O_{(k)}(\delta)\;-\;\max_{u\in T_k^0} s(q,\tilde x_u)\;=\;b_{(k)}-a_{(1)}\;}$$

**Sufficiency.** If $J^{@k}_{\mathrm{geo,full}}(\delta) > 0$ then $k$ outsiders each strictly outscore every member of $T_k^0$, so $T_k^\delta \subseteq O_k$, hence

$$T_k^\delta \cap T_k^0 = \varnothing, \qquad \mathrm{Recall}@k = 0 .$$

**Necessity (weak form).** Complete replacement implies $b_{(k)} \ge a_{(1)}$, not $>$. The strict inequality is not recoverable in general.

**Where the gap bites.** For continuous embeddings the tie set has measure zero and the distinction is cosmetic. Under a bit-flip threat model it is not. Flipped codes live on a discrete lattice, scores take finitely many values, and exact ties between an outsider and an original are a positive-probability event. On a tie the outcome is decided by the index's tie-break rule (typically internal ID order), which is not part of $\delta$ and not something the attacker controls. So state it as:

$$J^{@k}_{\mathrm{geo,full}}(\delta) > 0 \;\Longrightarrow\; \mathrm{Recall}@k = 0 \;\Longrightarrow\; J^{@k}_{\mathrm{geo,full}}(\delta) \ge 0$$

and report the certified attack under the strict condition. Do not claim an iff.

## 3. The displacement ladder (what $J_{\mathrm{geo,full}}$ is the top rung of)

$J^{@k}_{\mathrm{geo,full}}$ is a single bit: total success or nothing. It carries no information when it is negative, which is most of the time during optimization and during a budget sweep. The graded version:

$$\boxed{\;J_{(j)}(\delta) \;=\; s^O_{(j)}(\delta)\;-\;s^T_{(k-j+1)}(\delta)\;=\;b_{(j)}-a_{(k-j+1)}, \qquad j=1,\dots,k\;}$$

**Claim.** Absent cross-group ties, the number of outsiders that enter the attacked top-$k$ is

$$r(\delta) \;=\; \bigl|T_k^\delta \cap O_k\bigr| \;=\; \max\{\, j \in \{0,1,\dots,k\} : J_{(j)}(\delta) > 0 \,\}$$

with the convention $J_{(0)} \equiv +\infty$, and therefore

$$\mathrm{Recall}@k(\delta) \;=\; 1 - \frac{r(\delta)}{k}.$$

**Why.** $r \ge j$ holds exactly when the $j$-th best outsider outranks the $(k-j+1)$-th best original: $j$ outsiders can only occupy $j$ of the $k$ slots by pushing out the $j$ weakest originals, and the binding comparison is between the weakest of those $j$ outsiders and the strongest original they must displace.

$b_{(j)}$ is non-increasing in $j$ and $a_{(k-j+1)}$ is non-decreasing in $j$, so $J_{(j)}$ is non-increasing in $j$ and $r$ is a clean threshold crossing.

**Consequences.**
- $J_{(k)} = J^{@k}_{\mathrm{geo,full}}$. Complete replacement is the top rung.
- $J_{(1)} = b_{(1)} - a_{(k)} > 0$ is the cheapest nontrivial corruption: one eviction, $\mathrm{Recall}@k = (k-1)/k$.
- Sanity check, $k=3$, attacked originals $a=(10,5,1)$, attacked outsiders $b=(7,6,0)$: $J_{(1)}=6>0$, $J_{(2)}=1>0$, $J_{(3)}=-10<0$, so $r=2$ and $\mathrm{Recall}@3 = 1/3$. Merged top-3 is $\{10,7,6\}$, which confirms it.

Report $(J_{(1)},\dots,J_{(k)})$ or equivalently the budget at which each rung flips sign. That is a curve, not a point, and it is what a reviewer will ask for.

## 4. The fork the objective is hiding: what does $\delta$ perturb?

The formulation is only well-posed once this is fixed, and the two natural answers lead to *different* problems.

### 4a. Flips in the stored codes (index corruption)

$\delta$ assigns a set of bit flips to each stored vector under a global budget $B$. Then $s(q,\tilde x_u)$ depends on $\delta$ **only through the flips assigned to $u$**. The objective is separable across items and coupled only through the budget.

Under separability, the min/max surrogate is unnecessary. Complete replacement is exactly decidable by a one-dimensional sweep.

Define per-item cost curves, with $m$ the number of flips spent on that item:

$$c^-_u(\theta) = \min\{\,m : \exists\,\text{$m$ flips on $u$ with } s(q,\tilde x_u) < \theta\,\}, \qquad u \in T_k^0$$

$$c^+_v(\theta) = \min\{\,m : \exists\,\text{$m$ flips on $v$ with } s(q,\tilde x_v) > \theta\,\}, \qquad v \in O_k$$

Both are computable in closed form. For binary or scalar-quantized codes each bit contributes additively to the inner product, so the optimal $m$ flips are the $m$ largest per-bit deltas of the right sign and the curve is a sorted prefix sum. For product quantization the contribution is not additive within a subquantizer, but each subquantizer's best reachable centroid under a flip budget is found by enumerating its codebook ($M \times 256$ per item, trivial). Additivity is lost; separability across items is not.

Then, letting $P_k(\theta)$ be the $k$ cheapest outsiders to lift above $\theta$:

$$\boxed{\;B^\star \;=\; \min_{\theta}\;\Bigl[\;\sum_{u\in T_k^0} c^-_u(\theta)\;+\;\sum_{v\in P_k(\theta)} c^+_v(\theta)\;\Bigr]\;}$$

Complete replacement is feasible iff $B^\star \le B$, and the minimizing $\theta$ yields the flip assignment directly. Both cost curves are piecewise-constant step functions of $\theta$ ($c^-$ non-increasing, $c^+$ non-decreasing), so the sum has finitely many breakpoints and the minimization is exact by enumeration over them. No relaxation, no gradient, no local optimum.

This is a stronger statement than optimizing $J^{@k}_{\mathrm{geo,full}}$: it returns the exact minimum budget, not a lower bound from whatever the optimizer happened to find.

### 4b. Flips in the query representation (a single shared $\delta$)

Here one perturbation moves $q$ and every score responds. Separability is gone, the problem is genuinely a max-min, and $J^{@k}_{\mathrm{geo,full}}$ is the right object. But then optimize a relaxation of it, because the raw form is a bad signal:

$s^O_{(k)}$ is the $k$-th order statistic over $|O_k|$ items. Its gradient touches exactly one vector, whichever is currently $k$-th. Push that one up and the objective switches to a different vector. The result is plateaus and oscillation.

Fix: freeze a promoter set $P \subset O_k$, $|P| = k$, chosen as the $k$ cheapest outsiders to lift (from the unattacked geometry), then optimize a margin-hinge over the $k \times k$ pairs that must all be satisfied:

$$\mathcal{L}_\tau(\delta) \;=\; \sum_{v\in P}\;\sum_{u\in T_k^0}\;\bigl[\,\tau + s(q_\delta, x_u) - s(q_\delta, x_v)\,\bigr]_+$$

$\mathcal{L}_\tau = 0$ with $\tau > 0$ implies $J^{@k}_{\mathrm{geo,full}} \ge \tau > 0$. Dense gradient, every constraint contributes, and the freeze-and-refresh loop (re-pick $P$ every few steps) recovers most of what the order statistic would have given.

## 5. Caveats to state before reporting numbers

1. **Exact search only.** Everything above characterizes a flat, exhaustive scan. Under HNSW or IVF the returned set is a subset of what the traversal visits, and flips that change codes also change graph traversal and centroid assignment. $J^{@k}_{\mathrm{geo,full}} > 0$ is then neither necessary (traversal can drop an original without any score inversion) nor sufficient (a promoted outsider is useless if it is never visited). Report exact-search results as an oracle bound and measure the ANN index separately.

2. **Untargeted is the floor, not the goal.** Complete replacement by *arbitrary* outsiders is the cheapest complete replacement, because $P_k(\theta)$ is free to pick the cheapest promoters. Fixing $P$ to an adversary-chosen set gives the targeted variant with identical constraint structure and a weakly larger budget. So the right framing for $B^\star$ is a **lower bound on the cost of any complete-replacement attack**, targeted ones included. That is its most defensible use.

3. **$\mathrm{Recall}@k = 0$ is retrieval corruption, not downstream damage.** Under an untargeted objective the promoted items are whatever is cheapest, which usually means semantically unrelated. The generator's likely response is refusal or an off-topic answer, which is a denial-of-service outcome, not a steering outcome. Do not report $\mathrm{Recall}@k = 0$ as evidence of answer manipulation without a separate end-to-end measurement.

4. **Ties.** See §2. With quantized codes, log the fraction of trials where $b_{(k)} = a_{(1)}$ rather than assuming it away.

## 6. Summary of the corrected objects

| Object | Definition | Meaning |
|---|---|---|
| $J_{(j)}$ | $s^O_{(j)} - s^T_{(k-j+1)}$ | $j$-th rung; $r=\max\{j: J_{(j)}>0\}$ |
| $J^{@k}_{\mathrm{geo,full}}$ | $J_{(k)} = s^O_{(k)} - \max_{u\in T_k^0} s(q,\tilde x_u)$ | complete replacement, $\mathrm{Recall}@k=0$ |
| $\mathrm{Recall}@k$ | $1 - r(\delta)/k$ | graded retrieval corruption |
| $B^\star$ | threshold sweep, §4a | exact minimum flip budget (separable model) |
| $\mathcal{L}_\tau$ | $k\times k$ pairwise hinge, §4b | trainable surrogate (coupled model) |
