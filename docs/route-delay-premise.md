# Route-delay premise test

Negative result, recorded before building the route-steering machinery.

## Question

The route-delay spec assumes that steering an HNSW search onto a longer route
inflates search work while the returned answer stays correct. Two effects pull
against each other and the spec assumes the first one wins:

1. Degrading a stored code loosens the stopping threshold (HNSW stops when the
   best remaining candidate is worse than the worst member of $W$), which adds work.
2. The same degraded node stops being expanded, cutting off exploration, which
   removes work.

## Method

Graph built once on clean codes and never modified, since the threat model
corrupts stored codes and not index topology. `search()` takes the vector table
to score against, so the identical graph is traversed with clean or attacked
values. Work is counted as distance evaluations.

Attack: demote the non-answer members of $W$ (ANN ranks $k \ldots \mathrm{ef}$),
excluding anything in the clean exact top-$k$ from the editable pool. The exact
answer is then preserved by construction, since demotion only moves a vector
further from $q$ and no answer vector is touched. Control: the same budget and
the same pool, uniformly random bits.

No trained model, no gradients, no relaxation. Per-vector greedy on closed-form
codec arithmetic, evaluated on the real searcher.

## Result

Corpus 3000 x 32, uint8 scalar quantisation, $k=10$, $M=8$, 12 queries.
Work inflation in distance evaluations:

| ef | B=40 | B=200 | B=800 | B=3200 | random @ B=3200 |
|---|---|---|---|---|---|
| 64  | 2.1% | 7.1% | 7.4% | 7.4%  | 5.4% |
| 256 | 0.7% | 4.2% | 9.4% | 10.8% | 3.5% |

Exact top-$k$ preserved in 100% of trials, as the construction guarantees.
ANN top-$k$ preserved exactly in only 25-75% of trials at useful budgets.
Recall@10 at ef=256 falls from 0.975 to 0.917 at the largest budget.

## Findings

**The mechanism is real but weak.** Roughly 7-11% extra work, against a targeted
attack with an unrealistically large budget (3200 flips touching 800 of 3000
vectors).

**It saturates in budget.** At ef=64, 200 flips and 3200 flips both give ~7.4%.
Demoting more of $W$ does not help, because the distance distribution near rank
ef is dense: evicted candidates are replaced by ones only marginally worse, so
the stopping threshold barely moves. Graph structure, not the score margin,
bounds how far the search wanders.

**Targeting buys about 2x over random.** Real but small, and it means most of the
measured effect is a generic code-corruption effect rather than anything about
route structure.

**The strict quality constraint is the binding problem.** Exact top-$k$ holds
trivially. ANN top-$k$ equality fails most of the time at any budget that moves
work. Only the $\varepsilon$-tolerant recall constraint survives, and it is
degrading in exactly the regime where the attack starts to work.

## What this does and does not show

Tested: threshold loosening via tail-of-$W$ demotion. Not tested: opening new
graph regions by promoting distant nodes, which is the mechanism the route
spec actually proposes. This is a different lever and could behave differently.

So this does not prove route steering fails. It establishes the number route
steering has to beat (7-11%) and shows that the cheap, no-model version of the
attack does not reach a level anyone would care about.

## Limitations

Single backend, single corpus, synthetic Gaussian data, one graph
configuration. Neighbour selection uses simple top-$M$ rather than the
heuristic pruning from the HNSW paper, which changes graph connectivity and
therefore work. None of this generalises to IVF, ScaNN, or DiskANN.
