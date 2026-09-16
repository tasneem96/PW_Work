# Delay attack: threat models and what the measurements show

All on 8000x32, M=16 / efConstruction=500 / ef=64, distance-eval inflation at
matched exact top-k.

## The framework's model: per-target, transient

Each target query gets its own independent budget of f bits. Candidate vectors
come from that target's observed search trace, minus the true neighbour. The
optimiser runs fresh per target. Application is transient: the touched integer
rows are copied, bits XORed into the copy, the dequantised matrix patched, one
search run, then the rows restored in a finally block. Stored codes are never
mutated; the next target starts clean.

So "f = 100" means 100 bits on ONE query, not 100 bits across the query set,
and not one flip set serving all queries.

Collateral is the only place the shared store enters: while one target's flips
are live, a held-out bystander population is re-scored and the fraction losing
exact top-1 is counted (the collateral column). On GloVe-100 it stays ~0.0006%,
i.e. a handful of bystander losses across all (target, bystander) pairs; report
it as a count with its denominator, not as a per-instance rate.

Under this model, per-target lure injection scales with the per-target budget:

| f (bits/target) | lure  | tail | exact top-k |
|-----------------|-------|------|-------------|
| 200             | 1.4%  | 5.8% | 100%        |
| 800             | 25.2% | 5.5% | 100%        |

## Shared-budget model: one flip set for many queries (NOT run in the framework)

A separate experiment: one persistent flip set that must inflate work across a
query set. `experiments/global_small_budget.py` previews it with a greedy over
the real searcher. It saturates near zero:

| B (global) | work+ | exact top-k | recall |
|------------|-------|-------------|--------|
| 1          | 0.1%  | 100%        | 0.958  |
| 5          | 0.4%  | 100%        | 0.958  |
| 8          | 0.4%  | 100%        | 0.958  |

Every greedy pick is b7 (largest single-vector move) and it still buys ~0.4%.
Reason: a fixed-direction flip cannot serve a distribution of query directions.
This is a different objective (sum over a query set), not a reinterpretation of
the per-target budget.

## The caveat that matters for the delay arm

Per-target-transient is the natural threat model for retrieval corruption
(geo / target): a victim query is corrupted at the moment it is issued, with
collateral measured separately. Delay's value is aggregate and sustained, which
is the shared-budget model, where the effect is ~0.4%. So the delay arm has a
strong number in the model whose threat is hard to motivate and a weak number
in the model whose threat is real. geo / target does not have this asymmetry.

## Open items before review

- Shared-budget delay ("can one flip set slow many queries") is unanswered; the
  per-target budget cannot be reinterpreted as global.
- Budget is bits, not vectors: one target's f flips can spread over f distinct
  vectors. If a per-vector cap (B_V) is claimed, the baseline script must
  enforce the same cap as the main method or the comparison is confounded.

## Per-query transient at realistic small budgets (f = 1..32)

The framework's own model, untargeted, fine budget sweep. Mean distance-eval
inflation across queries:

| f  | lure | tail | random |
|----|------|------|--------|
| 1  | 0.4% | 0.1% | 0.0%   |
| 3  | 0.0% | 0.1% | 0.2%   |
| 5  | 0.6% | 0.2% | 0.1%   |
| 16 | 0.4% | 0.8% | 0.4%   |
| 32 | 0.6% | 1.4% | 0.4%   |

Noise, and non-monotonic (f=2 gives 0.5%, f=3 gives 0.0%). Indistinguishable
from random flips. Exact top-k 100%, recall flat at 0.953 throughout.

The 25% figure appears only at f=800 bits on a single query, a per-query budget
no bit-flip threat model supports. Between 1 and 32 bits there is no attack.

## Bottom line

Across every realistic configuration measured, delay is negligible: global
small budget ~0.4% (saturated), per-query small budget noise, shared budget
~0.4%. Only f=800 bits/query produces a headline number. HNSW search work is
governed by graph connectivity and the stopping rule, and a realistic number of
stored-code flips cannot move either while preserving recall. Report delay as a
bounded negative result; geo/target is the contribution. Remaining check:
confirm the small-budget curve on real GloVe-100 so the negative is stated on
the headline dataset, not this toy.
