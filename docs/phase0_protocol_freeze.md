# Phase 0: freeze the protocol and claims

**Exit gate.** No hyperparameter, target rule, eligibility rule, or success
metric can be tuned using Qtest.

Run it:

```bash
python -m braid protocol validate
python -m braid phase0 gate
```

## What is frozen, and where

Everything Phase 0 requires lives in one file, `configs/protocol_v2.json`
(v1 is superseded; see `protocol_changelog.md`).

| Frozen item | Protocol path |
| --- | --- |
| Datasets and corpus sizes | `datasets[]` |
| Embedding models | `embedding_models[]` |
| Numeric storage types | `system.numeric_types` |
| HNSW implementation and version | `system.hnsw_implementation` |
| Distance convention | `system.distance` |
| M values | `hnsw.M_grid` |
| efSearch grid | `hnsw.ef_search_grid` |
| Neighbour-selection rules | `hnsw.neighbor_selection` |
| Seeds | `seeds` |
| Qcal / Qtest split policy | `splits` |
| BV, BF, K grids and nesting rule | `budgets` |
| Finite-value policy and bit classes | `bitflip_policy` |
| Target-selection rules and eligibility | `targets` |
| Recall targets rho | `recall_targets` |
| Primary exact / stale / rebuilt comparisons | `comparisons` |
| Surrogate parameter grids (gamma, H, tau, epsilon, lambda) | `surrogate`, `targets.lambda_grid` |
| Claims, stated separately per family | `claims[]` |
| Leakage policy | `leakage_policy` |

## How the freeze is enforced

Three mechanisms, because a comment saying "do not edit" is not a mechanism.

1. **Content hash.** A frozen protocol stores a SHA-256 over its own canonical
   content. `load_protocol` recomputes it and raises `FreezeViolation` on any
   drift, so an edited value cannot be used by any code path in this
   repository. Changing a frozen value means a new versioned file plus a
   `protocol_changelog.md` entry, and the Phase 0 gate reads that changelog.
2. **Schema validation.** `Protocol.problems()` refuses protocols that are
   internally inconsistent or under-specified: unsorted or duplicated grids,
   an efSearch value below max(top_k), a target rule that is not deterministic
   or not pre-frozen, a missing claim family, a bit-flip policy without both
   the finite-only and unrestricted IEEE-754 conditions, `H_is_ef_search` set
   true, split fractions that do not sum to one, phases that are both
   selection phases and allowed to open Qtest, and run profiles that widen a
   frozen grid.
3. **A sealed test split.** `braid.splits.QuerySplit` does not expose held-out
   query ids. Reading them requires an explicit `unseal(phase=..., reason=...)`
   context; the count `n_test` stays public, since counting cannot tune
   anything. Every attempt, permitted or refused, is appended to
   `results/audit/leakage_audit.jsonl`. The Phase 0 gate fails if any permitted
   unseal came from a phase listed in `leakage_policy.selection_phases`
   (phases 0 to 7), and reports refused attempts as an advisory.

Build parameters are read from the protocol through `Protocol.hnsw_params()`,
so changing how graphs are built also requires a protocol version bump rather
than a code edit.

## Run profiles

The frozen grid is expensive. A profile selects a subset of it and is recorded
in every artifact:

| Profile | Datasets | M | efSearch | Corpus | Claim-bearing |
| --- | --- | --- | --- | --- | --- |
| `full` | all six | 8, 16, 32 | 10 to 400 | as declared | yes |
| `dev` | two synthetic | 8, 16 | 10, 50, 200 | 8000 | no |
| `smoke` | one synthetic | 16 | 10, 50, 200 | 3000 | no |

A profile can only subset; the validator rejects any profile value that is not
in the frozen grid. `full` is the only claim-bearing profile, and the Phase 1
gate reports non-claim-bearing runs as an advisory (or a blocking failure under
`--require-claim-bearing`).

## Claims, stated separately

Four claim families are declared separately, as the phasing requires, plus the
core graph-amplification hypothesis:

| Claim | Family | Primary metric |
| --- | --- | --- |
| C1 | untargeted retrieval failure | `ASR_index` |
| C2 | targeted route steering | change in real target visitation versus clean |
| C3 | targeted retrieval | `ASR_target` |
| C4 | work amplification | distance evaluations and `Delta_ef` at matched recall |
| C0 | graph amplification (core hypothesis) | `ASR_index` with the stale versus exact and rebuilt gap |

Each carries its own exit gate and its own stop condition, so a result under
one objective cannot be quietly reported as evidence for another.

## Known holes in this freeze

These are recorded rather than papered over. Each needs a new protocol version
before the affected work can proceed.

1. `all-MiniLM-L6-v2` is declared with `revision: null` and
   `status: declared_pending_pin`. The `msmarco-minilm-384` dataset is blocked
   until a protocol version pins the model revision.
2. The external corpora (`sift-1m`, `glove-100`, `msmarco-minilm-384`) are
   declared with `available: false` and no checksums. Requesting one raises
   `DatasetUnavailable` with the paths and checksum policy it expects, so a
   missing download can never be silently replaced by synthetic data.
3. The primary implementation is the instrumented reference implementation, not
   hnswlib. Promoting hnswlib to primary, which an externally reproducible
   claim eventually needs, is a protocol version change.
