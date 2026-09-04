# Phase 1: baseline system and instrumentation

**Exit gate.** Exact-search answers independently checked, stable work counters
across repeated clean runs, and every recorded local edge traceable to an
instrumented event.

Run it:

```bash
python -m braid phase1 sweep --profile smoke     # or dev, or full
python -m braid phase1 gate
```

`python -m braid run --profile smoke` chains protocol validation, the Phase 0
gate, the sweep, and the Phase 1 gate.

## What was built

### D, G(D), and the four conditions

`braid.vectors.VectorStore` holds D in its declared storage type (FP32 or
FP16) and hands out float32 views for arithmetic. Storage width and arithmetic
width are separate so that Phase 2 flips bits in what the database actually
stores, not in a working copy.

`braid.hnsw.reference.HnswGraph` holds the graph and no vectors. Which vectors
a search compares against is a per-search argument, which makes Section 15's
conditions three uses of one code path instead of three near-copies:

| Condition | Graph | Vectors |
| --- | --- | --- |
| `exact` | none | D' |
| `hnsw_clean` | G(D) | D |
| `hnsw_stale` | G(D) | D' |
| `hnsw_rebuilt` | G(D') | D' |

With D' = D, `identity_check` requires clean and stale to agree on ids, scores,
and deterministic counters. That identity is what later licenses attributing a
stale-versus-clean gap to the corruption rather than to two subtly different
code paths.

### Instrumentation

Every distance passes through `braid.hnsw.oracle.DistanceOracle`, so counters
cannot drift from what the search actually did. `braid.hnsw.trace` records:

- **entry point** (node and layer), per query;
- **layer enter / exit** and **greedy hops** per layer;
- **neighbour-list exposure**, which is the only channel that defines
  N_local(u);
- **distance evaluations**, which is the only channel that defines L(q);
- **candidate pushes, candidate pops (expansions), visited additions, result
  prunes**;
- **stopping events**, separated into `candidate_worse_than_furthest`,
  `candidate_queue_empty`, and `greedy_local_minimum`;
- **per-query latency**.

Each exposed local edge stores the sequence number of the event that exposed
it. `_provenance_check` in `braid.sweep` verifies, for every edge, that the
named event exists, is a `neighbor_list_exposed` event, is on the same node and
layer, and lists that neighbour; and that every node in L(q) was touched by a
distance-evaluation event. That is the gate's traceability requirement in
executable form.

Three trace levels exist. `NONE` for warm-up, `COUNTERS` for the sweep
(counters plus L(q) and the local edges, no event bodies), `FULL` for
provenance checking. `test_trace.py` asserts that the trace level never changes
the answer.

Exposure is a declared policy (`threat_model` or `white_box`) recorded in every
trace, so the global white-box upper bound of Section 17 can never be reported
as a gray-box number.

### Exact search, checked three ways

`braid.exact` provides a chunked matrix implementation (used in experiments), a
per-query loop written from the definitions, and hnswlib's brute-force index.
Agreement is judged on scores rather than id lists, because with genuine ties
two correct implementations may return different ids; the tie rate is reported
alongside. `cross_check_exact` is run per dataset in every sweep and is a
blocking gate check.

### The clean sweep

`braid.sweep.run_clean_sweep` produces, per (dataset, numeric type, M,
efSearch, k) cell, on calibration queries only:

- recall@k, top-1 hit rate, and the rate at which i*(q) appears in the top-k;
- the four work quantities of Section 16.3 as distributions with 95 percent
  intervals: expanded nodes, unique visited nodes, distance evaluations, and
  latency, kept apart rather than merged;
- e_clean(rho) for every frozen recall target, with right-censoring recorded
  rather than replaced by the largest tested efSearch, since Delta_ef is later
  defined against this number;
- repeated identical runs (counter stability), a repeated build (structure hash
  determinism), a FULL-trace sample (edge provenance and the knowledge fraction
  |A| / N), the D' = D identity check, and hnswlib recall parity.

Artifacts land in `results/sweep/<run id>/`: `cells.jsonl`, `summary.json`, and
`trace_sample.json`. Every artifact carries the protocol hash and the profile
name.

One deliberate shortcut: for a given efSearch the search runs once at
k = max(top_k) and is truncated for smaller k. HNSW's layer-0 search depends on
k only through `ef = max(ef_search, k)`, and the validator enforces
efSearch >= max(top_k), so truncation is exact. `test_reference_hnsw.py` pins
that equivalence.

## Fidelity to the deployed implementation

The primary implementation is a reference HNSW written here, because Phase 1
needs visited sets, expansions, neighbour lists, stopping events, and the
ability to search a clean graph over corrupted vectors, none of which hnswlib
exposes. The cost of that choice is that the attacks in Phases 3 to 7 will be
optimized against this code, not against a deployed system. Two things contain
that risk.

First, the build follows hnswlib's concrete rules, not a loose reading of the
paper: at most M links for a newly inserted element at every layer, the larger
`max_M0` cap used only when shrinking an existing node's list, descent to the
next layer from the closest selected neighbour, `extendCandidates` false, and
no retention of pruned connections. The last one matters most and was
originally declared wrong. Retaining pruned connections (the paper's
`keepPrunedConnections = true`) raised mean layer-0 degree from about M to
about 1.6M and lifted recall at efSearch 10 from 0.92 to 0.98 while hnswlib
sat at 0.91. A robustness result measured on that denser graph would have
overstated the deployed system's resilience. The correction is recorded in
`protocol_changelog.md` as the reason v1 was superseded, and
`keep_pruned_connections` is now a declared Phase 8 ablation.

Second, `braid.hnsw.native` builds an hnswlib index over the same corpus,
compares recall per efSearch cell against a declared tolerance (0.05 absolute),
and parses hnswlib's serialized index to compare graph statistics. Current
parity on `syn-clusters-d64` (n = 3000, M = 16): layer-0 mean degree 16.4
against 16.4, edges 51177 against 51192, worst recall gap 0.014.

Two hnswlib quirks are recorded rather than worked around: `space="cosine"`
normalizes stored copies at insert time, which would move the attack surface
off raw stored coordinates, and `BFIndex(space="cosine")` in 0.8.0 does not
normalize at all. Parity runs therefore pre-normalize and use the
inner-product space, which is numerically the same ranking under our
convention.

## Limitations to carry into later phases

1. **Latency here is not the paper's latency.** Per-query wall clock from a
   Python search is adequate for spotting gross regressions and useless as a
   timing claim. Section 16.3's timing discipline needs the native
   implementation on controlled hardware, so C4's latency component cannot be
   settled with this code.
2. **Scale.** Builds are O(minutes) at n = 20000 and the external
   million-vector corpora are not present in this environment. The full
   profile has never been run here; every artifact in `results/` so far comes
   from a non-claim-bearing profile, and the gate says so.
3. **Synthetic geometry.** Three synthetic corpora (clustered, isotropic,
   norm-skewed) exercise the pipeline and the norm-sensitivity of exponent
   flips, but they are not embeddings. No claim about real retrieval survives
   without the external datasets of Phase 8.
4. **The reference implementation is not a deployed system.** Parity is
   measured on recall and degree, not on candidate-set dynamics. A route
   surrogate validated against this implementation still needs Phase 4's
   validation gate against real traces, and eventually against hnswlib.
