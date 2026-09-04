# BRAID

Localized gray-box bit-flip stress testing of HNSW vector search.

The research question: can a very small corruption of stored vector bits make
HNSW return the wrong item, when the attacker sees only the local part of the
index that search touches? The scientific test that matters is separating
failures caused by changed vector geometry from failures amplified by the stale
HNSW graph itself.

This repository implements **Phase 0 (freeze the protocol and claims)** and
**Phase 1 (baseline system and instrumentation)**. Nothing here flips a bit.
The bit-flip engine (Phase 2), the geometry objective (Phase 3), the route
surrogate (Phase 4), targeted redirection (Phase 5), and work amplification
(Phase 6) are deliberately absent: each depends on the instrumentation and the
freeze being in place and checked first.

## Quickstart

```bash
pip install -r requirements.txt
python -m braid run --profile smoke      # validate, gate 0, sweep, gate 1
python -m braid phase1 sweep --profile dev
python -m braid phase1 gate
python -m pytest -q
```

Useful one-offs:

```bash
python -m braid protocol validate                       # schema plus freeze hash
python -m braid protocol hash
python -m braid phase1 trace --dataset syn-clusters-d64 --n 3000   # one full search trace
python -m braid phase1 parity --dataset syn-clusters-d64 --M 16    # versus hnswlib
```

## Phase status

| Phase | Status | Exit gate |
| --- | --- | --- |
| 0 protocol and claim freeze | done | `python -m braid phase0 gate` passes; see `docs/phase0_protocol_freeze.md` |
| 1 baseline system and instrumentation | done | `python -m braid phase1 gate` passes; see `docs/phase1_instrumentation.md` |
| 2 exact bit-flip and budget engine | not started | round-trip encoding, single-bit isolation, budget nesting |
| 3 geometry baseline and discrete search | not started | gradient checks, exact flip gains, no continuous bit relaxation |
| 4 untargeted route surrogate validation | not started | surrogate beats random ranking of real route behaviour |
| 5 targeted route redirection | not started | real target visitation above clean, random, and untargeted baselines |
| 6 search-work amplification | not started | real work increase on the exact-retained stratum |
| 7 to 10 | not started | see the research note's phasing appendix |

Phases 0 and 1 pass their gates on the `smoke` profile in this environment. No
claim-bearing (`full` profile) run has been executed here: the external
million-vector corpora are not present, and the gate reports a
non-claim-bearing profile as an advisory so a pipeline check cannot be mistaken
for a result.

## Layout

```
configs/protocol_v3.json     the frozen protocol (v1, v2 superseded; see docs/protocol_changelog.md)
braid/protocol.py            freeze enforcement, schema validation, run profiles
braid/splits.py              Qcal/Qtest with the held-out ids sealed at runtime
braid/audit.py               append-only log of every unseal attempt
braid/datasets.py            synthetic corpora; loader contract for external ones
braid/similarity.py          cosine and L2 conventions
braid/vectors.py             D and its storage type, separate from arithmetic width
braid/exact.py               exact search, checked three independent ways
braid/hnsw/params.py         HNSW parameters, read from the protocol
braid/hnsw/oracle.py         the single place a distance is computed
braid/hnsw/reference.py      instrumented HNSW (Algorithms 1 to 5, hnswlib's concrete rules)
braid/hnsw/trace.py          events, work counters, L(q), N_local(u), exposure policy
braid/hnsw/conditions.py     exact / clean / stale / rebuilt
braid/hnsw/native.py         hnswlib recall parity and serialized-graph parsing
braid/metrics.py             recall, work distributions, e_clean(rho)
braid/sweep.py               the clean M x efSearch sweep and its checks
braid/gates.py               the Phase 0 and Phase 1 exit gates
braid/cli.py                 python -m braid ...
docs/                        phase notes, notation map, protocol changelog
tests/                       99 tests, including gate-failure cases
```

## Three things worth knowing before building on this

1. **The freeze is mechanical.** A frozen protocol carries a hash over its own
   content; editing a value makes every code path refuse to load it. Held-out
   query ids are unreachable without an audited unseal, and the Phase 0 gate
   fails if a selection phase ever opened them. Build parameters come from the
   protocol, so changing how graphs are built needs a version bump.
2. **The primary HNSW is instrumented code here, not hnswlib.** hnswlib exposes
   no visited set, no expansions, no neighbour lists, and cannot search a clean
   graph over corrupted vectors, which is the whole stale condition. Parity
   against hnswlib is therefore a gate check, judged on seed-averaged recall
   across three build seeds per side: worst gap 0.022 on the `dev` profile,
   against a seed spread of 0.052 in the same cell, with layer-0 mean degree
   matching to the first decimal. An externally reproducible claim will still
   have to be reproduced on the deployed index.
3. **Latency here is not a latency claim.** Python wall clock catches gross
   regressions and nothing more. The work-amplification claim's timing
   component needs the native implementation on controlled hardware.
