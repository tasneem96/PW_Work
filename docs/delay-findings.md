# Delay attack: what the measurements actually show

All on 8000x32, M=16 / efConstruction=500 / ef=64, distance-eval inflation at
matched exact top-k. Two threat models, very different answers.

## Per-query tailored flips (optimistic, not the real model)

Flips recomputed for each query. Lure injection scales with budget:

| B   | lure  | tail | exact top-k |
|-----|-------|------|-------------|
| 200 | 1.4%  | 5.8% | 100%        |
| 800 | 25.2% | 5.5% | 100%        |

This requires re-tailoring the corpus to each incoming query. It is a
query-aware upper bound, not a persistent memory-fault attack.

## Global persistent flips (the honest model, realistic budget)

One flip set shared by all queries, greedy over the real searcher:

| B | work+ | exact top-k | recall |
|---|-------|-------------|--------|
| 1 | 0.1%  | 100%        | 0.958  |
| 2 | 0.2%  | 100%        | 0.958  |
| 5 | 0.4%  | 100%        | 0.958  |
| 8 | 0.4%  | 100%        | 0.958  |

Saturates at flip 5. Every flip greedily chosen is b7 (the largest single-
vector move), and it still buys 0.4%.

## Why

A persistent flip moves one stored vector in one fixed direction. "Toward the
query" points a different way for every query, so no single lure serves a query
distribution. Aggregate search work is therefore nearly immovable by a handful
of global flips. This is a property of the geometry, not of the optimizer.

## Consequence for the paper

Delay via persistent stored-code flips is negligible at realistic bit-flip
budgets. Report it as a bounded negative result. The retrieval-corruption arms
(geo / target), which manipulate ranking rather than control flow, are the
strong contribution; delay characterizes the limit of the threat model.
