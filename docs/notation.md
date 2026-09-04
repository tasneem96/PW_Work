# Notation to code

The compact notation reference of the research note, mapped to where each
symbol lives. Symbols for phases that are not implemented yet are listed so
that later work lands in the intended place instead of inventing a parallel
vocabulary.

| Symbol | Meaning | Where |
| --- | --- | --- |
| D | clean stored vectors | `braid.vectors.VectorStore` |
| D' | stored vectors after real bit flips | same type, `label="D'"`; produced in Phase 2 |
| e_i, e_{i,c} | stored vector i, feature c of it | `store.row(i)`, `store.data[i, c]` |
| q, Q | one query, a query set | `numpy` arrays; `Dataset.queries` |
| Qcal, Qtest | calibration and held-out queries | `braid.splits.QuerySplit` (test ids sealed) |
| s(q, e), d(q, e) | similarity, distance | `braid.similarity`, `braid.hnsw.oracle.DistanceOracle` |
| i*(q) | clean exact nearest neighbour | `braid.exact.ExactResult.nearest` |
| Exact_k(q; D') | exact top-k | `braid.exact.exact_topk` |
| G(D) | the HNSW graph | `braid.hnsw.reference.HnswGraph` |
| L(q) | local search trace: nodes whose distances were evaluated | `QueryTrace.local_pool()` |
| R(q) | L(q) without i*(q) | `QueryTrace.wrong_candidates(correct_id)` |
| A | union of L(q) over Qcal | `braid.hnsw.trace.merge_local_view` |
| \|A\| / N | knowledge fraction | `braid.hnsw.trace.knowledge_fraction` |
| N_local(u) | locally visible neighbours of u | `QueryTrace.neighbors(u, layer)` |
| u_0(q) | local entry node | `QueryTrace.entry_node` |
| M, efConstruction, efSearch, max_M0, mL | HNSW parameters | `braid.hnsw.params.HnswParams`, protocol `hnsw` |
| recall@k, Amplification(K), Recovery(K) | recall and its condition gaps | `braid.metrics` |
| e_clean(rho), e_stale(K; rho), Delta_ef | matched-recall efSearch | `braid.metrics.ef_at_recall`; Delta_ef in Phase 6 |
| v_i, f_{i,c}, b_{i,c,l} | vector, feature, and bit switches | Phase 2; budgets declared in protocol `budgets` |
| BV, BF, K | vector, feature, and bit budgets | protocol `budgets` |
| Bits, FlipBit_l, Decode, delta_{i,c,l} | exact bit-flip model | Phase 2 |
| J_geo | geometry objective | Phase 3 |
| P_gamma, tilde P_gamma, pi_h, P_hit, J_route | route surrogate | Phase 4; grids in protocol `surrogate` |
| v_tar(q), B_tar(q), P_tar, P_hit_tar, J_target | targeted route | Phase 5; rules in protocol `targets` |
| T(q), J_delay, c(u), J_cost | search-work amplification | Phase 6; cost proxies in protocol `surrogate.cost_proxies` |
| g^J_{i,c,l} | first-order gain of a real bit flip | Phase 3 onwards |
| ASR_index, ASR_target | attack success rates | Phase 8; declared in protocol `claims` |

Conventions worth stating once:

- Similarity is always "larger is better" and distance always "smaller is
  better", including under the L2 convention, where s = -d. Every ranking rule
  in the codebase is therefore shared between conventions.
- The cosine distance used by HNSW is d = 1 - s, matching hnswlib.
- Bit flips target raw stored coordinates. A deployment that stores
  pre-normalized vectors is a separate condition and must be reported
  separately (protocol `system.distance.storage`).
