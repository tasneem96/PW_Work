# Protocol changelog

A frozen protocol carries a SHA-256 over its own canonical content. Loading it
recomputes that hash and refuses the file if it drifted, so a frozen value
cannot be edited quietly. Changing anything means writing a new versioned
protocol file (`configs/protocol_v2.json`, and so on) and adding an entry here.
The Phase 0 gate reads this file and fails if the active protocol's id and hash
prefix are not recorded.

## braid-protocol-v2 (active)

- File: `configs/protocol_v2.json`
- Status: frozen
- Content hash: `d9375abeab7a43c173620127fd76ef8c4c7f0e64739700adce103e04b24a915c`
- Frozen at: 2026-09-04 (UTC)
- Supersedes: `braid-protocol-v1`

Why v1 was superseded, in one line: it mis-stated hnswlib's behaviour. v1
described the reference implementation as using the paper's
`keepPrunedConnections = true` "hnswlib default". hnswlib's
`getNeighborsByHeuristic2` does not keep pruned connections, and retaining them
raised mean layer-0 degree from about M to about 1.6M, which lifted reference
recall at efSearch = 10 to 0.98 against hnswlib's 0.91. A robustness result
measured on a denser, easier-to-search graph would have overstated the deployed
system's resilience.

Changes from v1:

1. `hnsw.neighbor_selection` is now declared explicitly: `extend_candidates`
   false, `keep_pruned_connections` false, at most M links selected per
   insertion at every layer, the larger cap used only when shrinking an
   existing node's list, and descent from the closest selected neighbour. The
   protocol validator now requires this block, which is why v1 no longer
   validates.
2. `hnsw.parity_tolerance` declares the reference-versus-hnswlib recall
   tolerance (0.05 absolute) that the Phase 1 gate enforces.
3. `keep_pruned_connections` is registered as a Phase 8 ablation rather than a
   silent build choice.
4. The implementation and cross-check descriptions record the corrected
   hnswlib behaviour.

Measured effect of the correction (syn-clusters-d64, n = 3000, M = 16,
ef_construction = 200, 64 calibration queries, recall@10):

| build variant | layer-0 mean degree | ef=10 | ef=20 | ef=50 |
| --- | --- | --- | --- | --- |
| reference, keep_pruned = true (v1) | 25.7 | 0.981 | 1.000 | 1.000 |
| reference, keep_pruned = false (v2) | 16.4 | 0.920 | 0.986 | 1.000 |
| hnswlib 0.8.0 | 16.4 | 0.914 | 0.972 | 1.000 |

### Code-level notes recorded against v2

- The `perturbed_corpus` query generator interprets the declared `sigma` as a
  relative perturbation norm (noise norm about `sigma` times the mean corpus
  norm), not a per-coordinate standard deviation. The per-coordinate reading
  made the perturbation grow with sqrt(d) and left d = 128 queries nearly
  orthogonal to their source points. The declared value (0.12, 0.15) is
  unchanged; the generator semantics are pinned by the package version, and no
  results predate the correction.
- Query streams are addressed by corpus size as well as dataset id, so a run
  that subsets the corpus draws queries from vectors that are actually present.
  The corpus size travels in every dataset fingerprint.

## braid-protocol-v1 (superseded)

- File: `configs/protocol_v1.json`
- Status: superseded by v2; kept for the record and no longer valid under the current validator
- Content hash: `01d3044f48590a243ed141972e5a48bd84e78909436201b85cffbe4926825955`
- Frozen at: 2026-09-04 (UTC)
- Scope: Phase 0 freeze covering datasets, embedding models, numeric types,
  HNSW implementation and version, distance convention, M values, the efSearch
  grid, seeds, the Qcal/Qtest split policy, the BV/BF/K grids, the finite-value
  policy, the target-selection rules, the recall targets, the primary
  exact/stale/rebuilt comparisons, the surrogate parameter grids, and the four
  claim families stated separately.

### Open items that require a new protocol version before use

1. `all-MiniLM-L6-v2` is declared with `revision: null` and
   `status: declared_pending_pin`. No claim may cite that model until a
   protocol version pins its revision hash. This blocks the
   `msmarco-minilm-384` dataset.
2. The external corpora (`sift-1m`, `glove-100`, `msmarco-minilm-384`) are
   declared with `available: false`. Their file checksums must be recorded
   here before their first use, and adding a checksum is a protocol version
   change, not an edit to v1.
3. The primary HNSW implementation is the instrumented reference
   (`braid.hnsw.reference`), with hnswlib 0.8.0 as the cross-check. Promoting
   hnswlib to primary, which any externally reproducible claim eventually
   needs, is a protocol version change.
