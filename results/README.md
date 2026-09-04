# Run artifacts

This directory is generated. Only this file and `.gitkeep` are tracked, because
committing run output would make the repository the source of truth for results
instead of the protocol plus the seed.

```
results/
  audit/leakage_audit.jsonl   append-only record of every Qtest unseal attempt
  gates/phase0.json           last Phase 0 exit-gate result
  gates/phase1.json           last Phase 1 exit-gate result
  sweep/<run id>/cells.jsonl  one row per (dataset, numeric type, M, efSearch, k) cell
  sweep/<run id>/summary.json per-dataset and per-M checks, provenance, e_clean(rho)
  sweep/<run id>/trace_sample.json  FULL traces for a query sample, plus the local view
```

A run id is `<UTC timestamp>-<profile>-<protocol hash prefix>`. Every artifact
carries the protocol hash and the profile name, so a result can be traced to
the exact frozen protocol that produced it, and a non-claim-bearing profile
cannot be mistaken for a full run.

Reproduce a run from a clean checkout:

```bash
python -m braid protocol validate
python -m braid phase0 gate
python -m braid phase1 sweep --profile dev
python -m braid phase1 gate
```
