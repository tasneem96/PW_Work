"""Metrics available in Phase 1: recall, work distributions, and e_clean(rho).

The attack-success metrics of Section 16 (ASR_index, ASR_target, target
visitation, Delta_ef) need a corrupted database and therefore belong to later
phases. What Phase 1 needs is the clean side of every one of those
comparisons: recall@k per grid cell, the four work quantities kept separate
(expanded nodes, unique visited nodes, distance evaluations, latency), and
e_clean(rho), the smallest efSearch on the frozen grid that meets a recall
target. Delta_ef is later defined against exactly this number, so computing it
here, from the clean sweep, is what stops it from being fitted after the fact.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def recall_at_k(truth: Sequence[set[int]], retrieved: np.ndarray, k: int) -> float:
    """Mean fraction of each query's exact top-k that was retrieved."""
    retrieved = np.asarray(retrieved)
    if retrieved.ndim == 1:
        retrieved = retrieved[None, :]
    if len(truth) != retrieved.shape[0]:
        raise ValueError(f"truth has {len(truth)} rows, retrieved has {retrieved.shape[0]}")
    scores = []
    for row, gold in zip(retrieved, truth):
        if not gold:
            continue
        found = set(int(i) for i in row[: int(k)] if i >= 0)
        scores.append(len(found & gold) / float(min(int(k), len(gold))))
    return float(np.mean(scores)) if scores else 0.0


def top1_hit_rate(truth_top1: Sequence[int], retrieved: np.ndarray) -> float:
    """Fraction of queries whose i*(q) is the first retrieved id."""
    retrieved = np.asarray(retrieved)
    return float(np.mean([int(r[0]) == int(t) for r, t in zip(retrieved, truth_top1)]))


def contains_rate(truth_top1: Sequence[int], retrieved: np.ndarray, k: int) -> float:
    """Fraction of queries whose i*(q) appears anywhere in the top-k.

    This is the quantity a later ASR_index inverts: an index-specific failure
    is a query where exact search still holds i*(q) but stale HNSW does not.
    """
    retrieved = np.asarray(retrieved)
    return float(
        np.mean([int(t) in set(int(i) for i in r[: int(k)]) for r, t in zip(retrieved, truth_top1)])
    )


def work_distribution(values: Iterable[float]) -> dict[str, float]:
    """Distribution summary, not just a mean.

    Section 16.3 asks for distributions and intervals rather than a single
    number, and corruption effects on search work are expected to be
    heavy-tailed, where a mean alone hides the interesting queries.
    """
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {}
    mean = float(arr.mean())
    sem = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
    return {
        "n": int(arr.size),
        "mean": mean,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
        "min": float(arr.min()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def work_summary(traces) -> dict[str, dict[str, float]]:
    """The four work quantities of Section 16.3, kept apart."""
    if not traces:
        return {}
    summaries = [t.work_summary() for t in traces]
    keys = ("distance_evals", "expansions", "unique_visited", "neighbor_lists_exposed", "latency_ns")
    return {key: work_distribution([s[key] for s in summaries]) for key in keys}


def ef_at_recall(
    recall_by_ef: dict[int, float], rho: float, ef_grid: Sequence[int]
) -> dict[str, Any]:
    """e_clean(rho) = min{e in E : recall(e) >= rho}, with censoring reported.

    When no efSearch on the frozen grid reaches ``rho`` the result is
    right-censored. Section 16.3 forbids replacing that with the largest tested
    value, so the censored flag travels with the number.
    """
    ordered = sorted(int(e) for e in ef_grid)
    for ef in ordered:
        if float(recall_by_ef.get(ef, float("nan"))) >= float(rho):
            return {"rho": float(rho), "ef": int(ef), "right_censored": False}
    return {
        "rho": float(rho),
        "ef": None,
        "right_censored": True,
        "max_ef_tested": ordered[-1] if ordered else None,
        "best_recall": max(recall_by_ef.values()) if recall_by_ef else None,
    }


def amplification(recall_exact: float, recall_stale: float) -> float:
    """Amplification(K) = R_exact(K) - R_HNSW,stale(K) (Section 16.1)."""
    return float(recall_exact) - float(recall_stale)


def recovery(recall_rebuilt: float, recall_stale: float) -> float:
    """Recovery(K) = R_HNSW,rebuild(K) - R_HNSW,stale(K) (Section 16.1)."""
    return float(recall_rebuilt) - float(recall_stale)


def paired_difference(a: Iterable[float], b: Iterable[float]) -> dict[str, float]:
    """Query-level paired effect of two conditions on the same queries."""
    x = np.asarray(list(a), dtype=np.float64)
    y = np.asarray(list(b), dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"paired comparison needs equal shapes, got {x.shape} and {y.shape}")
    diff = x - y
    mean = float(diff.mean())
    sem = float(diff.std(ddof=1) / np.sqrt(diff.size)) if diff.size > 1 else 0.0
    return {
        "n_pairs": int(diff.size),
        "mean_difference": mean,
        "sem": sem,
        "ci95_low": mean - 1.96 * sem,
        "ci95_high": mean + 1.96 * sem,
        "fraction_positive": float(np.mean(diff > 0)),
        "fraction_zero": float(np.mean(diff == 0)),
    }
