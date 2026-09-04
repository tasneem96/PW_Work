"""Metrics: recall, work distributions, and the pre-attack e_clean(rho)."""

from __future__ import annotations

import numpy as np
import pytest

from braid.metrics import (
    amplification,
    contains_rate,
    ef_at_recall,
    paired_difference,
    recall_at_k,
    recovery,
    top1_hit_rate,
    work_distribution,
)


def test_recall_at_k_counts_overlap_with_the_exact_answer():
    truth = [{1, 2, 3}, {4, 5, 6}]
    retrieved = np.array([[1, 2, 9], [7, 8, 9]])
    assert recall_at_k(truth, retrieved, 3) == pytest.approx(1 / 3)


def test_recall_ignores_padding_ids():
    truth = [{1, 2}]
    retrieved = np.array([[1, -1]])
    assert recall_at_k(truth, retrieved, 2) == pytest.approx(0.5)


def test_top1_and_contains_rates_differ():
    truth_top1 = [1, 2]
    retrieved = np.array([[9, 1], [2, 9]])
    assert top1_hit_rate(truth_top1, retrieved) == pytest.approx(0.5)
    assert contains_rate(truth_top1, retrieved, 2) == pytest.approx(1.0)


def test_e_clean_finds_the_smallest_ef_meeting_the_target():
    recall = {10: 0.80, 20: 0.91, 50: 0.97}
    result = ef_at_recall(recall, 0.9, [10, 20, 50])
    assert result == {"rho": 0.9, "ef": 20, "right_censored": False}


def test_e_clean_reports_right_censoring_instead_of_the_largest_value():
    recall = {10: 0.5, 20: 0.6}
    result = ef_at_recall(recall, 0.95, [10, 20])
    assert result["right_censored"] is True
    assert result["ef"] is None
    assert result["max_ef_tested"] == 20
    assert result["best_recall"] == 0.6


def test_work_distribution_reports_more_than_a_mean():
    summary = work_distribution([1, 2, 3, 4, 100])
    assert summary["p95"] > summary["p50"]
    assert summary["ci95_high"] > summary["mean"] > summary["min"]
    assert summary["n"] == 5


def test_amplification_and_recovery_are_plain_differences():
    assert amplification(0.9, 0.4) == pytest.approx(0.5)
    assert recovery(0.85, 0.4) == pytest.approx(0.45)


def test_paired_difference_reports_per_query_structure():
    result = paired_difference([2, 2, 2], [1, 2, 3])
    assert result["n_pairs"] == 3
    assert result["mean_difference"] == pytest.approx(0.0)
    assert result["fraction_positive"] == pytest.approx(1 / 3)
    assert result["fraction_zero"] == pytest.approx(1 / 3)


def test_paired_difference_requires_matched_shapes():
    with pytest.raises(ValueError):
        paired_difference([1, 2], [1, 2, 3])
