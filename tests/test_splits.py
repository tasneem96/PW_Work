"""Phase 0: held-out queries are sealed, and every attempt on them is logged."""

from __future__ import annotations

import numpy as np
import pytest

from braid.splits import LeakageError, make_split


def test_split_is_deterministic_disjoint_and_covering(protocol, temp_log):
    first = make_split(protocol, "syn-clusters-d64", 200, log=temp_log)
    second = make_split(protocol, "syn-clusters-d64", 200, log=temp_log)
    assert first.fingerprint()["split_hash"] == second.fingerprint()["split_hash"]
    assert first.is_disjoint()
    assert first.covers(200)
    assert first.n_cal + first.n_test == 200


def test_different_datasets_get_different_splits(protocol, temp_log):
    a = make_split(protocol, "syn-clusters-d64", 200, log=temp_log)
    b = make_split(protocol, "syn-iso-d128", 200, log=temp_log)
    assert not np.array_equal(a.cal_ids, b.cal_ids)


def test_test_ids_are_unreachable_without_unsealing(protocol, temp_log):
    split = make_split(protocol, "syn-clusters-d64", 200, log=temp_log)
    with pytest.raises(LeakageError):
        split.test_ids()
    # the count is public; the ids are not
    assert split.n_test > 0


def test_unsealing_in_an_allowed_phase_works_and_is_logged(protocol, temp_log):
    split = make_split(protocol, "syn-clusters-d64", 200, log=temp_log)
    with split.unseal(phase=8, reason="held-out evaluation", protocol=protocol) as token:
        ids = split.test_ids(unseal_token=token)
    assert ids.size == split.n_test
    records = temp_log.of_kind("test_split_unseal")
    assert len(records) == 1
    assert records[0]["detail"]["permitted"] is True
    assert records[0]["detail"]["phase"] == 8
    # the token stops working once the context exits
    with pytest.raises(LeakageError):
        split.test_ids(unseal_token=token)


def test_unsealing_from_a_selection_phase_is_refused_and_logged(protocol, temp_log):
    split = make_split(protocol, "syn-clusters-d64", 200, log=temp_log)
    with pytest.raises(LeakageError):
        with split.unseal(phase=3, reason="just peeking at the tuning set", protocol=protocol):
            pass
    records = temp_log.of_kind("test_split_unseal")
    assert len(records) == 1
    assert records[0]["detail"]["permitted"] is False
    assert records[0]["detail"]["phase"] == 3


def test_fingerprint_does_not_reveal_held_out_ids(protocol, temp_log):
    split = make_split(protocol, "syn-clusters-d64", 200, log=temp_log)
    fingerprint = split.fingerprint()
    assert "test_ids" not in fingerprint
    assert set(fingerprint) == {
        "dataset_id",
        "root_seed",
        "n_cal",
        "n_test",
        "cal_fraction",
        "test_fraction",
        "split_hash",
        "sealed",
    }
