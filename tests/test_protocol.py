"""Phase 0: the freeze must be enforced, not merely documented."""

from __future__ import annotations

import copy
import json

import pytest

from braid.protocol import (
    REQUIRED_CLAIM_FAMILIES,
    FreezeViolation,
    Protocol,
    ProtocolError,
    compute_protocol_hash,
    freeze_protocol,
    load_protocol,
)


def test_active_protocol_is_frozen_and_valid(protocol):
    assert protocol.frozen
    assert protocol.recorded_hash == protocol.content_hash
    assert protocol.problems() == []


def test_all_four_claim_families_are_stated_separately(protocol):
    families = protocol.claim_families
    for family in REQUIRED_CLAIM_FAMILIES:
        assert family in families
    for claim in protocol.doc["claims"]:
        assert claim["statement"] and claim["primary_metric"] and claim["exit_gate"]


def test_editing_a_frozen_protocol_breaks_its_hash(protocol, tmp_path):
    doc = copy.deepcopy(protocol.doc)
    doc["hnsw"]["ef_search_grid"] = [10, 20, 50]  # a quiet "improvement"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(FreezeViolation):
        load_protocol(path)


def test_draft_protocol_is_refused_when_frozen_is_required(protocol, tmp_path):
    doc = copy.deepcopy(protocol.doc)
    doc["status"] = "draft"
    doc.pop("protocol_hash", None)
    path = tmp_path / "draft.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(FreezeViolation):
        load_protocol(path, require_frozen=True)
    assert load_protocol(path, require_frozen=False).status == "draft"


def test_freeze_writes_a_verifying_hash(protocol, tmp_path):
    doc = copy.deepcopy(protocol.doc)
    doc["status"] = "draft"
    doc.pop("protocol_hash", None)
    doc.pop("frozen_at", None)
    path = tmp_path / "to_freeze.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    frozen = freeze_protocol(path)
    assert frozen.frozen
    assert frozen.recorded_hash == frozen.content_hash
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["protocol_hash"] == compute_protocol_hash(reloaded)


@pytest.mark.parametrize(
    "mutate,expected_fragment",
    [
        (lambda d: d["hnsw"].__setitem__("ef_search_grid", [50, 20, 10]), "sorted ascending"),
        (lambda d: d["hnsw"].__setitem__("top_k", [1, 100]), "recall is capped"),
        (lambda d: d["hnsw"].pop("neighbor_selection"), "neighbor_selection"),
        (lambda d: d["hnsw"]["parity_tolerance"].__setitem__("build_seeds", 1), "build-seed variance"),
        (lambda d: d["hnsw"].pop("parity_tolerance"), "parity_tolerance must be declared"),
        (lambda d: d["claims"].pop(0), "required family"),
        (lambda d: d["splits"].__setitem__("test_fraction", 0.9), "must sum to 1"),
        (lambda d: d["targets"]["rules"][0].__setitem__("deterministic", False), "deterministic"),
        (lambda d: d["surrogate"].__setitem__("H_is_ef_search", True), "H counts surrogate steps"),
        (lambda d: d["bitflip_policy"].__setitem__("modes", ["finite_only"]), "unrestricted_ieee754"),
        (
            lambda d: d["leakage_policy"].__setitem__("test_unseal_allowed_phases", [1, 8]),
            "both selection phases and allowed",
        ),
        (
            lambda d: d["profiles"]["smoke"].__setitem__("ef_search_grid", [7]),
            "widens ef_search_grid",
        ),
    ],
)
def test_validator_rejects_specific_protocol_defects(protocol, mutate, expected_fragment):
    doc = copy.deepcopy(protocol.doc)
    mutate(doc)
    problems = Protocol(doc=doc).problems()
    assert any(expected_fragment in problem for problem in problems), problems


def test_profile_cannot_widen_the_frozen_grid(protocol):
    for name in protocol.profile_names:
        profile = protocol.profile(name)
        assert set(profile.m_grid) <= set(protocol.m_grid)
        assert set(profile.ef_search_grid) <= set(protocol.ef_search_grid)
        assert set(profile.numeric_types) <= set(protocol.numeric_types)
        assert set(profile.dataset_ids) <= set(protocol.dataset_ids)


def test_only_the_full_profile_is_claim_bearing(protocol):
    claim_bearing = [n for n in protocol.profile_names if protocol.profile(n).claim_bearing]
    assert claim_bearing == ["full"]


def test_unknown_profile_is_an_error(protocol):
    with pytest.raises(ProtocolError):
        protocol.profile("does-not-exist")


def test_build_params_come_from_the_protocol_declaration(protocol):
    params = protocol.hnsw_params(M=16, seed=1)
    selection = protocol.neighbor_selection
    assert params.extend_candidates == selection["extend_candidates"]
    assert params.keep_pruned_connections == selection["keep_pruned_connections"]
    assert params.ef_construction == protocol.ef_construction
    assert params.convention == protocol.convention
    assert params.m0 == 2 * 16
