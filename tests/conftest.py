"""Shared fixtures. Tests run on small corpora; the frozen grid is for real runs."""

from __future__ import annotations

import numpy as np
import pytest

from braid.audit import AuditLog
from braid.datasets import load_dataset
from braid.protocol import load_protocol
from braid.rng import derive_int
from braid.vectors import make_store

SMALL_N = 600
SMALL_Q = 32


@pytest.fixture(scope="session")
def protocol():
    return load_protocol()


@pytest.fixture
def temp_log(tmp_path) -> AuditLog:
    return AuditLog(tmp_path / "leakage_audit.jsonl")


@pytest.fixture(scope="session")
def small_dataset(protocol):
    return load_dataset(
        protocol, "syn-clusters-d64", numeric_type="fp32", n=SMALL_N, n_queries=SMALL_Q
    )


@pytest.fixture(scope="session")
def small_params(protocol):
    return protocol.hnsw_params(M=8, seed=derive_int(protocol.root_seed, "test", "M=8"))


@pytest.fixture
def tiny_store():
    rng = np.random.default_rng(11)
    return make_store(rng.normal(size=(120, 16)), "fp32", label="tiny")
