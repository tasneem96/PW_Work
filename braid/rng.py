"""Deterministic, label-addressed random streams.

Every stochastic decision in the project (corpus synthesis, query sampling,
the Qcal/Qtest split, HNSW level assignment, tie-breaking) draws from a stream
derived from the frozen root seed plus a textual label. Two different labels
never share a stream, and the same label always reproduces the same stream,
so a rerun does not need any stored state beyond the root seed.
"""

from __future__ import annotations

import hashlib

import numpy as np


def _label_entropy(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def seed_sequence(root_seed: int, *labels: str) -> np.random.SeedSequence:
    """SeedSequence for ``root_seed`` mixed with a slash-joined label path."""
    if not labels:
        return np.random.SeedSequence(int(root_seed))
    label = "/".join(str(part) for part in labels)
    return np.random.SeedSequence([int(root_seed), _label_entropy(label)])


def generator(root_seed: int, *labels: str) -> np.random.Generator:
    """A fresh ``np.random.Generator`` for ``(root_seed, labels)``."""
    return np.random.default_rng(seed_sequence(root_seed, *labels))


def derive_int(root_seed: int, *labels: str, bits: int = 31) -> int:
    """A deterministic non-negative integer seed, for APIs that want an int."""
    state = seed_sequence(root_seed, *labels).generate_state(1, dtype=np.uint32)[0]
    return int(state) % (1 << bits)
