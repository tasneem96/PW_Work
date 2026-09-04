"""Similarity and distance conventions (Sections 3.2 and 9.2 of the note).

Two conventions are supported and the choice is frozen by the protocol:

``cosine``
    s(q, e) = <q, e> / (||q|| ||e||) and d(q, e) = 1 - s(q, e).
``l2``
    d(q, e) = ||q - e||^2 and s(q, e) = -d(q, e), so that "larger s is better"
    holds under both conventions and every downstream ranking rule is shared.

All arithmetic is float32 regardless of storage dtype; see
:mod:`braid.vectors` for why storage width and arithmetic width are separate.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

Convention = Literal["cosine", "l2"]
CONVENTIONS: tuple[Convention, ...] = ("cosine", "l2")

EPS_NORM = np.float32(1e-12)


def check_convention(convention: str) -> Convention:
    if convention not in CONVENTIONS:
        raise ValueError(f"unknown distance convention {convention!r}; expected one of {CONVENTIONS}")
    return convention  # type: ignore[return-value]


def as_f32(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def norms(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 norms, floored away from zero."""
    out = np.linalg.norm(as_f32(matrix), axis=-1)
    return np.maximum(out.astype(np.float32), EPS_NORM)


def normalize(matrix: np.ndarray) -> np.ndarray:
    m = as_f32(matrix)
    return m / norms(m)[..., None]


def similarity(query: np.ndarray, vectors: np.ndarray, convention: Convention = "cosine") -> np.ndarray:
    """Similarity of one query against a stack of vectors ("larger is better")."""
    check_convention(convention)
    q = as_f32(query).reshape(-1)
    v = as_f32(vectors).reshape(-1, q.shape[0])
    if convention == "cosine":
        qn = float(np.maximum(np.linalg.norm(q), EPS_NORM))
        return (v @ q) / (norms(v) * qn)
    diff = v - q
    return -np.einsum("ij,ij->i", diff, diff)


def distance(query: np.ndarray, vectors: np.ndarray, convention: Convention = "cosine") -> np.ndarray:
    """Distance of one query against a stack of vectors ("smaller is better")."""
    check_convention(convention)
    if convention == "cosine":
        return np.float32(1.0) - similarity(query, vectors, "cosine")
    return -similarity(query, vectors, "l2")


def similarity_matrix(queries: np.ndarray, vectors: np.ndarray, convention: Convention = "cosine") -> np.ndarray:
    """|Q| x |V| similarity matrix ("larger is better")."""
    check_convention(convention)
    q = as_f32(queries).reshape(1, -1) if as_f32(queries).ndim == 1 else as_f32(queries)
    v = as_f32(vectors)
    if convention == "cosine":
        return (q @ v.T) / np.outer(norms(q), norms(v))
    qq = np.einsum("ij,ij->i", q, q)[:, None]
    vv = np.einsum("ij,ij->i", v, v)[None, :]
    return -(qq + vv - 2.0 * (q @ v.T)).astype(np.float32)


def distance_from_similarity(sim: np.ndarray, convention: Convention = "cosine") -> np.ndarray:
    check_convention(convention)
    if convention == "cosine":
        return np.float32(1.0) - sim
    return -sim
