"""The stored-vector database D (Section 3.1) and its numeric storage type.

Storage width and arithmetic width are deliberately separate. The attack model
flips bits in the *stored* representation (FP32 or FP16, Section 7), while
HNSW and exact search compare in float32. A ``VectorStore`` therefore keeps the
canonical stored array in its declared dtype and hands out float32 views for
arithmetic, so no code path can silently attack a value that the database does
not actually store.

Nothing here flips bits. The bit-level encode/flip/decode engine is Phase 2;
this module only fixes the representation that Phase 2 will operate on, plus
the read-side invariants (finiteness, shape, dtype) that Phase 1 asserts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from . import similarity as sim

NumericType = Literal["fp32", "fp16"]
NUMERIC_TYPES: tuple[NumericType, ...] = ("fp32", "fp16")

DTYPES: dict[str, np.dtype] = {
    "fp32": np.dtype(np.float32),
    "fp16": np.dtype(np.float16),
}
BIT_WIDTH: dict[str, int] = {"fp32": 32, "fp16": 16}


def check_numeric_type(numeric_type: str) -> NumericType:
    if numeric_type not in NUMERIC_TYPES:
        raise ValueError(f"unknown numeric type {numeric_type!r}; expected one of {NUMERIC_TYPES}")
    return numeric_type  # type: ignore[return-value]


@dataclass(frozen=True)
class VectorStore:
    """An immutable-by-convention view of D (or of a corrupted D')."""

    data: np.ndarray
    numeric_type: NumericType
    label: str = "D"

    def __post_init__(self) -> None:
        check_numeric_type(self.numeric_type)
        if self.data.ndim != 2:
            raise ValueError(f"vector store must be 2-D, got shape {self.data.shape}")
        if self.data.dtype != DTYPES[self.numeric_type]:
            raise ValueError(
                f"store declared {self.numeric_type} but array dtype is {self.data.dtype}"
            )
        object.__setattr__(self, "_f32_cache", None)
        object.__setattr__(self, "_norm_cache", None)

    # -- shape ---------------------------------------------------------------
    @property
    def n(self) -> int:
        return int(self.data.shape[0])

    @property
    def dim(self) -> int:
        return int(self.data.shape[1])

    @property
    def bit_width(self) -> int:
        return BIT_WIDTH[self.numeric_type]

    def __len__(self) -> int:
        return self.n

    # -- arithmetic views ----------------------------------------------------
    def as_f32(self) -> np.ndarray:
        cached = object.__getattribute__(self, "_f32_cache")
        if cached is None:
            cached = sim.as_f32(self.data)
            object.__setattr__(self, "_f32_cache", cached)
        return cached

    def norms(self) -> np.ndarray:
        cached = object.__getattribute__(self, "_norm_cache")
        if cached is None:
            cached = sim.norms(self.as_f32())
            object.__setattr__(self, "_norm_cache", cached)
        return cached

    def rows(self, ids) -> np.ndarray:
        return self.as_f32()[np.asarray(ids, dtype=np.int64)]

    def row(self, i: int) -> np.ndarray:
        return self.as_f32()[int(i)]

    # -- invariants ----------------------------------------------------------
    def all_finite(self) -> bool:
        return bool(np.isfinite(self.data.astype(np.float64, copy=False)).all())

    def assert_finite(self) -> None:
        if not self.all_finite():
            raise ValueError(f"vector store {self.label!r} contains non-finite values")

    def coordinate_range(self) -> tuple[float, float]:
        f = self.data.astype(np.float64, copy=False)
        return float(f.min()), float(f.max())

    # -- construction --------------------------------------------------------
    def with_data(self, data: np.ndarray, label: str) -> "VectorStore":
        """A sibling store with the same numeric type and new contents.

        Used by the stale/rebuilt conditions to stand in a corrupted D' where
        the clean D was, without ever mutating the clean array in place.
        """
        return VectorStore(
            data=np.ascontiguousarray(data, dtype=DTYPES[self.numeric_type]),
            numeric_type=self.numeric_type,
            label=label,
        )

    def copy(self, label: str | None = None) -> "VectorStore":
        return VectorStore(self.data.copy(), self.numeric_type, label or self.label)

    def content_hash(self) -> str:
        import hashlib

        h = hashlib.sha256()
        h.update(str(self.numeric_type).encode())
        h.update(str(self.data.shape).encode())
        h.update(np.ascontiguousarray(self.data).tobytes())
        return h.hexdigest()


def make_store(data: np.ndarray, numeric_type: str, label: str = "D") -> VectorStore:
    """Cast ``data`` into the declared storage type once, at the boundary."""
    nt = check_numeric_type(numeric_type)
    arr = np.ascontiguousarray(data, dtype=DTYPES[nt])
    return VectorStore(arr, nt, label)
