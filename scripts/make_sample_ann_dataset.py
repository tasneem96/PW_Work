#!/usr/bin/env python3
"""Generate a small ANN-Benchmarks-shaped HDF5 file for offline testing.

Same four datasets and dtypes as ``glove-25-angular.hdf5``, just much smaller,
with exact cosine ground truth computed in float64.  Row norms deliberately
vary (like real GloVe vectors), so a test that forgets to normalize actually
fails instead of passing by accident.

    python scripts/make_sample_ann_dataset.py [out.hdf5]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

D = 25
N_TRAIN = 20_000
N_TEST = 200
N_NEIGHBORS = 100
SEED = 20260828
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "sample-25-angular.hdf5"


def make_vectors(rng, n):
    """Clustered directions with widely varying magnitudes, like word vectors."""
    centroids = rng.normal(0, 1.0, (40, D))
    directions = centroids[rng.integers(0, 40, n)] + rng.normal(0, 0.45, (n, D))
    norms = np.exp(rng.normal(0.0, 0.8, (n, 1)))  # magnitudes spread over ~0.1-10x
    return (directions * norms).astype(np.float32)


def main() -> int:
    import h5py

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    rng = np.random.default_rng(SEED)
    train = make_vectors(rng, N_TRAIN)
    test = make_vectors(rng, N_TEST)

    # Exact cosine ground truth, in float64 to keep the ordering unambiguous.
    a = np.asarray(train, dtype=np.float64)
    b = np.asarray(test, dtype=np.float64)
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    b /= np.linalg.norm(b, axis=1, keepdims=True)
    cosine = b @ a.T
    neighbors = np.argsort(-cosine, axis=1, kind="stable")[:, :N_NEIGHBORS]
    distances = 1.0 - np.take_along_axis(cosine, neighbors, axis=1)

    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, "w") as f:
        f.attrs["distance"] = "angular"
        f.attrs["point_type"] = "float"
        f.create_dataset("train", data=train, dtype="float32")
        f.create_dataset("test", data=test, dtype="float32")
        f.create_dataset("neighbors", data=neighbors.astype(np.int32), dtype="int32")
        f.create_dataset("distances", data=distances.astype(np.float32), dtype="float32")

    size_mb = out.stat().st_size / 1e6
    print(f"wrote {out} ({size_mb:.1f} MB)")
    print(f"  train {train.shape}  test {test.shape}  neighbors {neighbors.shape}")
    print("  distances stored as 1 - cosine")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
