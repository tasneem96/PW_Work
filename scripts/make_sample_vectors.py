#!/usr/bin/env python3
"""Generate the synthetic fixture used by the tests and the offline demo.

These are NOT GloVe vectors.  They are deterministic pseudo-random vectors
arranged into topical clusters so that nearest-neighbour retrieval has a
verifiable right answer without a 171 MB download.  Real usage should point
``--vectors`` at ``glove.6B.50d.txt`` (see scripts/download_glove.sh).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DIM = 50
SEED = 20260826

CLUSTERS = {
    "royalty": "king queen prince princess monarch throne crown royal kingdom emperor",
    "animals": "dog cat horse cow sheep wolf fox rabbit mouse tiger",
    "food": "bread cheese coffee tea sugar rice pasta soup apple bacon",
    "tech": "computer software network server database algorithm code laptop internet processor",
    "weather": "rain snow storm cloud wind sunshine fog thunder frost drizzle",
    "family": "mother father sister brother daughter son uncle aunt cousin grandmother",
    "transport": "car train bus bicycle airplane ship truck subway ferry motorcycle",
    "music": "guitar piano violin drum melody rhythm concert album singer orchestra",
}


def main() -> None:
    rng = np.random.default_rng(SEED)
    out = Path(__file__).resolve().parent.parent / "data" / "sample.synthetic.50d.txt"
    lines = []
    for words in CLUSTERS.values():
        centroid = rng.normal(0.0, 1.0, DIM)
        for word in words.split():
            # Tight enough that same-cluster words are each other's neighbours.
            vector = centroid + rng.normal(0.0, 0.35, DIM)
            lines.append(word + " " + " ".join(f"{v:.5f}" for v in vector))
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} vectors ({DIM}d) to {out}")


if __name__ == "__main__":
    main()
