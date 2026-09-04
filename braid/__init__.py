"""BRAID: localized gray-box bit-flip stress testing of HNSW vector search.

Phase 0 (protocol freeze) and Phase 1 (baseline system and instrumentation)
of the internal project phasing are implemented here. Later phases (exact
bit-flip engine, geometry/route objectives, targeted and work-amplification
attacks) are deliberately absent; nothing in this package performs a bit flip.
"""

__version__ = "0.1.0"

PHASES_IMPLEMENTED = (0, 1)
