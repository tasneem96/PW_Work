"""Canonical index configuration.

Follows the common literature operating point for glove-100-angular:
efConstruction from ann-benchmarks, M from hnswlib's default and the middle of
the ann-benchmarks sweep. ef_search is pinned rather than swept, so any result
reported at this setting is one point on a curve and should say so.
"""

M = 16
EF_CONSTRUCTION = 500
EF_SEARCH = 64
K = 10
