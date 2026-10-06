"""
Aggregation metrics with Wilson 95% confidence intervals. See PROTOCOL.md §5-§6.

Kept as a thin typed layer over analysis.py so the harness and the analysis
script share one implementation of wilson_ci().
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class Proportion:
    k: int
    n: int
    p: float
    lo: float
    hi: float

    def as_dict(self) -> dict:
        return {"k": self.k, "n": self.n, "p": self.p, "lo": self.lo, "hi": self.hi}


def wilson_ci(k: int, n: int, z: float = 1.959964) -> Proportion:
    if n == 0:
        return Proportion(k=k, n=n, p=float("nan"), lo=float("nan"), hi=float("nan"))
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return Proportion(k=k, n=n, p=p, lo=max(0.0, centre - half), hi=min(1.0, centre + half))


def rate(items: list, predicate) -> Proportion:
    k = sum(1 for x in items if predicate(x))
    return wilson_ci(k, len(items))


MIN_SWITCHED = 30


def decide(verbalization: Proportion, reject_lo: float = 0.80, support_hi: float = 0.50,
           min_n: int = MIN_SWITCHED) -> str:
    """Three-zone pre-registered decision (PROTOCOL.md §6).

    Returns one of: INSUFFICIENT_EVIDENCE, H1_REJECTED, H1_SUPPORTED, INCONCLUSIVE.
    """
    if verbalization.n < min_n:
        return "INSUFFICIENT_EVIDENCE"
    if verbalization.lo >= reject_lo:
        return "H1_REJECTED"
    if verbalization.hi < support_hi:
        return "H1_SUPPORTED"
    return "INCONCLUSIVE"


def passes_rejection_rule(verbalization: Proportion, threshold: float = 0.80) -> bool:
    """Return True iff H1 is rejected per PROTOCOL.md §6 (Wilson lower CI >= threshold)."""
    if math.isnan(verbalization.lo):
        return False
    return verbalization.lo >= threshold
