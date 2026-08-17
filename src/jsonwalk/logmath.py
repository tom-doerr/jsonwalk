"""Tiny log-space helpers.

Kept dependency-free so the search and scoring logic can be unit-tested
without importing torch.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

NEG_INF = float("-inf")


def logsumexp(values: Iterable[float]) -> float:
    """Numerically stable log(sum(exp(v))). Empty input -> -inf."""
    vals = [v for v in values if v != NEG_INF]
    if not vals:
        return NEG_INF
    top = max(vals)
    if top == float("inf"):
        return top
    return top + math.log(sum(math.exp(v - top) for v in vals))


def sigmoid(x: float) -> float:
    """Logistic function, overflow-safe for large |x|."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
