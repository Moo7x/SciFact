"""Retrieval metrics, with uncertainty attached.

Only recall-style metrics appear here, and that is a deliberate restriction. Precision is not
computable on SciFact: an unannotated retrieved abstract is not a labelled negative, it is
merely unlabelled. Counting it as wrong would punish a retriever for surfacing something that
may well be relevant. The project plan names this trap explicitly.

Every metric ships with a bootstrap confidence interval. A point estimate on 300 claims that
does not carry its interval invites exactly the comparison it cannot support.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

BOOTSTRAP_SEED = 12345
BOOTSTRAP_RESAMPLES = 2000


@dataclass(frozen=True)
class Estimate:
    """A point estimate with a percentile bootstrap interval."""

    value: float
    low: float
    high: float
    n: int

    def __str__(self) -> str:
        return f"{self.value:6.1%} [{self.low:.1%}, {self.high:.1%}]"

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2


def bootstrap(
    per_claim: Sequence[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    alpha: float = 0.05,
) -> Estimate:
    """Percentile bootstrap over per-claim scores.

    Resampling claims (not individual judgements) is the right unit: claims are what is
    independently sampled from the population, and two judgements about the same claim are not
    independent of each other.
    """
    n = len(per_claim)
    if n == 0:
        return Estimate(0.0, 0.0, 0.0, 0)

    point = sum(per_claim) / n
    rng = random.Random(seed)  # noqa: S311
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += per_claim[rng.randrange(n)]
        means.append(total / n)
    means.sort()

    lo = means[int(alpha / 2 * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return Estimate(point, lo, hi, n)


def hit_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    """1.0 if any gold item appears in the top k. The retrieval question for Stage 2.

    Stage 2 asks whether the evidence reached the classifier at all. One gold abstract in the
    candidate set answers that; requiring all of them answers a different question.
    """
    return float(any(item in gold for item in ranked[:k]))


def recall_at_k(ranked: Sequence[int], gold: set[int], k: int) -> float:
    """Fraction of gold items appearing in the top k."""
    if not gold:
        return 0.0
    return sum(1 for item in ranked[:k] if item in gold) / len(gold)


def reciprocal_rank(ranked: Sequence[int], gold: set[int]) -> float:
    """1/rank of the first gold item, or 0 if none is ranked.

    Sensitive to where the first correct answer lands, unlike hit@k which only asks whether it
    landed inside the window. Useful because the cost of a deep rank is real: Stage 4's agent
    pays per document it reads.
    """
    for i, item in enumerate(ranked, start=1):
        if item in gold:
            return 1.0 / i
    return 0.0
