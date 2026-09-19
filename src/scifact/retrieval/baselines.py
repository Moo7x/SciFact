"""Baselines that exist to absorb credit BM25 should not get.

A baseline is not a weak competitor. It is a **control**: it answers "how much of this score
comes from something other than the method?" If BM25 cannot beat a retriever that ignores the
claim entirely, then whatever BM25 is doing, it is not matching claims to evidence.

Three controls, in increasing order of embarrassment if BM25 loses:

- `RandomRetriever` -- the floor. Beating this proves only that the code runs.
- `PopularityRetriever` -- always returns the same abstracts regardless of the claim. Catches a
  corpus where a few documents happen to be plausible answers to everything.
- `PositionalSentenceRanker` -- for sentence-level retrieval only. Measured on train, ~80% of
  gold evidence sits in the back half of its abstract and 21.5% in the final tenth. A ranker
  exploiting only that prior does no retrieval at all, so any sentence-level score must be read
  against it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence


class RandomRetriever:
    """Uniformly random ranking. The floor."""

    def __init__(self, doc_ids: Sequence[int], seed: int = 0) -> None:
        self.doc_ids = list(doc_ids)
        self.seed = seed

    def rank(self, query: str, k: int | None = None) -> list[int]:
        # Seeded per query so the same claim always gets the same ranking: a baseline that
        # changes between runs cannot be compared against.
        rng = random.Random(f"{self.seed}:{query}")  # noqa: S311
        ids = self.doc_ids[:]
        rng.shuffle(ids)
        return ids[:k] if k is not None else ids


class PopularityRetriever:
    """Returns the longest abstracts, ignoring the claim entirely.

    Length is a stand-in for any claim-independent prior. If this scores well, the metric is
    measuring a property of the corpus rather than of retrieval.
    """

    def __init__(self, doc_ids: Sequence[int], lengths: Sequence[int]) -> None:
        self.order = [
            doc_id
            for doc_id, _ in sorted(zip(doc_ids, lengths, strict=True), key=lambda p: (-p[1], p[0]))
        ]

    def rank(self, query: str, k: int | None = None) -> list[int]:
        return self.order[:k] if k is not None else list(self.order)


class PositionalSentenceRanker:
    """Ranks sentences last-first, ignoring the claim. The control for OQ-004."""

    def rank_sentences(self, query: str, n_sentences: int, k: int | None = None) -> list[int]:
        order = list(range(n_sentences - 1, -1, -1))
        return order[:k] if k is not None else order
