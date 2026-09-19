"""BM25 over the abstract corpus. Written out rather than imported, so it can be inspected.

BM25 scores a document D against a query Q as

    score(D, Q) = sum over q in Q of   IDF(q) * ( f(q,D) * (k1 + 1) )
                                       ---------------------------------------
                                       ( f(q,D) + k1 * (1 - b + b * |D|/avgdl) )

Three ideas, each fixing a failure of the one before it:

1. **Term frequency.** A document mentioning "melanoma" six times is probably more about
   melanoma than one mentioning it once. Raw counts alone fail immediately: they say a document
   with "the" 50 times is extremely about "the".

2. **IDF** fixes that. A term appearing in nearly every document carries almost no information;
   a term in three documents is nearly an identifier. Here

       IDF(q) = ln( (N - n(q) + 0.5) / (n(q) + 0.5) + 1 )

   with N documents and n(q) containing q. The +1 inside the log keeps IDF positive even for a
   term in every document -- without it, ubiquitous terms go negative and *penalise* a match.

3. **Saturation and length normalisation** fix what remains. Raw TF is linear, so the 20th
   mention adds as much as the 2nd, which is false: the 2nd mention tells you a lot, the 20th
   almost nothing. The `f/(f + k1*...)` form saturates hyperbolically towards `k1+1`.

   `k1` sets how fast: low k1 means one occurrence is nearly as good as many. `b` controls
   length normalisation between none (b=0) and full (b=1). Long documents win on raw counts
   simply by being long, and b discounts that.

Defaults k1=1.2, b=0.75 are the Robertson/Okapi values. They are a starting point to tune on
`train_tune`, not a law.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

TOKEN_RE = re.compile(r"[a-z0-9]+")

DEFAULT_K1 = 1.2
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens.

    No stemming and no stopword list. Both are choices that change results, so they belong in a
    tuning experiment where their effect is measured, not baked into the baseline where they
    would silently take credit.
    """
    return TOKEN_RE.findall(text.lower())


class BM25:
    """An in-memory BM25 index over a fixed document collection."""

    def __init__(
        self,
        doc_ids: list[int],
        documents: list[str],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids and documents must be the same length")

        self.k1 = k1
        self.b = b
        self.doc_ids = doc_ids

        tokenized = [tokenize(d) for d in documents]
        self.doc_len = [len(t) for t in tokenized]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        n_docs = len(tokenized)

        # Inverted index: term -> list of (document position, term frequency).
        # Built so scoring touches only documents containing a query term, rather than all
        # 5,183 -- the difference between a scan and a lookup.
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for pos, tokens in enumerate(tokenized):
            for term, freq in Counter(tokens).items():
                self.postings.setdefault(term, []).append((pos, freq))

        self.idf: dict[str, float] = {}
        for term, posting in self.postings.items():
            n_q = len(posting)
            self.idf[term] = math.log((n_docs - n_q + 0.5) / (n_q + 0.5) + 1.0)

    def score_all(self, query: str) -> list[float]:
        scores = [0.0] * len(self.doc_ids)
        for term in tokenize(query):
            posting = self.postings.get(term)
            if posting is None:
                continue
            idf = self.idf[term]
            for pos, freq in posting:
                norm = 1.0 - self.b + self.b * (self.doc_len[pos] / self.avgdl)
                scores[pos] += idf * (freq * (self.k1 + 1.0)) / (freq + self.k1 * norm)
        return scores

    def rank(self, query: str, k: int | None = None) -> list[int]:
        """Return doc_ids best-first.

        Ties broken by doc_id so the ranking is deterministic. Without that, two runs of the
        same code can disagree and the disagreement looks like a real effect.
        """
        scores = self.score_all(query)
        order = sorted(range(len(scores)), key=lambda pos: (-scores[pos], self.doc_ids[pos]))
        ids = [self.doc_ids[pos] for pos in order]
        return ids[:k] if k is not None else ids


def build_index(
    doc_ids: Iterable[int],
    texts: Iterable[str],
    k1: float = DEFAULT_K1,
    b: float = DEFAULT_B,
) -> BM25:
    return BM25(list(doc_ids), list(texts), k1=k1, b=b)
