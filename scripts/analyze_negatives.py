"""Test whether NOT_ENOUGH_INFO abstracts are lexically similar but topically off.

The hypothesis, from reading examples by hand: an abstract cited by a NOT_ENOUGH_INFO claim
shares keywords with the claim, yet discusses a different question — it neither supports nor
contradicts. If true, these are *hard negatives*: things a lexical retriever will rank highly
and be wrong about.

That distinction decides real design questions, so it needs measuring rather than believing.

Three conditions are compared:

1. **SUPPORTED/CONTRADICTED** — claim vs. the abstract that actually carries its evidence.
2. **NOT_ENOUGH_INFO** — claim vs. the abstract it cites, which carries no evidence.
3. **RANDOM** — claim vs. an unrelated abstract. The control.

If the hypothesis holds, condition 2 sits near condition 1 and far above condition 3.

A crude retrieval rank is also computed: how highly would the cited abstract rank among all
5,183 if you scored by content-word overlap alone? That is deliberately *not* BM25 — no IDF
weighting, no length normalisation — because the BM25 baseline belongs to Stage 1 and is gated
behind the evaluation protocol. This is a diagnostic, not a baseline, and must not be reported
as one.

Usage::

    python scripts/analyze_negatives.py
    python scripts/analyze_negatives.py --sample 300 --seed 0
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scifact.data.schema import Claim, Document, load_claims, load_corpus  # noqa: E402

DATA_DIR = REPO_ROOT / "data"

STOPWORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "without",
        "from",
        "by",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "has",
        "have",
        "had",
        "do",
        "does",
        "did",
        "not",
        "no",
        "nor",
        "so",
        "such",
        "it",
        "its",
        "they",
        "them",
        "we",
        "our",
        "you",
        "your",
        "he",
        "she",
        "his",
        "her",
        "which",
        "who",
        "whom",
        "what",
        "when",
        "where",
        "how",
        "all",
        "any",
        "both",
        "each",
        "more",
        "most",
        "other",
        "some",
        "only",
        "own",
        "same",
        "too",
        "very",
        "can",
        "will",
        "just",
        "should",
        "now",
        "also",
        "may",
        "might",
        "must",
        "been",
        "between",
        "during",
        "after",
        "before",
        "through",
        "under",
        "over",
        "into",
        "out",
        "up",
        "down",
    ]
)

WORD_RE = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 2)


def doc_words(doc: Document) -> frozenset[str]:
    return content_words(doc.title + " " + " ".join(doc.abstract))


def coverage(claim_words: frozenset[str], target: frozenset[str]) -> float:
    """Fraction of the claim's content words that appear in the target text."""
    return len(claim_words & target) / len(claim_words) if claim_words else 0.0


def summarise(name: str, values: list[float]) -> str:
    if not values:
        return f"  {name:<28} (no cases)"
    ordered = sorted(values)
    mean = sum(ordered) / len(ordered)

    def pct(p: int) -> float:
        return ordered[min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))]

    return (
        f"  {name:<28} n={len(values):>4}   mean={mean:.0%}   "
        f"p25={pct(25):.0%}  p50={pct(50):.0%}  p75={pct(75):.0%}"
    )


def rank_of_cited(
    claim_words: frozenset[str],
    cited_id: int,
    corpus_words: dict[int, frozenset[str]],
) -> int:
    """1-based rank of `cited_id` when every abstract is scored by raw overlap count.

    Deliberately crude. Not BM25, not a baseline, not reportable as retrieval performance.
    """
    target = len(claim_words & corpus_words[cited_id])
    better = sum(1 for wid, w in corpus_words.items() if len(claim_words & w) > target)
    return better + 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "dev"])
    parser.add_argument("--sample", type=int, default=250, help="claims per group for ranking")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    corpus = load_corpus(DATA_DIR / "corpus.jsonl")
    claims = load_claims(DATA_DIR / f"claims_{args.split}.jsonl")
    # Seeded deliberately: the sampling must be reproducible so a reported number can be
    # re-derived. Not cryptographic, and not required to be.
    rng = random.Random(args.seed)  # noqa: S311

    print("precomputing content words for 5,183 abstracts...", flush=True)
    corpus_words = {doc_id: doc_words(doc) for doc_id, doc in corpus.items()}
    all_ids = list(corpus_words)

    # ------------------------------------------------- condition 1/2/3 overlap
    evidence_cov: list[float] = []
    nei_cov: list[float] = []
    random_cov: list[float] = []

    nei_claims: list[tuple[Claim, int]] = []
    pos_claims: list[tuple[Claim, int]] = []

    nei_without_citation = 0

    for claim in claims:
        cw = content_words(claim.claim)
        if not cw:
            continue

        if claim.has_evidence:
            for doc_id in claim.evidence:
                if doc_id in corpus_words:
                    evidence_cov.append(coverage(cw, corpus_words[doc_id]))
                    pos_claims.append((claim, doc_id))
        else:
            if not claim.cited_doc_ids:
                nei_without_citation += 1
            for doc_id in claim.cited_doc_ids:
                if doc_id in corpus_words:
                    nei_cov.append(coverage(cw, corpus_words[doc_id]))
                    nei_claims.append((claim, doc_id))

        random_cov.append(coverage(cw, corpus_words[rng.choice(all_ids)]))

    print()
    print("=" * 78)
    print("HYPOTHESIS: NOT_ENOUGH_INFO abstracts share the claim's vocabulary")
    print("            but do not address the claim")
    print("=" * 78)
    print("How much of the claim's vocabulary appears in the abstract?")
    print()
    print(summarise("1. has evidence", evidence_cov))
    print(summarise("2. NOT_ENOUGH_INFO (cited)", nei_cov))
    print(summarise("3. random abstract", random_cov))
    print()

    if evidence_cov and nei_cov and random_cov:
        e = sum(evidence_cov) / len(evidence_cov)
        n = sum(nei_cov) / len(nei_cov)
        r = sum(random_cov) / len(random_cov)
        span = e - r
        position = (n - r) / span if span else 0.0
        print(f"  NOT_ENOUGH_INFO sits {position:.0%} of the way from RANDOM to EVIDENCE.")
        print("  (near 100% = lexically indistinguishable from genuine evidence)")

    if nei_without_citation:
        print(f"\n  note: {nei_without_citation} NEI claims cite no document at all")

    # ------------------------------------------------- crude retrieval rank
    print()
    print("=" * 78)
    print("CRUDE RETRIEVAL RANK  (overlap count only -- NOT BM25, NOT a baseline)")
    print("=" * 78)
    print("Where does the cited abstract land among all 5,183 when ranked by")
    print("raw content-word overlap with the claim?")
    print()

    for label, pairs in (("has evidence", pos_claims), ("NOT_ENOUGH_INFO", nei_claims)):
        chosen = rng.sample(pairs, min(args.sample, len(pairs)))
        ranks = [
            rank_of_cited(content_words(c.claim), doc_id, corpus_words) for c, doc_id in chosen
        ]
        ranks.sort()
        top1 = sum(1 for r in ranks if r == 1)
        top10 = sum(1 for r in ranks if r <= 10)
        top100 = sum(1 for r in ranks if r <= 100)
        median = ranks[len(ranks) // 2]
        print(
            f"  {label:<18} n={len(ranks):>4}   median rank={median:>5,}   "
            f"top-1={top1 / len(ranks):>5.1%}  top-10={top10 / len(ranks):>5.1%}  "
            f"top-100={top100 / len(ranks):>5.1%}"
        )

    print()
    print("If the two rows above look alike, a lexical retriever cannot separate")
    print("'relevant' from 'actually answers the claim'. That is a reasoning problem,")
    print("not a retrieval problem, and no amount of retrieval tuning fixes it.")

    # ------------------------------------------------- separability
    # How well could a SINGLE threshold on lexical coverage separate the two groups?
    # This is a diagnostic of how much the distributions overlap. It is NOT a proposed
    # abstention rule: choosing an operating point is Mounir's decision and belongs in
    # docs/EVALUATION.md.
    print()
    print("=" * 78)
    print("SEPARABILITY  (how much do the two distributions overlap?)")
    print("=" * 78)

    labelled = [(c, 1) for c in evidence_cov] + [(c, 0) for c in nei_cov]
    n_pos = len(evidence_cov)
    n_neg = len(nei_cov)

    best_acc, best_t = 0.0, 0.0
    for step in range(0, 101):
        t = step / 100
        tp = sum(1 for v, y in labelled if y == 1 and v >= t)
        tn = sum(1 for v, y in labelled if y == 0 and v < t)
        acc = (tp + tn) / len(labelled)
        if acc > best_acc:
            best_acc, best_t = acc, t

    # AUC by the Mann-Whitney U identity: P(random positive scores above random negative).
    concordant = sum(1 for p in evidence_cov for n in nei_cov if p > n)
    ties = sum(1 for p in evidence_cov for n in nei_cov if p == n)
    auc = (concordant + 0.5 * ties) / (n_pos * n_neg)

    print(f"  best single-threshold accuracy   {best_acc:.1%}  (at coverage >= {best_t:.0%})")
    print(f"  AUC                              {auc:.3f}")
    print(f"  majority-class floor             {max(n_pos, n_neg) / (n_pos + n_neg):.1%}")
    print()
    print("  AUC is the probability that a randomly chosen evidence-bearing abstract")
    print("  scores above a randomly chosen NOT_ENOUGH_INFO one. 0.5 = coin flip,")
    print("  1.0 = perfectly separable. Anything in between means every threshold you")
    print("  could pick trades false accepts against false abstentions -- which is")
    print("  exactly the Stage 5 calibration problem, visible here before any model exists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
