"""Measure structural properties of the SciFact corpus and claims.

This exists because impressions from reading ten examples must be checked against all of them
before they shape a design decision. Ten examples generate hypotheses; they do not confirm them.

Everything here is a *structural* measurement — lengths, positions, lexical overlap. None of it
is an evaluation metric, and it reports on the training split by default.

Usage::

    python scripts/describe_data.py
    python scripts/describe_data.py --split dev
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scifact.data.schema import load_claims, load_corpus  # noqa: E402

DATA_DIR = REPO_ROOT / "data"

# A deliberately small, visible stopword list. A library list would hide the choice; the only
# job here is to stop "the/of/and" from dominating a lexical-overlap number.
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
    ]
)

WORD_RE = re.compile(r"[a-z0-9]+")


def content_words(text: str) -> set[str]:
    return {w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 2}


def percentiles(values: list[int], points: tuple[int, ...] = (50, 75, 90, 95, 99)) -> str:
    if not values:
        return "n/a"
    ordered = sorted(values)
    parts = []
    for p in points:
        idx = min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1)))
        parts.append(f"p{p}={ordered[idx]}")
    return "  ".join(parts)


def bar(fraction: float, width: int = 40) -> str:
    return "#" * round(fraction * width) + "." * (width - round(fraction * width))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "dev"])
    args = parser.parse_args()

    corpus = load_corpus(DATA_DIR / "corpus.jsonl")
    claims = load_claims(DATA_DIR / f"claims_{args.split}.jsonl")

    # ------------------------------------------------------------------ abstracts
    sent_counts = [len(d) for d in corpus.values()]
    word_counts = [len(" ".join(d.abstract).split()) for d in corpus.values()]
    structured = sum(1 for d in corpus.values() if d.structured)

    print("=" * 74)
    print(f"ABSTRACTS  (n={len(corpus):,})")
    print("=" * 74)
    print(f"  sentences per abstract   {percentiles(sent_counts)}")
    print(f"  words per abstract       {percentiles(word_counts)}")
    print(f"  longest abstract         {max(word_counts):,} words")
    print(f"  marked 'structured'      {structured:,} ({structured / len(corpus):.1%})")

    # ------------------------------------------------------------------ evidence
    positions: list[float] = []
    is_final: list[bool] = []
    set_sizes: list[int] = []
    contiguous = 0
    total = 0
    labels: Counter[str] = Counter()
    overlaps: list[float] = []
    claim_lengths: list[int] = []
    no_evidence = 0

    for claim in claims:
        claim_lengths.append(len(claim.claim.split()))
        if not claim.has_evidence:
            no_evidence += 1
        cw = content_words(claim.claim)
        for doc_id, rationales in claim.evidence.items():
            doc = corpus.get(doc_id)
            if doc is None:
                continue
            n = len(doc)
            for r in rationales:
                total += 1
                labels[r.label] += 1
                set_sizes.append(len(r.sentences))
                contiguous += r.is_contiguous
                for i in r.sentences:
                    positions.append(i / (n - 1) if n > 1 else 1.0)
                is_final.append(max(r.sentences) == n - 1)
                ew = content_words(" ".join(doc.abstract[i] for i in r.sentences))
                if cw:
                    overlaps.append(len(cw & ew) / len(cw))

    print()
    print("=" * 74)
    print(f"CLAIMS AND EVIDENCE  ({args.split}: {len(claims):,} claims, {total:,} rationales)")
    print("=" * 74)
    print(
        f"  claims with NO evidence  {no_evidence:,} ({no_evidence / len(claims):.1%})"
        "   <- the abstention cases"
    )
    print(f"  rationale labels         {dict(labels)}")
    print(f"  sentences per rationale  {percentiles(set_sizes, (50, 75, 90, 99))}")
    print(f"  contiguous indices       {contiguous:,} / {total:,} ({contiguous / total:.1%})")
    print(
        f"  includes FINAL sentence  {sum(is_final):,} / {len(is_final):,} "
        f"({sum(is_final) / len(is_final):.1%})"
    )

    print()
    print("  Where evidence sits in the abstract (0.0 = first sentence, 1.0 = last):")
    buckets = [0] * 10
    for p in positions:
        buckets[min(9, int(p * 10))] += 1
    peak = max(buckets)
    for i, count in enumerate(buckets):
        print(
            f"    {i / 10:.1f}-{(i + 1) / 10:.1f}  {bar(count / peak)} {count:>5,}"
            f"  ({count / len(positions):.1%})"
        )

    # ------------------------------------------------------------------ lexical
    print()
    print("=" * 74)
    print("CLAIM / EVIDENCE LEXICAL OVERLAP   (this is roughly BM25's ceiling)")
    print("=" * 74)
    print(f"  words per claim          {percentiles(claim_lengths, (50, 90, 99))}")
    print("  % of the claim's content words that also appear in its gold evidence:")
    print(f"    {percentiles([round(o * 100) for o in overlaps], (10, 25, 50, 75, 90))}")
    zero = sum(1 for o in overlaps if o == 0.0)
    print(
        f"  rationales sharing ZERO content words with their claim: "
        f"{zero:,} / {len(overlaps):,} ({zero / len(overlaps):.1%})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
