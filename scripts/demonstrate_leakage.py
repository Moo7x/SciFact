"""Demonstrate, on the real data, the two ways this project's numbers can be faked.

Not a metric and not a baseline. This is a teaching artifact: it makes two abstract failures
concrete enough to recognise, using SciFact itself rather than a toy example.

**Demo 1 — annotation leakage.** ``cited_doc_ids`` looks like harmless input metadata. It is an
annotation artifact: the annotator recorded which abstracts they consulted, and the evidence is
inside them. Handing that field to a retriever hands it the answer, and the resulting number is
not wrong so much as meaningless.

**Demo 2 — selection bias from repeated evaluation.** Every choice made by looking at a split
spends that split. Trying many variants and reporting the winner's score inflates it even when
no variant is better than any other, because the maximum of several noisy estimates is biased
upward. This simulates it rather than arguing it.

Usage::

    python scripts/demonstrate_leakage.py
"""

from __future__ import annotations

import math
import random
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scifact.data.schema import Document, load_claims, load_corpus  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
WORD_RE = re.compile(r"[a-z0-9]+")

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
        "which",
        "who",
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


def content_words(text: str) -> frozenset[str]:
    return frozenset(w for w in WORD_RE.findall(text.lower()) if w not in STOPWORDS and len(w) > 2)


def doc_words(doc: Document) -> frozenset[str]:
    return content_words(doc.title + " " + " ".join(doc.abstract))


def demo_annotation_leakage(seed: int = 0, sample: int = 200) -> None:
    corpus = load_corpus(DATA_DIR / "corpus.jsonl")
    claims = load_claims(DATA_DIR / "claims_train.jsonl")
    corpus_words = {i: doc_words(d) for i, d in corpus.items()}
    rng = random.Random(seed)  # noqa: S311

    with_ev = [c for c in claims if c.has_evidence]

    # Is the evidence-bearing document always among cited_doc_ids?
    inside = sum(1 for c in with_ev for doc_id in c.evidence if doc_id in c.cited_doc_ids)
    total_ev_docs = sum(len(c.evidence) for c in with_ev)
    cited_sizes = [len(c.cited_doc_ids) for c in with_ev]

    print("=" * 78)
    print("DEMO 1 - ANNOTATION LEAKAGE:  what cited_doc_ids actually contains")
    print("=" * 78)
    print(
        f"  evidence-bearing docs that are also in cited_doc_ids: "
        f"{inside}/{total_ev_docs} ({inside / total_ev_docs:.1%})"
    )
    print(
        f"  documents cited per claim: mean={sum(cited_sizes) / len(cited_sizes):.2f}  "
        f"min={min(cited_sizes)}  max={max(cited_sizes)}"
    )

    chosen = rng.sample(with_ev, min(sample, len(with_ev)))
    honest_hits = 0
    leaked_hits = 0

    for claim in chosen:
        cw = content_words(claim.claim)
        gold = set(claim.evidence)

        # Honest: rank all 5,183 abstracts.
        best_id = max(corpus_words, key=lambda i: len(cw & corpus_words[i]))
        honest_hits += best_id in gold

        # Leaked: rank only the abstracts the annotator recorded consulting.
        pool = [i for i in claim.cited_doc_ids if i in corpus_words] or list(corpus_words)
        best_leaked = max(pool, key=lambda i: len(cw & corpus_words[i]))
        leaked_hits += best_leaked in gold

    n = len(chosen)
    print()
    print(f"  Top-1 document accuracy, identical scoring function, n={n}:")
    print(f"    search all 5,183 abstracts        {honest_hits / n:>6.1%}")
    print(f"    search only cited_doc_ids         {leaked_hits / n:>6.1%}   <- leaked")
    print(f"    difference                        {(leaked_hits - honest_hits) / n:>+6.1%}")
    print()
    print("  The second number is not a better retriever. It is not a retriever at all:")
    print("  the candidate set was built from the answer key. Nothing about the method")
    print("  changed between those two rows.")


def demo_selection_bias(
    n_eval: int = 300, true_p: float = 0.60, trials: int = 4000, seed: int = 0
) -> None:
    """Simulate picking the best of m identical models on a finite evaluation set."""
    rng = random.Random(seed)  # noqa: S311
    sigma = math.sqrt(true_p * (1 - true_p) / n_eval)

    print()
    print("=" * 78)
    print("DEMO 2 - SELECTION BIAS:  the cost of evaluating many variants on one split")
    print("=" * 78)
    print(f"  Setup: {trials:,} simulations. Every 'model' has the SAME true recall of")
    print(f"  {true_p:.0%}. None is better than any other. Each is scored on {n_eval} claims.")
    print(f"  Standard error of a single estimate: {sigma:.4f} = {sigma:.1%}")
    print()
    print(f"  {'variants tried':>14}  {'best observed':>13}  {'inflation':>10}  {'theory':>8}")
    print(f"  {'-' * 14}  {'-' * 13}  {'-' * 10}  {'-' * 8}")

    for m in (1, 2, 5, 10, 20, 50, 100):
        peaks = []
        for _ in range(trials):
            peaks.append(
                max(sum(rng.random() < true_p for _ in range(n_eval)) / n_eval for _ in range(m))
            )
        observed = sum(peaks) / len(peaks)
        theory = sigma * math.sqrt(2 * math.log(m)) if m > 1 else 0.0
        print(f"  {m:>14}  {observed:>12.1%}  {observed - true_p:>+9.1%}  {theory:>+7.1%}")

    print()
    print("  'Theory' is the expected maximum of m independent normal deviates,")
    print("  sigma * sqrt(2 * ln m) -- an upper-bound approximation, so it runs high")
    print("  for small m and tracks closely as m grows.")
    print()
    print("  Read the last column as: this much of your winner's margin is noise you")
    print("  selected for. It is not a bug in the models. It is what taking a maximum does.")


def main() -> int:
    demo_annotation_leakage()
    demo_selection_bias()
    print()
    print("=" * 78)
    print("Both failures produce numbers that look good and mean nothing, and neither")
    print("raises an error. The only defence is a protocol written down in advance.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
