"""Simple fixed verifiers.

These are not meant to be good. They exist so the failure-decomposition machinery produces real
numbers before a learned model is introduced, and so that whatever Stage 3 trains has a floor it
must clear to have demonstrated anything.

`MajorityVerifier` is the floor: 41.0% of train is SUPPORT, so a model scoring 41% has learned
nothing at all.

`LexicalVerifier` is a genuine attempt with word-level features only. Its point is not accuracy
but *where it breaks*: it can tell relevance from irrelevance reasonably well, and cannot tell
agreement from disagreement, because that distinction is not in the vocabulary. Seeing that fail
is the argument for a cross-encoder in Stage 3.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from scifact.verify.labels import CONTRADICT, NEI, SUPPORT, Verdict

TOKEN_RE = re.compile(r"[a-z0-9]+")

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
        "so",
        "such",
        "it",
        "its",
        "they",
        "them",
        "we",
        "our",
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
    ]
)

# Cues that flip or weaken a finding. Crude by design: the whole point is to show that polarity
# in scientific prose does not reduce to a keyword list.
NEGATION_CUES = frozenset(
    [
        "no",
        "not",
        "none",
        "never",
        "neither",
        "nor",
        "without",
        "lack",
        "lacks",
        "lacking",
        "absence",
        "absent",
        "fail",
        "fails",
        "failed",
        "failure",
        "unable",
        "cannot",
        "did",
        "n't",
        "unchanged",
        "unaffected",
        "no-effect",
        "ineffective",
        "insufficient",
        "decrease",
        "decreased",
        "decreases",
        "decreasing",
        "reduce",
        "reduced",
        "reduces",
        "reducing",
        "lower",
        "lowers",
        "lowered",
        "inhibit",
        "inhibits",
        "inhibited",
        "inhibition",
        "suppress",
        "suppresses",
        "suppressed",
        "suppression",
        "loss",
        "lost",
        "prevent",
        "prevents",
        "prevented",
        "worse",
        "worsened",
        "attenuate",
        "attenuated",
        "attenuates",
        "impair",
        "impaired",
        "unlikely",
        "rarely",
        "rare",
        "negative",
    ]
)

POSITIVE_CUES = frozenset(
    [
        "increase",
        "increased",
        "increases",
        "increasing",
        "higher",
        "raise",
        "raised",
        "elevates",
        "elevated",
        "elevation",
        "enhance",
        "enhanced",
        "enhances",
        "enhancement",
        "improve",
        "improved",
        "improves",
        "improvement",
        "promote",
        "promotes",
        "promoted",
        "induce",
        "induces",
        "induced",
        "upregulate",
        "upregulated",
        "greater",
        "more",
        "associated",
        "correlated",
        "effective",
        "efficacy",
        "significant",
        "positive",
    ]
)


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def _content(text: str) -> set[str]:
    return {t for t in _tokens(text) if t not in STOPWORDS and len(t) > 2}


class MajorityVerifier:
    """Always predicts the training majority class. The floor."""

    name = "majority (always SUPPORT)"

    def predict(self, claim: str, sentences: Sequence[str]) -> Verdict:
        return SUPPORT


class LexicalVerifier:
    """Overlap decides relevance; polarity-cue agreement decides direction.

    `nei_threshold` is the fraction of the claim's content words that must appear in some
    candidate sentence before the verifier is willing to commit to a direction at all. Tuned on
    train, never on dev.
    """

    def __init__(self, nei_threshold: float = 0.45) -> None:
        self.nei_threshold = nei_threshold

    @property
    def name(self) -> str:
        return f"lexical (nei_threshold={self.nei_threshold:.2f})"

    def predict(self, claim: str, sentences: Sequence[str]) -> Verdict:
        claim_words = _content(claim)
        if not claim_words or not sentences:
            return NEI

        best_sentence = ""
        best_overlap = 0.0
        for sentence in sentences:
            overlap = len(claim_words & _content(sentence)) / len(claim_words)
            if overlap > best_overlap:
                best_overlap, best_sentence = overlap, sentence

        if best_overlap < self.nei_threshold:
            return NEI

        # Direction: does the evidence carry the same polarity as the claim?
        claim_tokens, evidence_tokens = set(_tokens(claim)), set(_tokens(best_sentence))
        claim_neg = bool(claim_tokens & NEGATION_CUES)
        evidence_neg = bool(evidence_tokens & NEGATION_CUES)
        claim_pos = bool(claim_tokens & POSITIVE_CUES)
        evidence_pos = bool(evidence_tokens & POSITIVE_CUES)

        # Disagreement in either direction reads as contradiction.
        if (claim_pos and evidence_neg and not evidence_pos) or (
            claim_neg and evidence_pos and not evidence_neg
        ):
            return CONTRADICT
        return SUPPORT
