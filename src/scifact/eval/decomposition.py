"""Separate retrieval failure from reasoning failure. The fix for OQ-001.

The project plan's Stage 2 said: add a classifier over retrieved evidence, and measure the two
failure modes independently. That is not possible from a single run. A wrong verdict is equally
consistent with "the evidence was never retrieved" and with "the evidence was retrieved and
misread" -- one signal, two hypotheses.

The fix is to vary one thing at a time. Three conditions, each removing one source of error:

| Condition | Evidence the verifier sees | Error it can still make |
|---|---|---|
| **A. oracle sentences** | the gold rationale sentences | reasoning only |
| **B. oracle abstract** | every sentence of the gold abstract | reasoning + sentence selection |
| **C. retrieved** | every sentence of BM25's top-k abstracts | reasoning + selection + retrieval |

The differences are the decomposition:

    A - B  = the cost of finding the right sentence inside the right abstract
    B - C  = the cost of finding the right abstract in 5,183

**An asymmetry worth stating rather than hiding.** Conditions A and B are defined only for
claims that *have* evidence. For a NOT_ENOUGH_INFO claim there is no gold rationale to hand
over: "perfect evidence" for it means "nothing in the corpus settles this", which is not a set
of sentences. Handing over its `cited_doc_ids` instead would leak the answer key (see
`scripts/demonstrate_leakage.py`).

So A and B are scored on SUPPORT/CONTRADICT claims only, and C is scored twice -- once on the
same subset for a like-for-like comparison, and once on all claims, which is the only number
that describes the real system.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from scifact.data.schema import Claim, Document
from scifact.eval.metrics import Estimate, bootstrap
from scifact.verify.labels import NEI, VERDICTS, Verdict, claim_label


class Verifier(Protocol):
    # Declared read-only so an implementation may satisfy it with either a plain class
    # attribute or a computed property. A bare `name: str` would require a mutable attribute
    # and reject the property form.
    @property
    def name(self) -> str: ...

    def predict(self, claim: str, sentences: Sequence[str]) -> Verdict: ...


class DocumentRetriever(Protocol):
    def rank(self, query: str, k: int | None = None) -> list[int]: ...


@dataclass(frozen=True)
class ConditionResult:
    condition: str
    accuracy: Estimate
    confusion: dict[tuple[Verdict, Verdict], int]  # (gold, predicted) -> count

    def per_class_recall(self) -> dict[Verdict, float]:
        out: dict[Verdict, float] = {}
        for gold in VERDICTS:
            total = sum(c for (g, _), c in self.confusion.items() if g == gold)
            correct = self.confusion.get((gold, gold), 0)
            out[gold] = correct / total if total else float("nan")
        return out


def _evaluate(
    verifier: Verifier,
    cases: Sequence[tuple[Claim, list[str]]],
    condition: str,
) -> ConditionResult:
    per_claim: list[float] = []
    confusion: dict[tuple[Verdict, Verdict], int] = {}

    for claim, sentences in cases:
        gold = claim_label(claim)
        predicted = verifier.predict(claim.claim, sentences)
        per_claim.append(float(predicted == gold))
        key = (gold, predicted)
        confusion[key] = confusion.get(key, 0) + 1

    return ConditionResult(condition, bootstrap(per_claim), confusion)


def condition_a_oracle_sentences(
    verifier: Verifier, claims: Sequence[Claim], corpus: dict[int, Document]
) -> ConditionResult:
    """Gold rationale sentences only. Isolates reasoning."""
    cases = []
    for claim in claims:
        if not claim.has_evidence:
            continue
        sentences = [
            corpus[doc_id].abstract[i]
            for doc_id, rationales in claim.evidence.items()
            if doc_id in corpus
            for r in rationales
            for i in r.sentences
            if i < len(corpus[doc_id].abstract)
        ]
        cases.append((claim, sentences))
    return _evaluate(verifier, cases, "A. oracle sentences")


def condition_b_oracle_abstract(
    verifier: Verifier, claims: Sequence[Claim], corpus: dict[int, Document]
) -> ConditionResult:
    """The whole gold abstract. Adds sentence-selection error."""
    cases = []
    for claim in claims:
        if not claim.has_evidence:
            continue
        sentences = [
            s for doc_id in claim.evidence if doc_id in corpus for s in corpus[doc_id].abstract
        ]
        cases.append((claim, sentences))
    return _evaluate(verifier, cases, "B. oracle abstract")


def condition_c_retrieved(
    verifier: Verifier,
    claims: Sequence[Claim],
    corpus: dict[int, Document],
    retriever: DocumentRetriever,
    k: int,
    evidence_only: bool,
) -> ConditionResult:
    """BM25's top-k abstracts. The full pipeline.

    `cited_doc_ids` is never consulted -- the retriever sees the claim text and nothing else.
    """
    cases = []
    for claim in claims:
        if evidence_only and not claim.has_evidence:
            continue
        sentences = [
            s
            for doc_id in retriever.rank(claim.claim, k=k)
            if doc_id in corpus
            for s in corpus[doc_id].abstract
        ]
        cases.append((claim, sentences))
    scope = "evidence-bearing only" if evidence_only else "all claims"
    return _evaluate(verifier, cases, f"C. retrieved@{k} ({scope})")


def print_decomposition(results: Sequence[ConditionResult]) -> None:
    print(f"  {'condition':<38} {'accuracy':>22}   {'n':>5}")
    print(f"  {'-' * 38} {'-' * 22}   {'-' * 5}")
    for r in results:
        print(f"  {r.condition:<38} {r.accuracy!s:>22}   {r.accuracy.n:>5}")

    print()
    print(f"  {'condition':<38} " + "  ".join(f"{v[:9]:>9}" for v in VERDICTS))
    print(f"  {'-' * 38} " + "  ".join("-" * 9 for _ in VERDICTS))
    for r in results:
        recalls = r.per_class_recall()
        cells = "  ".join(
            f"{recalls[v]:>8.1%}" if recalls[v] == recalls[v] else "      n/a" for v in VERDICTS
        )
        print(f"  {r.condition:<38} {cells}")

    print()
    print("  Second table is per-class recall: of the claims whose true verdict was X,")
    print("  what fraction were predicted X. A model can look fine on accuracy while")
    print(f"  scoring 0% on one class -- {NEI} is the one to watch.")
