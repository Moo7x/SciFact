"""Run a retriever over a claim set and report metrics with intervals.

Two guards are built in rather than left to discipline:

* `cited_doc_ids` is never read. It is an annotation artifact and using it leaks the answer
  (demonstrated in `scripts/demonstrate_leakage.py`: 62.5% -> 99.5% with no change of method).
  The harness does not accept it as an argument, so it cannot be passed by accident.
* Which split was evaluated is carried in the result and printed, so a `train_tune` number
  cannot be mistaken for a `dev` number in a later write-up.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from scifact.data.schema import Claim
from scifact.eval.metrics import Estimate, bootstrap, hit_at_k, reciprocal_rank

DEFAULT_KS = (1, 3, 5, 10, 20)


class DocumentRetriever(Protocol):
    def rank(self, query: str, k: int | None = None) -> list[int]: ...


@dataclass(frozen=True)
class RetrievalResult:
    name: str
    split: str
    n_claims: int
    hit_at: dict[int, Estimate]
    mrr: Estimate

    def row(self, ks: Sequence[int] = DEFAULT_KS) -> str:
        cells = "  ".join(f"{self.hit_at[k].value:>6.1%}" for k in ks)
        return f"  {self.name:<26} {cells}   {self.mrr.value:>6.3f}"


def evaluate_document_retrieval(
    retriever: DocumentRetriever,
    claims: Sequence[Claim],
    split: str,
    name: str,
    ks: Sequence[int] = DEFAULT_KS,
) -> RetrievalResult:
    """Evaluate over claims that have gold evidence.

    Claims with no evidence are excluded here on purpose. Retrieval recall is undefined when
    there is nothing to retrieve; those 37.6% of claims are the abstention problem and are
    scored in Stage 5, not by this metric.
    """
    scored = [c for c in claims if c.has_evidence]
    max_k = max(ks)

    per_k: dict[int, list[float]] = {k: [] for k in ks}
    rr: list[float] = []

    for claim in scored:
        gold = set(claim.evidence)
        ranked = retriever.rank(claim.claim, k=max_k)
        for k in ks:
            per_k[k].append(hit_at_k(ranked, gold, k))
        rr.append(reciprocal_rank(ranked, gold))

    return RetrievalResult(
        name=name,
        split=split,
        n_claims=len(scored),
        hit_at={k: bootstrap(v) for k, v in per_k.items()},
        mrr=bootstrap(rr),
    )


def print_table(results: Sequence[RetrievalResult], ks: Sequence[int] = DEFAULT_KS) -> None:
    if not results:
        return
    head = "  ".join(f"{'hit@' + str(k):>6}" for k in ks)
    print(f"  {'method':<26} {head}      MRR")
    print(f"  {'-' * 26} {'-' * len(head)}   ------")
    for r in results:
        print(r.row(ks))

    print()
    print(f"  split={results[0].split}   n={results[0].n_claims} claims with evidence")
    print("  95% bootstrap intervals (2,000 resamples over claims):")
    for r in results:
        widest = max(ks, key=lambda k: r.hit_at[k].half_width)
        print(f"    {r.name:<26} hit@{widest}: {r.hit_at[widest]}")
