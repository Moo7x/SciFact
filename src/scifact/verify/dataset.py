"""Build (claim, evidence) training pairs for the cross-encoder.

Negatives come from two sources, and the distinction is load-bearing:

* **Gold hard negatives.** Abstracts a human read and labelled NOINFO for that claim.
  Confirmed against Wadden et al. 2020 s3.3 -- "annotators are shown a single claim-cited
  abstract pair" -- so NOINFO means examined and found empty, not unexamined. The paper's own
  baseline uses them identically.

* **Non-rationale sentences** from abstracts that *do* carry evidence. Also safe: the annotator
  marked which sentences are the rationale, so the rest of that abstract is labelled by
  omission within an abstract they read.

Nothing else is used. Retrieved-but-unannotated abstracts are **not** negatives -- they may be
relevant and simply unlabelled, and training on them as negatives corrupts both the model and
the evaluation. That is the trap named in the project plan and it is avoided by construction
here: this module never touches a retriever.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from scifact.data.schema import Claim, Document
from scifact.verify.labels import CONTRADICT, NEI, SUPPORT, Verdict

LABEL_TO_ID: dict[Verdict, int] = {SUPPORT: 0, CONTRADICT: 1, NEI: 2}
ID_TO_LABEL: dict[int, Verdict] = {v: k for k, v in LABEL_TO_ID.items()}


@dataclass(frozen=True)
class Pair:
    """One training example: a claim, some evidence text, and what that evidence shows."""

    claim: str
    evidence: str
    label: Verdict
    claim_id: int
    doc_id: int

    @property
    def label_id(self) -> int:
        return LABEL_TO_ID[self.label]


def build_pairs(
    claims: list[Claim],
    corpus: dict[int, Document],
    max_negatives_per_claim: int = 2,
    seed: int = 0,
) -> list[Pair]:
    """Positives from gold rationales, negatives from NOINFO abstracts and non-rationale text."""
    rng = random.Random(seed)  # noqa: S311
    pairs: list[Pair] = []

    for claim in claims:
        if claim.has_evidence:
            for doc_id, rationales in claim.evidence.items():
                doc = corpus.get(doc_id)
                if doc is None:
                    continue
                for rationale in rationales:
                    text = " ".join(
                        doc.abstract[i] for i in rationale.sentences if i < len(doc.abstract)
                    )
                    if not text:
                        continue
                    label: Verdict = SUPPORT if rationale.label == "SUPPORT" else CONTRADICT
                    pairs.append(Pair(claim.claim, text, label, claim.id, doc_id))

                # Non-rationale sentences from the same abstract: the annotator read this
                # abstract and did not mark these, so they are labelled by omission.
                used = {i for r in rationales for i in r.sentences}
                spare = [i for i in range(len(doc.abstract)) if i not in used]
                rng.shuffle(spare)
                for i in spare[:max_negatives_per_claim]:
                    pairs.append(Pair(claim.claim, doc.abstract[i], NEI, claim.id, doc_id))
        else:
            # Gold hard negatives: a human read this abstract and found nothing.
            for doc_id in claim.cited_doc_ids:
                doc = corpus.get(doc_id)
                if doc is None:
                    continue
                # Sample sentences rather than the whole abstract, so positives and negatives
                # have comparable length. A length cue the model could exploit would be a
                # shortcut, and a shortcut is a result that does not transfer.
                idxs = list(range(len(doc.abstract)))
                rng.shuffle(idxs)
                for i in idxs[:max_negatives_per_claim]:
                    pairs.append(Pair(claim.claim, doc.abstract[i], NEI, claim.id, doc_id))

    return pairs


def label_counts(pairs: list[Pair]) -> dict[Verdict, int]:
    counts: dict[Verdict, int] = {SUPPORT: 0, CONTRADICT: 0, NEI: 0}
    for p in pairs:
        counts[p.label] += 1
    return counts
