"""Claim-level verdicts.

A claim's verdict is derived from its rationales: SUPPORT or CONTRADICT if it has evidence of
that kind, NEI if it has none. Verified on the real data: no claim in train or dev carries both
SUPPORT and CONTRADICT rationales, so the derivation is unambiguous.

Distribution:

    train (809):  SUPPORT 332   CONTRADICT 173   NEI 304
    dev   (300):  SUPPORT 124   CONTRADICT  64   NEI 112

Majority class is SUPPORT at 41.0% of train. That is the floor any verifier must clear.
"""

from __future__ import annotations

from typing import Final, Literal

from scifact.data.schema import Claim

Verdict = Literal["SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO"]

# `Final[Verdict]` rather than a bare assignment. Without the annotation the inferred type is
# plain `str`, and every function returning one of these constants then fails to satisfy a
# `-> Verdict` signature. The annotation is what keeps a typo in a verdict string a type error
# instead of a runtime surprise.
SUPPORT: Final[Verdict] = "SUPPORT"
CONTRADICT: Final[Verdict] = "CONTRADICT"
NEI: Final[Verdict] = "NOT_ENOUGH_INFO"

VERDICTS: Final[tuple[Verdict, ...]] = (SUPPORT, CONTRADICT, NEI)


def claim_label(claim: Claim) -> Verdict:
    labels = {r.label for rationales in claim.evidence.values() for r in rationales}
    if not labels:
        return NEI
    if len(labels) > 1:
        # Does not occur in the shipped data. If a future release introduces it, this must be
        # a deliberate decision rather than a silent coin flip.
        raise ValueError(f"claim {claim.id} has mixed rationale labels: {sorted(labels)}")
    label = labels.pop()
    return SUPPORT if label == "SUPPORT" else CONTRADICT
