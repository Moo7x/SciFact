"""Claim verification: given a claim and candidate evidence, decide what the evidence shows."""

from scifact.verify.baselines import LexicalVerifier, MajorityVerifier
from scifact.verify.labels import CONTRADICT, NEI, SUPPORT, Verdict, claim_label

__all__ = [
    "CONTRADICT",
    "NEI",
    "SUPPORT",
    "LexicalVerifier",
    "MajorityVerifier",
    "Verdict",
    "claim_label",
]
