"""Tests for verdict derivation and the fixed verifiers."""

from __future__ import annotations

import pytest

from scifact.data.schema import Claim, Document, Rationale
from scifact.eval.decomposition import (
    condition_a_oracle_sentences,
    condition_b_oracle_abstract,
)
from scifact.verify.baselines import LexicalVerifier, MajorityVerifier
from scifact.verify.labels import CONTRADICT, NEI, SUPPORT, claim_label


def _claim(cid: int, text: str, label: str | None, sentences: tuple[int, ...] = (0,)) -> Claim:
    evidence: dict[int, tuple[Rationale, ...]] = (
        {} if label is None else {7: (Rationale(sentences, label),)}
    )
    return Claim(id=cid, claim=text, evidence=evidence, cited_doc_ids=(7,))


DOC = Document(
    doc_id=7,
    title="Study",
    abstract=(
        "Treatment increased survival significantly.",
        "Background material about mice.",
        "Nothing relevant here at all.",
    ),
    structured=False,
)


# ------------------------------------------------------------------ labels


def test_no_evidence_is_nei() -> None:
    assert claim_label(_claim(1, "x", None)) == NEI


def test_support_and_contradict_derive_from_rationales() -> None:
    assert claim_label(_claim(1, "x", "SUPPORT")) == SUPPORT
    assert claim_label(_claim(2, "x", "CONTRADICT")) == CONTRADICT


def test_mixed_labels_raise_rather_than_guess() -> None:
    """Does not occur in the shipped data. If a future release introduces it, the choice must
    be deliberate rather than a silent coin flip."""
    mixed = Claim(
        id=3,
        claim="x",
        evidence={7: (Rationale((0,), "SUPPORT"), Rationale((1,), "CONTRADICT"))},
        cited_doc_ids=(7,),
    )
    with pytest.raises(ValueError, match="mixed rationale labels"):
        claim_label(mixed)


# ------------------------------------------------------------------ verifiers


def test_majority_ignores_the_evidence_entirely() -> None:
    v = MajorityVerifier()
    assert v.predict("anything", []) == SUPPORT
    assert v.predict("anything", ["completely contradictory text"]) == SUPPORT


def test_lexical_abstains_with_no_candidates() -> None:
    assert LexicalVerifier().predict("treatment increased survival", []) == NEI


def test_lexical_abstains_below_threshold() -> None:
    v = LexicalVerifier(nei_threshold=0.9)
    assert v.predict("treatment increased survival", ["unrelated botany text"]) == NEI


def test_lexical_detects_polarity_disagreement() -> None:
    v = LexicalVerifier(nei_threshold=0.2)
    assert v.predict("treatment increased survival", ["treatment reduced survival"]) == CONTRADICT


def test_lexical_agrees_on_matching_polarity() -> None:
    v = LexicalVerifier(nei_threshold=0.2)
    assert v.predict("treatment increased survival", ["treatment increased survival"]) == SUPPORT


def test_lexical_threshold_monotonically_increases_abstention() -> None:
    """Raising the bar can only make the verifier abstain more, never less."""
    sentences = ["treatment increased survival in the cohort"]
    abstentions = [
        LexicalVerifier(t).predict("treatment increased survival", sentences) == NEI
        for t in (0.0, 0.3, 0.6, 0.9, 1.0)
    ]
    assert abstentions == sorted(abstentions)


# ------------------------------------------------------------------ decomposition


def test_conditions_exclude_claims_without_evidence() -> None:
    """A and B are undefined for NEI claims: there is no gold rationale to hand over, and
    substituting cited_doc_ids would leak the answer key."""
    claims = [_claim(1, "x", "SUPPORT"), _claim(2, "y", None)]
    corpus = {7: DOC}
    assert condition_a_oracle_sentences(MajorityVerifier(), claims, corpus).accuracy.n == 1
    assert condition_b_oracle_abstract(MajorityVerifier(), claims, corpus).accuracy.n == 1


def test_condition_b_sees_more_sentences_than_condition_a() -> None:
    seen: list[int] = []

    class Spy:
        name = "spy"

        def predict(self, claim: str, sentences: list[str]) -> str:
            seen.append(len(sentences))
            return SUPPORT

    claims = [_claim(1, "x", "SUPPORT", sentences=(0,))]
    corpus = {7: DOC}
    condition_a_oracle_sentences(Spy(), claims, corpus)  # type: ignore[arg-type]
    condition_b_oracle_abstract(Spy(), claims, corpus)  # type: ignore[arg-type]
    assert seen == [1, 3]


def test_per_class_recall_reports_nan_for_absent_classes() -> None:
    claims = [_claim(1, "x", "SUPPORT")]
    result = condition_a_oracle_sentences(MajorityVerifier(), claims, {7: DOC})
    recalls = result.per_class_recall()
    assert recalls[SUPPORT] == 1.0
    assert recalls[NEI] != recalls[NEI]  # NaN: the class was never present
