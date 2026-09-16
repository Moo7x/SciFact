"""Tests for the claim/corpus schema layer.

These use inline fixtures rather than the real corpus, so they run in CI where the data is
absent. The specific thing they lock down is the bug that motivated the module: a wrong key
name must raise, not silently produce an empty result.
"""

from __future__ import annotations

import pytest

from scifact.data.schema import (
    Rationale,
    SchemaError,
    parse_claim,
    parse_document,
    parse_rationale,
)

# The real on-disk shape, verified against the 2026-09-11 release.
REAL_CLAIM = {
    "id": 3,
    "claim": "1,000 genomes project enables mapping of genetic sequence variation.",
    "evidence": {"13734012": [{"sentences": [4], "label": "CONTRADICT"}]},
    "cited_doc_ids": [13734012],
}


def test_parses_the_real_shape() -> None:
    claim = parse_claim(REAL_CLAIM)
    assert claim.id == 3
    assert claim.has_evidence
    rationales = claim.evidence[13734012]
    assert len(rationales) == 1
    assert rationales[0].sentences == (4,)
    assert rationales[0].label == "CONTRADICT"


def test_wrong_key_name_raises_instead_of_returning_empty() -> None:
    """The regression test for the actual bug.

    Code that used `.get("sentence_indices", [])` produced an empty list here and reported
    'no evidence' for all 505 annotated claims without raising. The parser must refuse.
    """
    with pytest.raises(SchemaError, match="sentences"):
        parse_rationale({"sentence_indices": [4], "label": "SUPPORT"}, "test")


def test_error_names_the_keys_it_actually_found() -> None:
    """A schema error is only useful if it tells you what was there instead."""
    with pytest.raises(SchemaError) as exc:
        parse_rationale({"sentence_indices": [4], "label": "SUPPORT"}, "test")
    assert "sentence_indices" in str(exc.value)


def test_unexpected_label_raises() -> None:
    with pytest.raises(SchemaError, match="unexpected label"):
        parse_rationale({"sentences": [1], "label": "MAYBE"}, "test")


def test_empty_evidence_is_valid_and_means_not_enough_info() -> None:
    claim = parse_claim({"id": 1, "claim": "x", "evidence": {}, "cited_doc_ids": [99]})
    assert not claim.has_evidence
    assert claim.cited_doc_ids == (99,)


def test_missing_required_key_raises() -> None:
    with pytest.raises(SchemaError, match="cited_doc_ids"):
        parse_claim({"id": 1, "claim": "x", "evidence": {}})


def test_document_parses_and_reports_length() -> None:
    doc = parse_document(
        {"doc_id": 7, "title": "T", "abstract": ["a.", "b.", "c."], "structured": False}
    )
    assert len(doc) == 3
    assert doc.abstract[2] == "c."


@pytest.mark.parametrize(
    ("sentences", "expected"),
    [((4,), True), ((3, 4, 5), True), ((1, 4), False), ((), True)],
)
def test_contiguity(sentences: tuple[int, ...], expected: bool) -> None:
    assert Rationale(sentences=sentences, label="SUPPORT").is_contiguous is expected
