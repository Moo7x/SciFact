"""Typed access to SciFact claims and corpus, with loud schema validation.

This module exists because of a real bug. Both scripts originally reached into the raw JSON
with ``rationale.get("sentence_indices", [])``. The actual key is ``sentences``. ``.get()`` with
a default returned ``[]`` every single time, silently, so the viewer displayed abstracts with
no evidence marked and the statistics script counted zero rationales across 809 claims.

Nothing raised. Nothing warned. The output looked perfectly reasonable.

The lesson generalises: **``.get(key, default)`` converts a typo into a plausible-looking wrong
answer.** It is the right call when a key is genuinely optional. It is the wrong call when the
key is required by the schema, because then a mistake in the key name is indistinguishable from
a legitimately absent value.

So everything here fails loudly on an unexpected shape, and the failure names the keys it did
find — the single most useful thing an error can tell you when a schema has drifted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The verified on-disk schema, checked against the release downloaded 2026-09-11
# (archive sha256 11c62128...76be). If a future release changes these, the loaders below
# raise rather than silently returning empty results.
CLAIM_KEYS = frozenset({"id", "claim", "evidence", "cited_doc_ids"})
DOC_KEYS = frozenset({"doc_id", "title", "abstract", "structured"})
RATIONALE_SENTENCES_KEY = "sentences"
RATIONALE_LABEL_KEY = "label"

VALID_LABELS = frozenset({"SUPPORT", "CONTRADICT"})


class SchemaError(ValueError):
    """Raised when the data on disk does not match the schema this code was written against."""


@dataclass(frozen=True)
class Rationale:
    """One annotated justification: which sentences, and what they establish."""

    sentences: tuple[int, ...]
    label: str

    @property
    def is_contiguous(self) -> bool:
        s = sorted(self.sentences)
        return s == list(range(s[0], s[0] + len(s))) if s else True


@dataclass(frozen=True)
class Claim:
    id: int
    claim: str
    # doc_id -> rationales. Empty dict means NOT_ENOUGH_INFO against every cited abstract.
    evidence: dict[int, tuple[Rationale, ...]]
    cited_doc_ids: tuple[int, ...]

    @property
    def has_evidence(self) -> bool:
        return bool(self.evidence)


@dataclass(frozen=True)
class Document:
    doc_id: int
    title: str
    abstract: tuple[str, ...]
    structured: bool

    def __len__(self) -> int:
        return len(self.abstract)


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise SchemaError(
            f"{where}: required key {key!r} is missing. Keys present: {sorted(mapping)}. "
            f"The data format has changed, or the key name in this code is wrong."
        )
    return mapping[key]


def parse_rationale(raw: dict[str, Any], where: str) -> Rationale:
    sentences = _require(raw, RATIONALE_SENTENCES_KEY, where)
    label = _require(raw, RATIONALE_LABEL_KEY, where)
    if not isinstance(sentences, list) or not all(isinstance(i, int) for i in sentences):
        raise SchemaError(
            f"{where}: {RATIONALE_SENTENCES_KEY!r} must be a list of ints, got {sentences!r}"
        )
    if label not in VALID_LABELS:
        raise SchemaError(
            f"{where}: unexpected label {label!r}, expected one of {sorted(VALID_LABELS)}"
        )
    return Rationale(sentences=tuple(sentences), label=label)


def parse_claim(raw: dict[str, Any]) -> Claim:
    claim_id = _require(raw, "id", "claim")
    where = f"claim {claim_id}"
    evidence_raw = _require(raw, "evidence", where)
    if not isinstance(evidence_raw, dict):
        raise SchemaError(f"{where}: 'evidence' must be an object, got {type(evidence_raw)}")

    evidence: dict[int, tuple[Rationale, ...]] = {}
    for doc_id_str, rationales in evidence_raw.items():
        evidence[int(doc_id_str)] = tuple(
            parse_rationale(r, f"{where} doc {doc_id_str}") for r in rationales
        )

    return Claim(
        id=claim_id,
        claim=_require(raw, "claim", where),
        evidence=evidence,
        cited_doc_ids=tuple(_require(raw, "cited_doc_ids", where)),
    )


def parse_document(raw: dict[str, Any]) -> Document:
    doc_id = _require(raw, "doc_id", "document")
    where = f"doc {doc_id}"
    abstract = _require(raw, "abstract", where)
    if not isinstance(abstract, list):
        raise SchemaError(f"{where}: 'abstract' must be a list of sentences")
    return Document(
        doc_id=doc_id,
        title=_require(raw, "title", where),
        abstract=tuple(abstract),
        structured=bool(raw.get("structured", False)),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/download_data.py")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_claims(path: Path) -> list[Claim]:
    return [parse_claim(raw) for raw in _read_jsonl(path)]


def load_corpus(path: Path) -> dict[int, Document]:
    docs = [parse_document(raw) for raw in _read_jsonl(path)]
    return {d.doc_id: d for d in docs}
