"""Tests for BM25, metrics and splits. Inline fixtures, so these run in CI without the corpus."""

from __future__ import annotations

import pytest

from scifact.data.schema import Claim, Rationale
from scifact.data.splits import split_train
from scifact.eval.metrics import bootstrap, hit_at_k, recall_at_k, reciprocal_rank
from scifact.retrieval.baselines import PopularityRetriever, RandomRetriever
from scifact.retrieval.bm25 import BM25, tokenize

DOCS = {
    1: "melanoma treatment melanoma melanoma response",
    2: "cardiac arrest resuscitation outcomes",
    3: "melanoma diagnosis",
    4: "unrelated botany plants photosynthesis",
}


@pytest.fixture
def index() -> BM25:
    return BM25(list(DOCS), list(DOCS.values()))


# ------------------------------------------------------------------ BM25


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Melanoma, treatment (2020)!") == ["melanoma", "treatment", "2020"]


def test_ranks_the_relevant_document_first(index: BM25) -> None:
    assert index.rank("melanoma")[0] in {1, 3}


def test_query_term_absent_from_corpus_scores_nothing(index: BM25) -> None:
    assert all(s == 0.0 for s in index.score_all("zzzznonexistent"))


def test_idf_is_never_negative(index: BM25) -> None:
    """The +1 inside the log matters: without it a term in every document scores negative,
    turning a match into a penalty."""
    ubiquitous = BM25([1, 2], ["common word", "common other"])
    assert ubiquitous.idf["common"] > 0


def test_term_frequency_saturates(index: BM25) -> None:
    """Ten mentions must not score ten times one mention -- that is the point of k1."""
    one = BM25([1], ["melanoma filler filler filler filler filler filler filler filler filler"])
    many = BM25([1], ["melanoma " * 10])
    assert many.score_all("melanoma")[0] < 10 * one.score_all("melanoma")[0]


def test_b_controls_length_normalisation() -> None:
    docs = ["term extra " + "pad " * 50, "term"]
    no_norm = BM25([1, 2], docs, b=0.0).score_all("term")
    full_norm = BM25([1, 2], docs, b=1.0).score_all("term")
    # With normalisation on, the short document is favoured more strongly.
    assert (full_norm[1] - full_norm[0]) > (no_norm[1] - no_norm[0])


def test_ranking_is_deterministic_under_ties(index: BM25) -> None:
    assert index.rank("nothingmatcheshere") == index.rank("nothingmatcheshere")


def test_mismatched_lengths_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        BM25([1, 2], ["only one"])


# ------------------------------------------------------------------ metrics


def test_hit_at_k_is_binary() -> None:
    assert hit_at_k([5, 9, 3], {3}, 3) == 1.0
    assert hit_at_k([5, 9, 3], {3}, 2) == 0.0


def test_recall_at_k_is_fractional() -> None:
    assert recall_at_k([1, 2, 9], {1, 2}, 3) == 1.0
    assert recall_at_k([1, 9, 9], {1, 2}, 3) == 0.5


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([9, 9, 7], {7}) == pytest.approx(1 / 3)
    assert reciprocal_rank([9, 9], {7}) == 0.0


def test_bootstrap_brackets_the_point_estimate() -> None:
    est = bootstrap([1.0] * 70 + [0.0] * 30)
    assert est.value == pytest.approx(0.70)
    assert est.low <= est.value <= est.high
    assert est.n == 100


def test_bootstrap_interval_narrows_with_n() -> None:
    small = bootstrap([1.0] * 35 + [0.0] * 15)
    large = bootstrap([1.0] * 350 + [0.0] * 150)
    assert large.half_width < small.half_width


def test_bootstrap_is_reproducible() -> None:
    data = [1.0, 0.0, 1.0, 1.0, 0.0]
    assert bootstrap(data) == bootstrap(data)


# ------------------------------------------------------------------ baselines


def test_random_retriever_is_stable_for_the_same_query() -> None:
    r = RandomRetriever([1, 2, 3, 4, 5])
    assert r.rank("a claim") == r.rank("a claim")


def test_random_retriever_differs_across_queries() -> None:
    r = RandomRetriever(list(range(60)))
    assert r.rank("claim one") != r.rank("claim two")


def test_popularity_retriever_ignores_the_query() -> None:
    p = PopularityRetriever([1, 2, 3], [5, 50, 10])
    assert p.rank("anything") == p.rank("something else") == [2, 3, 1]


# ------------------------------------------------------------------ splits


def _claim(i: int, *, evidence: bool) -> Claim:
    ev: dict[int, tuple[Rationale, ...]] = (
        {100 + i: (Rationale((0,), "SUPPORT"),)} if evidence else {}
    )
    return Claim(id=i, claim=f"c{i}", evidence=ev, cited_doc_ids=(100 + i,))


def _evidence_ratio(group: list[Claim]) -> float:
    return sum(c.has_evidence for c in group) / len(group)


def test_split_is_deterministic() -> None:
    claims = [_claim(i, evidence=i % 3 != 0) for i in range(100)]
    assert [c.id for c in split_train(claims)[1]] == [c.id for c in split_train(claims)[1]]


def test_split_is_disjoint_and_complete() -> None:
    claims = [_claim(i, evidence=i % 3 != 0) for i in range(100)]
    fit, tune = split_train(claims)
    assert not {c.id for c in fit} & {c.id for c in tune}
    assert len(fit) + len(tune) == 100


def test_split_preserves_the_abstention_ratio() -> None:
    claims = [_claim(i, evidence=i % 3 != 0) for i in range(300)]
    fit, tune = split_train(claims)
    assert _evidence_ratio(fit) == pytest.approx(_evidence_ratio(tune), abs=0.02)
