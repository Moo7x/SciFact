# Evaluation protocol

> **Authoritative.** Every number in this repository must be produced under these rules.
> Written by Claude on 2026-09-19 at Mounir's direction. Amend it deliberately and date the
> amendment; do not let it drift to match whatever the code happens to do.

---

## 1. Splits

| Split | n | Role | Reuse |
|---|---|---|---|
| `train` | 809 | tuning anything **not** learned from train itself | freely |
| `train_fit` | ~647 | fitting learned parameters, from Stage 3 | freely |
| `train_tune` | ~162 | choosing hyperparameters of a *trained* model | freely |
| `dev` | 300 | **the holdout** | once per stage, recorded below |
| `test` | 300 | unusable — unlabelled, leaderboard only | never |

`train_fit` / `train_tune` is an 80/20 stratified split on whether a claim carries evidence
(seed `20260919`, `src/scifact/data/splits.py`). Stratified because 37.6% of claims have no
evidence and both halves must keep that balance.

**Stage-dependent rule.** The split exists to stop a model being tuned on data it was also
fitted on. A method that fits nothing — BM25, any unsupervised baseline — has no such exposure,
so it may tune on all 809 `train` claims. From Stage 3 onward, anything with learned parameters
fits on `train_fit` and tunes on `train_tune`.

**Rejected: the shipped 5-fold CV split.** It re-partitions `train ∪ dev`, dissolving the
official dev set into training (fold 1 puts 240 of the 300 dev claims on the training side).
Since `test` is unlabelled, adopting it would leave **no untouched labelled data anywhere**, and
every subsequent number would be in-sample. Its tighter point estimate is not worth losing the
only holdout the dataset has. See OQ-002.

## 2. Dev-split budget

`dev` degrades with every decision made against it. Each use is logged here.

| Date | Stage | What was measured | Why it justified spending the holdout |
|---|---|---|---|
| — | — | not yet spent | — |

Sweeping hyperparameters on `dev` is forbidden; `scripts/run_stage1.py --sweep` refuses it.

## 3. What counts as correct

**Primary metric: hit@k on documents.** 1.0 if any gold evidence abstract appears in the top k.

This matches what the next stage needs: Stage 2 asks whether the evidence reached the classifier
at all. Requiring *every* gold abstract answers a different question, and claims cite a mean of
only 1.17 documents, so the two rarely diverge. `recall@k` (fraction of gold retrieved) is
computed alongside.

**Secondary: MRR.** Sensitive to *where* the first hit lands, not just whether it fell inside
the window. Stage 4's agent pays per document it reads, so rank depth is a real cost.

**Precision is not reported, at any stage.** An unannotated retrieved abstract is not a labelled
negative — it is unlabelled. Counting it as an error would punish a retriever for surfacing
something possibly relevant. This is the trap named in the project plan, and the defence is to
not compute the metric.

**Scope.** Retrieval metrics are computed over claims that have evidence (505 of 809 in train).
Recall is undefined when there is nothing to retrieve. The 37.6% with no evidence are the
abstention problem and are scored in Stage 5, never folded into a retrieval number.

## 4. Uncertainty is mandatory

Every reported figure carries a 95% percentile bootstrap interval, 2,000 resamples, resampling
**claims** (the independent unit) rather than individual judgements.

**A point estimate without its interval may not be compared to another point estimate.** At
n=505 and p≈0.93 the interval half-width is ~2.2pp; at n=300 and p≈0.6 it is ~5.5pp. Differences
smaller than that are not differences.

## 5. Required controls

No retrieval number is interpretable alone. Each result is reported beside:

| Control | Answers |
|---|---|
| **random** | does the code run? |
| **longest-abstract** | is the metric measuring a corpus property rather than retrieval? |
| **positional** (sentence-level only) | ~80% of gold evidence sits in the back half of its abstract. Does the score come from matching, or from knowing conclusions come last? See OQ-004. |

A baseline exists to absorb credit that does not belong to the method.

## 6. Forbidden inputs

**`cited_doc_ids` is never an input to any retriever or classifier.** It is an annotation
artifact: all 564 evidence-bearing documents in train are inside it, and claims cite a mean of
1.17 documents, so it *is* the answer key. Demonstrated in
`scripts/demonstrate_leakage.py` — 62.5% → 99.5% top-1 with no change of method.
`src/scifact/eval/harness.py` does not accept it as an argument.

**Nothing fitted on evaluation data.** IDF over the shared corpus is fine — the corpus is
legitimately given and shared across splits by design. A threshold, calibration curve or
normalisation fitted on `dev` is not.

## 7. Contamination

SciFact has been public since 2020; pretrained models have plausibly seen it. This is not
fixable. It limits claims about **absolute** performance while leaving **controlled
comparisons** intact, since it affects both arms equally. Every headline result must state it.

## 8. Reporting rules

- Never call a `train` or `train_tune` number held-out.
- State the split, n, and the interval with every figure.
- State how many configurations were tried before the reported one. Trying m variants inflates
  the winner by roughly `σ√(2 ln m)`; at 25 configurations and σ≈1.1pp that is ~2.8pp of pure
  selection noise. See `docs/concepts/05`.
- A result where the cheaper method wins is a result. Report it.
