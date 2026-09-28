# SciFact — Evidence Retrieval and Claim Verification

[![CI](https://github.com/Moo7x/SciFact/actions/workflows/ci.yml/badge.svg)](https://github.com/Moo7x/SciFact/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)

Given a scientific claim, find the abstracts that bear on it, identify the sentences that
support or contradict it, and return an evidence-linked verdict — or say explicitly that the
evidence is insufficient.

Built on [SciFact](https://github.com/allenai/scifact) (Wadden et al., 2020): 1,409 expert-written
claims checked against 5,183 biomedical abstracts. This is an implementation-and-comparison
study over a public benchmark, not a new method. The emphasis is on **measuring honestly**:
every number below comes from a re-runnable command, states its split, and carries a confidence
interval.

> **Status:** Stages 0–3 complete. All results so far are on **training data**
> (`train` / `train_tune`). The held-out `dev` split has deliberately not been touched yet.

---

## The research question

*When does iterative, adaptive evidence retrieval improve grounded decisions enough to justify
its extra cost and latency, compared to a fixed pipeline?*

Stages 0–3 build the fixed pipeline and the measurement machinery needed to answer that fairly.
Stage 4 builds the adaptive alternative.

---

## Results so far

### Stage 1 — Lexical retrieval (BM25, written from scratch)

`train`, 505 claims with evidence, searching all 5,183 abstracts:

| method | hit@1 | hit@5 | hit@10 | MRR |
|---|---|---|---|---|
| random (floor) | 0.0% | 0.2% | 0.2% | 0.000 |
| longest-abstract (control) | 0.0% | 0.0% | 0.4% | 0.001 |
| **BM25** (k1=1.2, b=0.75) | **74.9%** [70.9, 78.6] | **90.1%** | **92.7%** | **0.813** |

A 25-point grid over `k1` and `b` spanned 92.5–93.7% hit@10: a 1.2-point spread, narrower than
the ~2.2-point confidence interval. No setting was distinguishable from the defaults, so the
defaults were kept.

### Stage 2 — Separating retrieval failure from reasoning failure

One run over retrieved evidence cannot tell *"the evidence was never found"* from *"it was found
and misread."* So the verifier is scored under three conditions, removing one error source at a
time:

| condition | what the verifier sees |
|---|---|
| A. oracle sentences | the gold rationale sentences |
| B. oracle abstract | every sentence of the gold abstract |
| C. retrieved@3 | every sentence of BM25's top 3 abstracts |

A lexical verifier (word overlap + polarity cues) came out **more accurate as the evidence got
worse** (A 27.5% → C 42.4% on `train`). A negative decomposition cost is a diagnostic in its own
right: the model was not responding to evidence at all, just to *how much text* it was given. It
also lost to always predicting the majority class (65.7%). A single run over retrieved evidence
would have reported 42.4% and hidden all of this.

### Stage 3 — Fine-tuned cross-encoder

`cross-encoder/ms-marco-MiniLM-L-6-v2` (22.7M parameters), fine-tuned with a hand-written
PyTorch training loop.

`train_tune`, 101 evidence-bearing claims, gold rationale sentences given (condition A), so all
three verifiers see identical input:

| verifier | accuracy | SUPPORT recall | CONTRADICT recall |
|---|---|---|---|
| majority (always SUPPORT) | **66.3%** [56.4, 75.2] | 100% | 0% |
| lexical (Stage 2) | 28.7% [19.8, 37.6] | 35.8% | 14.7% |
| **cross-encoder** | 44.6% [34.7, 54.5] | 25.4% | **82.4%** |

It fixed the failure it was meant to fix: negation, CONTRADICT 14.7% → 82.4%. It broke SUPPORT,
and it still scores below always predicting SUPPORT. A controlled ablation showed why
(pair level, same `train_tune` pairs in every arm):

| class weighting | SUPPORT | CONTRADICT | NEI | macro-recall |
|---|---|---|---|---|
| on | 22.4% | 79.7% | 79.9% | 60.7% |
| off | **60.8%** | **20.3%** | 89.7% | 56.9% |

Removing class weights recovered SUPPORT and dropped CONTRADICT in the same run. Macro-recall
barely moves. **The weights do not make the model better; they choose which class it
sacrifices.** This model cannot reliably tell a claim from its own negation, and where it
can't, the class prior decides.

**That conclusion is about this model, not yet about the task.** A post-hoc audit
([ADR-0002](docs/adr/0002-stage-3-model-choice-audit.md)) found the model was a poor match:
its original head scores search relevance (one output) rather than entailment, and its
web-trained vocabulary fragments biomedical text the most of five candidates. Zero-shot,
with no SciFact training, a PubMed-pretrained NLI model reached 80% SUPPORT and 73%
CONTRADICT recall at once, with no see-saw, though it has probably seen SciFact's abstracts
in pretraining. The small model was chosen because it trains comfortably on a 4 GB GPU;
testing the better-matched ones properly is Stage 3b.

---

## What the analysis found about the dataset

Observations from this project's own measurements. They are not claimed as novel; SciFact has
been studied extensively. Each one changed how the project is evaluated.

1. **The shipped 5-fold cross-validation split re-partitions `train ∪ dev`, not `train`.**
   Verified by claim ID: in fold 1, 240 of the 300 official dev claims sit on the training side.
   Since `test` is unlabelled, adopting it leaves no untouched labelled data anywhere. Rejected
   for that reason.

2. **23% of dev claims have a near-duplicate twin in train** (token Jaccard ≥ 0.8), and every
   inspected twin cites the same abstract. SciFact built its CONTRADICT claims by negating
   SUPPORT claims, and the two halves of a pair can land in different splits:

   ```
   DEV   [CONTRADICT] A high microerythrocyte count raises vulnerability to severe anemia...
   TRAIN [SUPPORT]    A high microerythrocyte count protects against severe anemia...
   ```

   Dev results will therefore be reported separately for twinned and untwinned claims.

3. **`cited_doc_ids` is the answer key in disguise.** It looks like input metadata. Every
   evidence-bearing abstract is in it, and claims cite a mean of 1.17 documents. With the scoring
   function unchanged, restricting retrieval to it lifts top-1 accuracy from **62.5% to 99.5%**.
   The evaluation harness does not accept that field at all.

4. **Evidence position is heavily skewed.** About 80% of gold evidence sentences sit in the back
   half of their abstract (21.5% in the final tenth). A "retriever" that ignores the claim and
   returns the last sentences would score well above chance, so sentence-level results need a
   positional control.

5. **Lexical retrieval has a visible ceiling.** 11.1% of gold rationales share *zero* content
   words with their claim. Nothing BM25 can tune will find those.

6. **"Not enough info" abstracts are hard negatives.** They share a lot of the claim's vocabulary
   (32% coverage vs 5% for a random abstract), and 58% of them rank in the top 10 lexically.
   Word overlap separates them from true evidence with AUC 0.858: useful, but a poor basis for
   deciding when to abstain.

---

## How evaluation works

The full protocol is in [`docs/EVALUATION.md`](docs/EVALUATION.md). The short version:

- **Splits.** `train` (809) is split 80/20 into `train_fit` and `train_tune` for anything that
  learns. `dev` (300) is the holdout and is used at most once per stage. `test` is unlabelled
  and never used.
- **Every number carries a 95% bootstrap interval**, resampled over claims. A difference smaller
  than the interval is not reported as a difference.
- **Macro-recall is required alongside accuracy.** Accuracy on imbalanced classes rewards a model
  for abandoning the rare ones, and Stage 3 did exactly that.
- **Controls are mandatory.** A baseline exists to absorb the credit that doesn't belong to the
  method.
- **Tuning is counted.** Trying *m* variants inflates the best one by roughly σ√(2 ln m).
  [`scripts/demonstrate_leakage.py`](scripts/demonstrate_leakage.py) simulates it: on 300 claims,
  picking the best of 20 equally good models adds about 5 points of pure noise.

---

## Reproduce

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/). Training runs on a CUDA GPU if
available (about 3 minutes on an RTX 3050 Ti) and falls back to CPU (about 40 minutes).

```bash
uv sync --all-groups
uv run python scripts/download_data.py          # fetches SciFact and verifies checksums
uv run python scripts/describe_data.py          # dataset measurements
uv run python scripts/demonstrate_leakage.py    # the two leakage demonstrations
uv run python scripts/run_stage1.py             # BM25 against its controls
uv run python scripts/run_stage2.py             # three-condition failure decomposition
uv run python scripts/train_crossencoder.py     # Stage 3 fine-tune
uv run python scripts/eval_stage3.py            # confusion matrix + decomposition
uv run python scripts/run_oq010.py              # class-weighting ablation
uv run pytest                                   # 46 tests; CI runs them on Linux and Windows
```

The dataset is not committed. `data/MANIFEST.json` records its SHA-256 checksums, so
`scripts/download_data.py --verify` confirms you are running on exactly the data behind these
results.

---

## Repository layout

```
src/scifact/
  data/        typed loaders that fail loudly on schema drift; deterministic splits
  retrieval/   BM25 and control baselines
  verify/      verdict labels, lexical verifier, cross-encoder wrapper, training pairs
  eval/        metrics with bootstrap intervals, the three-condition decomposition
scripts/       every reported number comes from a command in here
tests/         unit tests using inline fixtures, so CI runs without the dataset
lessons/       runnable walkthroughs of the training code: tokenization, batching
docs/
  EVALUATION.md       the protocol
  OPEN_QUESTIONS.md   every open question, and how each resolved one was answered
  concepts/           explanations written during the project, one per topic
  adr/                architecture decision records
```

---

## Limitations

- **Small data.** 809 training claims. Most intervals here are several points wide, and some
  differences discussed above do not clear them. That is stated wherever it applies.
- **Nothing is held-out yet.** Every result is on `train` or `train_tune`.
- **Contamination.** SciFact has been public since 2020, so pretrained models have plausibly seen
  it. This limits claims about absolute performance. Comparisons made under matched conditions
  are affected equally on both sides.
- **One trained model.** Stage 3 fine-tuned a single 22.7M-parameter model chosen for cost,
  not fit. Better-matched models were only evaluated zero-shot (ADR-0002).

---

## Roadmap

- **Stage 4:** fixed pipeline vs a bounded agent (reformulate, search again, read, stop) under
  the same call budget. A result where the cheaper pipeline wins will be reported as a result.
- **Stage 5:** abstention as a calibrated decision with a defended operating point.
- **Stage 6:** reliability — structured evidence IDs with schema validation, tracing, timeouts
  and fallbacks, regression tests, and measured behaviour under hostile instructions injected
  into documents.

---

## Data and citation

SciFact claims and annotations are released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The abstracts are under
[ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/). This repository does not
redistribute them; `scripts/download_data.py` fetches them from the source.

```bibtex
@inproceedings{Wadden2020FactOF,
  title     = {Fact or Fiction: Verifying Scientific Claims},
  author    = {David Wadden and Shanchuan Lin and Kyle Lo and Lucy Lu Wang and
               Madeleine van Zuylen and Arman Cohan and Hannaneh Hajishirzi},
  booktitle = {EMNLP},
  year      = {2020}
}
```
