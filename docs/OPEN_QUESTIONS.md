# Open questions

Questions asked that I could not answer, and decisions not yet made.
Per PART IX of the plan, an unanswered question here is the highest-value signal in the project.

Status key: **OPEN** / **ANSWERED** (with date and where the answer lives) / **DEFERRED** (with trigger)

---

## OQ-001 — Stage 2 cannot decompose failure as written — ANSWERED 2026-09-19

**Raised:** 2026-09-11 (Claude)

The plan's Stage 2 says: add a fixed classifier over retrieved evidence, and measure retrieval
failure and reasoning failure *independently*. A single run over retrieved evidence cannot do
this. A wrong verdict is equally consistent with "the evidence was never retrieved" and with
"the evidence was retrieved and the classifier misread it" — one signal, two hypotheses.

Separating them requires at least one condition in which retrieval is not the variable.

**Why this blocks Stage 1, not Stage 2:** if the protocol requires gold-conditioned runs, the
Stage 1 evaluation harness must support them from the start. Retrofitting that is a common
route to leakage.

**Owner:** Mounir — this is evaluation protocol. Do not let Claude design it.

## OQ-002 — Held-out split vs the shipped 5-fold CV split — OPEN

**Raised:** 2026-09-11 (Claude). **Unblocked 2026-09-11:** counts are now measured.

### Measured facts (verified, not assumed)

| Split | Claims | Labelled? |
|---|---|---|
| `claims_train.jsonl` | 809 | yes |
| `claims_dev.jsonl` | 300 | yes |
| `claims_test.jsonl` | 300 | **no** — leaderboard only |
| `corpus.jsonl` | 5,183 abstracts | — |

Verified by claim-ID set operations, not by arithmetic on file sizes:

- `train` and `dev` are disjoint (overlap 0). `test` overlaps neither.
- **The shipped 5-fold CV split is a re-partition of `train` ∪ `dev` (1,109 claims), not of
  `train` alone.** The five fold-dev sets partition all 1,109 claims exactly once each.
- Concretely: in fold 1, **240 of the 300 official dev claims sit on the *training* side.**

### Why that matters more than it first looks

The plan's own trap list says *"do not tune on dev and then report it as held-out."*
The CV split makes that trap unavoidable rather than optional: adopting it dissolves the
official dev set into training, and because `test` is unlabelled, **no untouched labelled
data remains anywhere in the dataset.** Every subsequent number would be an in-sample number.

### The actual trilemma

1. **Official train/dev.** 809 train, 300 held out. Simple and standard. But dev then gets
   consumed by model selection, and after that it is no longer a clean holdout.
2. **Shipped 5-fold CV.** Uses all 1,109 claims, and averaging across folds tightens the
   estimate. Cost: nothing labelled is left untouched, so "held-out" leaves the vocabulary.
3. **Carve a holdout out of `train` only.** Keeps the official dev set pristine as a second,
   independent check. Cost: shrinks training data, and the holdout is small.

### The arithmetic to do before deciding

For a proportion metric (recall@k, accuracy) estimated on `n` claims, the normal-approximation
95% interval half-width is `1.96 * sqrt(p(1-p)/n)`. At `p ≈ 0.6`, `n = 300` gives roughly
±5.5 percentage points. Work the same number for each option above, then ask: **is that
interval narrower than the improvement Stage 3 is supposed to produce?** If it is not, the
comparison cannot be made on that split no matter how the model performs.

**Owner:** Mounir. Outcome becomes an ADR. Claude states the facts; the choice is not Claude's.

## OQ-003 — What counts as a correct retrieval? — OPEN

**Raised:** 2026-09-11 (Claude)

Not yet discussed. Abstract-level or sentence-level? Does retrieving the right abstract but
the wrong sentence count as a hit? The plan's trap list warns that unannotated retrieved
documents are not guaranteed negatives, which means precision is not straightforwardly
computable here. This is `docs/EVALUATION.md` territory.

**Owner:** Mounir.

## OQ-004 — A positional baseline is mandatory before any BM25 number means anything — OPEN

**Raised:** 2026-09-16 (Claude), from a measurement prompted by Mounir's observation that
evidence tends to be the concluding sentence.

Measured on train: ~80% of gold evidence sentences sit in the **back half** of their abstract
(21.5% in the final tenth alone). Only 15.4% of rationales include the literal last sentence,
so the effect is a strong skew rather than a rule.

**Consequence.** A "retriever" that ignores the claim entirely and returns the last two
sentences of each cited abstract would score well above chance. Any BM25 recall@k reported
without that comparison is uninterpretable: it cannot be told apart from the positional prior.

A baseline exists to absorb the credit that does not belong to your method. This one is not
optional.

**Owner:** Mounir — it belongs in `docs/EVALUATION.md` as a required comparison condition.
Claude states the requirement; what the baseline is exactly, and what it must beat, is protocol.

## OQ-005 — Do sentence-segmentation errors split real evidence? — OPEN

**Raised:** 2026-09-16 (Claude), from Mounir's observation that related lines get split apart.

Abstracts arrive pre-segmented by an automatic splitter that breaks on punctuation, not
meaning. 98.3% of rationales are contiguous, but the median rationale is one sentence, so most
of that contiguity is trivial.

**The unmeasured risk:** if the splitter cut one logical statement across two lines and only
one is annotated, a retriever that finds the annotated sentence still holds half the
justification — and would be scored as fully correct. That inflates sentence-level recall
relative to what a reader would accept.

Not yet quantified. Requires deciding what counts as a bad split first, which is protocol.

## OQ-006 — Retrieval score is a weak abstention signal; what replaces it? — OPEN

**Raised:** 2026-09-17 (Claude), from Mounir's observation that NOT_ENOUGH_INFO abstracts
share the claim's keywords while discussing a different question.

Measured on train (`scripts/analyze_negatives.py`, seed 0):

| Condition | n | mean claim-vocabulary coverage |
|---|---|---|
| has evidence | 564 | 62% |
| NOT_ENOUGH_INFO (cited) | 330 | 32% |
| random abstract | 809 | 5% |

NEI sits ~47% of the way from random to evidence. Crude lexical ranking puts **58.4% of NEI
cited abstracts in the top 10** of all 5,183, and 24.8% at rank 1.

Separability of the two labelled groups on that signal alone: **AUC 0.858**, best
single-threshold accuracy 78.3% against a 63.1% majority-class floor.

**The consequence.** The obvious abstention rule — *"abstain when the top retrieval score is
low"* — inherits an 86%-reliable discriminator as its ceiling, because the negatives are
high-scoring by construction. It is not useless signal, but it is not calibration either.

**The question:** what signal does the abstention decision actually use, and how is it
calibrated? This is Stage 5 and Checkpoint 6, surfaced early because it constrains what Stage 2's
classifier must output — a score that can be calibrated, not just an argmax.

**Owner:** Mounir. Claude reports the separability; choosing the operating point is protocol.

## OQ-007 — Are the NEI cited documents usable as guaranteed hard negatives? — OPEN

**Raised:** 2026-09-17 (Claude).

The project plan warns: *"Unannotated retrieved documents are not guaranteed negatives — they
may be relevant but unlabelled. Treating them as negatives corrupts training and evaluation
both."*

The NEI cited documents are different: a human annotated them as carrying no evidence for that
claim. They appear to be genuine labelled negatives, and 304 of them ship in the training split
— which would remove the need for hard-negative mining in Stage 3 entirely.

**Unverified:** whether the SciFact annotation process guarantees this, or whether "no evidence
annotated" can also mean "not examined." Requires reading Wadden et al. 2020 on how NEI claims
were constructed, **not** inferring it from the files.

**Blocks:** Stage 3 training data construction. Getting this wrong springs exactly the trap the
plan names.

---

## OQ-001 resolution (2026-09-19)

Answered by `src/scifact/eval/decomposition.py` and written up in
`docs/concepts/06-stage-2-failure-decomposition.md`.

Three conditions, varying one source of error at a time: oracle sentences, oracle abstract,
retrieved@k. `A - B` is the cost of sentence selection, `B - C` the cost of document retrieval.
Conditions A and B are defined only for the 505 evidence-bearing claims, because NOT_ENOUGH_INFO
has no gold rationale to hand over and substituting `cited_doc_ids` would leak the answer.

The first run immediately earned its keep: the decomposition costs came out **negative** for the
lexical verifier (27.5% with oracle sentences, 42.4% with retrieved evidence). A single run over
retrieved evidence would have reported 42.4% and hidden the pathology entirely.

**A negative decomposition cost is diagnostic, not a bug:** it means the model is not responding
to evidence quality at all.

## OQ-008 — Does a cross-encoder actually fix the polarity failure? — OPEN

**Raised:** 2026-09-19 (Claude), from the Stage 2 result.

The lexical verifier scores 11-16% recall on CONTRADICT. The diagnosis is representational:
"treatment reduced mortality" and "treatment did not reduce mortality" share nearly every
content word and every cue word, so a bag of words cannot separate them.

That diagnosis predicts a cross-encoder should help substantially, since attention over the pair
can represent negation scope. **It is a prediction, not a result.** Contamination is a live
confound: SciFact has been public since 2020, so a pretrained model may be recalling rather than
reasoning.

Stage 3 must report CONTRADICT recall specifically, not just overall accuracy -- overall accuracy
can improve while the actual failure is untouched, because SUPPORT dominates.
