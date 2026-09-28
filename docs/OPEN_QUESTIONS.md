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

## OQ-007 — Are the NEI cited documents usable as guaranteed hard negatives? — ANSWERED 2026-09-19: YES

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

## OQ-008 — Does a cross-encoder actually fix the polarity failure? — ANSWERED 2026-09-19: YES, and it broke SUPPORT

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

---

## OQ-007 resolution (2026-09-19) — confirmed by reading the paper

Wadden et al. 2020, section 3.3, read directly rather than inferred from the files:

> "For each claim, all of the claim's cited abstracts are annotated for evidence. Annotators are
> shown a single claim - cited abstract pair, and asked to label the pair as SUPPORTS, REFUTES,
> or NOINFO." ... "Overall, the annotators found evidence in 63% of cited abstracts."

**A human read every cited abstract and judged it.** NOINFO means "examined and found to contain
no evidence", not "not examined". These are genuine labelled negatives.

The paper's own baseline uses them the same way (section 5):

> "For each claim, we use cited abstracts labeled NOINFO, as well as non-rationale sentences from
> abstracts labeled SUPPORTS and REFUTES as negative examples."

**Cleared for use as hard negatives in Stage 3.** The plan's trap — unannotated retrieved
documents are not guaranteed negatives — still applies to everything else the retriever surfaces.

### Two things found while reading that were not asked about

**1. CONTRADICT claims are negations of SUPPORT claims** (section 3.2):

> "To obtain examples where an abstract REFUTES a claim, an NLP expert wrote negations of
> existing claims, taking precautions not to bias the negations by using obvious keywords like
> not."

This explains the Stage 2 result exactly. CONTRADICT recall was 11-16% for the lexical verifier
because a negated claim shares almost every content word with its original **by construction**.
The dataset was deliberately built so that lexical overlap cannot solve it. That was not a
weakness of the heuristic; it was the dataset working as designed.

**2. The corpus contains deliberate distractors** (section 3.1):

> "we identify five papers cited in the same paper as each source citance but in a different
> paragraph, and add these to the corpus as distractor abstracts."

Which explains OQ-006: NEI abstracts are topically adjacent because topically adjacent abstracts
were added on purpose.

## OQ-009 — Negation twins are split across train and dev — OPEN, and serious

**Raised:** 2026-09-19 (Claude), following from the negation procedure above.

If an expert wrote negations of existing claims, the original and its negation can land in
different splits. Measured by claim-token Jaccard between every dev claim and every train claim:

| Jaccard threshold | dev claims with a train twin |
|---|---|
| >= 0.6 | 116 / 300 (38.7%) |
| >= 0.7 | 91 / 300 (30.3%) |
| >= 0.8 | 69 / 300 (23.0%) |
| >= 0.9 | 14 / 300 (4.7%) |

**Every example inspected cites the same documents as its twin.** Real cases:

```
jaccard=0.80  DEV   [NEI]        A deficiency of vitamin B12 increases blood levels of homocysteine.
              TRAIN [NEI]        A deficiency of vitamin B12 decreases blood levels of homocysteine.

jaccard=0.71  DEV   [CONTRADICT] A high microerythrocyte count raises vulnerability to severe anemia...
              TRAIN [SUPPORT]    A high microerythrocyte count protects against severe anemia...
```

Same abstract, one word different, opposite labels, opposite sides of the split.

**Why it matters.** A cross-encoder is trained on (claim, abstract) pairs. For roughly a quarter
of dev, it has already seen that exact abstract during training, paired with a nearly identical
claim. That is textbook near-duplicate contamination, and it sits inside the split this project
designated as its only clean holdout (see `docs/EVALUATION.md` section 1).

**Two readings, and it is not obvious which dominates:**

1. *Contamination.* The model may have memorised the abstract, so dev overstates generalisation.
2. *Deliberate probe.* Splitting the pair forces the model to attend to the polarity word rather
   than memorise the abstract — arguably the hardest and most informative test in the dataset.

**Required, either way:** every dev result must be reported **twice** — on twinned and untwinned
claims separately. A large gap quantifies reading 1; no gap supports reading 2. Reporting only
the pooled number makes the question unanswerable.

Not yet done. Blocks any dev evaluation.

---

## OQ-008 resolution (2026-09-19) — yes for CONTRADICT, at the cost of SUPPORT

CONTRADICT recall 11-16% (lexical) -> 84-88% (cross-encoder). The representational diagnosis was
right: attention over the pair can encode negation scope where a bag of words cannot.

But SUPPORT recall collapsed to 21.6%, and to **0.0%** at any confidence threshold above 0.5.
73 of 125 SUPPORT pairs are predicted CONTRADICT. The model effectively learned two classes.

Full write-up: `docs/concepts/07-stage-3-the-model-collapsed-support.md`.

## OQ-010 — Which caused the SUPPORT collapse: class weighting or negative sampling? — OPEN

**Raised:** 2026-09-19 (Claude). Both causes are mine, and they are separable by experiment.

**Cause A — class weighting overshot.** Weights SUPPORT 1.59, CONTRADICT 1.90, NEI 0.54.
CONTRADICT is rarest so it got the largest weight; the model went from ignoring it to
over-claiming it. CONTRADICT precision is 35%.

**Cause B — negative sampling drowns SUPPORT.** `build_pairs` emits, per evidence-bearing claim,
one positive (the gold rationale sentence) and **two** negatives drawn from *the same abstract*.
Every SUPPORT example therefore ships with two near-identical counterexamples -- same claim,
same abstract, adjacent prose -- labelled NEI, 2:1 against it. CONTRADICT does not suffer this,
because a negated claim carries an explicit polarity flip that no same-abstract negative mimics.

**The experiment**, two runs, changing one thing each:

1. `--no-class-weights` -- isolates cause A.
2. `--max-negatives 1` -- isolates cause B (requires exposing that argument).

Two configurations, not twenty-five: at sigma ~= 4pp on n=162, selection inflation at m=2 is
about 1.6pp, small enough that a real effect should survive it.

**Prediction on record, so it can be wrong:** cause B dominates. Weighting shifts a decision
boundary, which a threshold can partly undo; a 2:1 ratio of near-identical contradictory
training signal is a harder thing for the model to recover from.

## OQ-011 — Is the model's under-confidence a calibration problem or a capacity problem? — OPEN

**Raised:** 2026-09-19 (Claude).

At tau = 0.70 the model abstains on **everything**: not one sentence across 162 claims reaches
0.70 confidence on a directional call. Its probability mass never concentrates.

Two readings, and they need different fixes:

1. **Calibration.** Weighted cross-entropy distorts the output distribution away from the true
   priors -- a cost recorded when the weighting was added. Temperature scaling on `train_tune`
   would test this cheaply.
2. **Capacity or genuine difficulty.** 22.7M parameters on 2,196 examples of a task requiring
   scientific-negation reasoning. The model may be correctly uncertain.

Distinguishing them matters for Stage 5: a threshold cannot be "the calibrated operating point"
if the scores underneath it are not calibrated at all.

## OQ-012 — Does the web-trained vocabulary hurt on biomedical text? — PARTLY ANSWERED 2026-09-28

**Raised:** 2026-09-27 (Claude), from Lesson 1.

The cross-encoder's tokenizer was built from MS MARCO (web search) text. On SciFact it shatters
domain terms: `microerythrocyte` -> 6 pieces, `homozygous` -> 4, `thalassemia` -> 3. Acronyms
lose their link to what they stand for: `SMA` (severe malarial anaemia) -> `sm ##a`, sharing no
token with `anemia` -> `an ##emia`.

**Hypothesis, not a result:** a vocabulary built from biomedical text keeps more domain terms
whole, and a model pretrained on it would connect acronyms and paraphrases better. This could be
part of why SUPPORT (paraphrase-heavy) collapsed while CONTRADICT (near-identical wording,
polarity flipped) did not.

**Measurable cheaply before any training:** mean tokens-per-word on SciFact claims under each
tokenizer. If the biomedical tokenizer does not fragment noticeably less, the hypothesis dies
there, for free.

Not pursued now -- building is paused for the lesson track.

---

## OQ-010 resolution (2026-09-27) — my prediction was wrong; the real finding is a see-saw

GPU runs, same seed, tune set identical across arms (547 pairs: 125 SUPPORT, 64 CONTRADICT,
358 NEI). Best checkpoint per arm:

| arm | config | SUPPORT | CONTRADICT | NEI | macro-recall |
|---|---|---|---|---|---|
| baseline | weights ON, 2 neg | 22.4% | 79.7% | 79.9% | 60.7% |
| no_weights | weights OFF, 2 neg | **60.8%** | **20.3%** | 89.7% | 56.9% |
| one_neg | weights ON, 1 neg | 14.4% | 85.9% | 74.3% | 58.2% |

**The prediction on record was "one_neg moves SUPPORT more than no_weights". Wrong.** Cutting
negatives made SUPPORT *worse* (22% -> 14%). Removing class weights moved it most (22% -> 61%).

**But removing weights did not fix anything. It moved the failure.** CONTRADICT fell from 80%
to 20% in the same run. Macro-recall is 57-61% in all three arms -- within noise of each other at
these class sizes. So class weighting does not make the model better or worse overall. **It
decides which of SUPPORT or CONTRADICT the model sacrifices.**

The reading this points to: the model does not reliably tell SUPPORT from CONTRADICT, and where it
cannot, a prior decides -- and the class weights set that prior. That is consistent with how the
data was built: every CONTRADICT claim is a negation of a SUPPORT claim, scored against the same
evidence, so separating them requires detecting the negation itself, and nothing else helps.

### Two flaws in my own experiment design, stated plainly

1. **Arm B was not a single-variable change.** Class weights are computed from class counts,
   and halving the negatives changes the counts. CON/SUP weight ratio stayed at 1.77 (the
   pressure I suspected), but NEI/SUP doubled from 0.34 to 0.69. Arm A is clean; arm B's
   interpretation carries that confound.
2. **Tune-loss values are not comparable across arms.** `no_weights` reports 0.588 against
   `baseline`'s 0.798 because one is unweighted cross-entropy and the other weighted -- different
   functions, not better and worse models. Only the recalls compare. "Best checkpoint" is also
   chosen by a different criterion in each arm for the same reason.

## OQ-013 — Does the model tell a claim from its own negation? — OPEN

**Raised:** 2026-09-27 (Claude), from OQ-010.

The sharpest possible test of the see-saw reading. For each SUPPORT/CONTRADICT twin pair that
shares evidence (OQ-009 found many), check whether the model gives the two claims *different*
predictions. If it predicts the same label for a claim and its negation, it is not reading the
negation at all -- and no loss weighting can fix that.

Cheap: inference only, no training. Deferred until the lesson track is done.

## OQ-012 update (2026-09-28) — hypothesis survived its cheap test

Tokens per word on 40,679 words of train text: ms-marco-MiniLM (the Stage 3 model) **1.68, worst
of five**; PubMedBERT **1.41, best**; deberta-v3 1.46; SciBERT 1.47. PubMedBERT keeps
`erythrocyte`, `homozygous` and `thalassemia` whole where the chosen tokenizer uses 5, 4 and 3
pieces. Fragmentation is confirmed. Whether it *causes* worse verification is not yet separated
from the head and pretraining differences; see ADR-0002.
