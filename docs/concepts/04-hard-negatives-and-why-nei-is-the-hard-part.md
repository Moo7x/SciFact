# Hard negatives, and why NOT_ENOUGH_INFO is the hard part

*Measured 2026-09-17 on the training split via `scripts/analyze_negatives.py`.*

Started from an observation Mounir made reading NOT_ENOUGH_INFO examples by hand:

> *"the abstract shares key words present in the claim yet its not talking about it
> specifically or trying to deny or verify even — sharing keywords but talking about
> other topic than the one in the claim"*

The hypothesis turned out **half right**, and the half that was wrong is the more useful half.

---

## The measurement

How much of a claim's vocabulary appears in the abstract, under three conditions?

```
1. has evidence              n= 564   mean=62%   p25=50%  p50=64%  p75=75%
2. NOT_ENOUGH_INFO (cited)   n= 330   mean=32%   p25=18%  p50=33%  p75=45%
3. random abstract           n= 809   mean= 5%   p25= 0%  p50= 0%  p75= 9%
```

And where the cited abstract lands when all 5,183 are ranked by raw overlap:

```
has evidence       median rank=1   top-1=67.6%  top-10=88.4%  top-100=98.4%
NOT_ENOUGH_INFO    median rank=6   top-1=24.8%  top-10=58.4%  top-100=76.8%
```

## What was right

**NOT_ENOUGH_INFO abstracts are nothing like random text.** Mean coverage 32% against 5% for
random, whose *median is zero*. They genuinely do share the claim's vocabulary.

And they will be retrieved: **58% of them land in the top 10** of a purely lexical ranking, a
quarter of them at rank 1. A retriever will hand these to the classifier constantly. They are
hard negatives in the technical sense — high-scoring, and wrong.

## What was wrong, and why it matters more

They are **not** lexically indistinguishable from real evidence. NOT_ENOUGH_INFO sits about
**47% of the way** from random to evidence, not at 100%. There is a real 30-point gap in mean
coverage, and the retrieval ranks separate too (top-1: 68% vs 25%).

So the picture is not "impossible to tell apart." It is worse, in a more interesting way:

```
best single-threshold accuracy   78.3%  (at coverage >= 45%)
AUC                              0.858
majority-class floor             63.1%
```

**AUC 0.858** means: pick one evidence-bearing abstract and one NOT_ENOUGH_INFO abstract at
random, and crude word overlap ranks them correctly 86% of the time. Far better than chance
(0.5), nowhere near reliable (1.0).

Look at where the distributions collide: evidence `p25 = 50%`, NEI `p75 = 45%`. A quarter of
genuine evidence scores *below* where a quarter of the negatives score *above*. **Every
threshold you can pick trades false accepts against false abstentions.** There is no value that
cleanly separates them, because no such value exists in this signal.

That is the Stage 5 calibration problem, visible before any model has been trained.

---

## Four design consequences

### 1. Retrieval score is a poor abstention signal

The tempting design is *"if the top retrieval score is low, abstain."* The data says this is
weak: negatives score high by construction. It is not *useless* — AUC 0.858 is real signal —
but a system abstaining on retrieval score alone inherits an 86%-reliable discriminator as its
ceiling, and would call that "calibration."

### 2. This is a reasoning problem wearing a retrieval problem's clothes

Retrieval finds the NEI abstract easily. The mistake happens afterwards, when something must
decide that a topically-related abstract does not actually settle the claim.

This is the concrete form of **OQ-001**: Stage 2 must separate retrieval failure from reasoning
failure, and here is a whole class of error that is *purely* reasoning. Tuning retrieval will
not move it.

### 3. SciFact ships free hard negatives — take them

Training a reranker normally requires **hard negative mining**: a sub-problem of its own, where
you sample high-scoring wrong answers and hope you have not accidentally sampled unlabelled
true positives.

SciFact hands you 304 human-curated hard negatives in the training split. They are *annotated*
as insufficient, which is exactly the guarantee mining cannot give you.

This connects directly to the trap in the project plan:

> *"Unannotated retrieved documents are not guaranteed negatives — they may be relevant but
> unlabelled."*

The NEI cited documents **are** guaranteed negatives. Anything else the retriever surfaces is
not. That distinction has to survive into the training code, or the trap gets sprung quietly.

### 4. Abstention is a third class, not a fallback

37.6% of training claims are NOT_ENOUGH_INFO, and those abstracts look topically plausible.
A system that treats abstention as "what I do when confused" will abstain on hard SUPPORT cases
and confidently answer on well-written NEI cases — failing in both directions at once.

---

## Method note, so these numbers are not over-read

The ranking here uses **raw content-word overlap**: no IDF weighting, no length normalisation,
no stemming. It is a diagnostic, not BM25, and must never be quoted as retrieval performance.

Real BM25 weights rare terms heavily, so it may well separate these groups *better* — a claim
and its true evidence tend to share specific technical terms, while a topically-adjacent
abstract shares the general ones. Whether that holds is a Stage 1 measurement, not an assumption.
