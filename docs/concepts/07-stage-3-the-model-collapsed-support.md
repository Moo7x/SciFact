# Stage 3 — The cross-encoder fixed CONTRADICT and destroyed SUPPORT

*Measured 2026-09-19 on `train_tune` (162 claims, 101 with evidence). Model:
`cross-encoder/ms-marco-MiniLM-L-6-v2`, 22.7M params, 3 epochs, 414 steps, CPU.*

---

## 1. Training was healthy

Train loss 1.07 → 0.79, tune loss 1.07 → 0.80, best at step 350 then flat. No divergence.
`ln(3) = 1.0986` is the random-guess anchor and the run left it behind by step 70.

**Three epochs was right.** The tune curve flattened and never rose, so the model was not
overfitting by the end — which on 2,196 examples against 22.7M parameters is mildly surprising
and probably says the task is hard rather than that the model is well-regularised.

## 2. The prediction that was confirmed

OQ-008 predicted a cross-encoder would fix the CONTRADICT failure, because the Stage 2 diagnosis
was representational: a bag of words cannot separate "reduced mortality" from "did not reduce
mortality", but attention over the pair can.

| | CONTRADICT recall |
|---|---|
| Stage 2 lexical verifier | 11–16% |
| cross-encoder | **84–88%** |

Confirmed, and by a wide margin.

## 3. The prediction that was wrong

Pair-level confusion, n=547:

```
gold \ predicted     SUPPORT  CONTRADICT    NEI   recall
SUPPORT                   27          73     25   21.6%
CONTRADICT                 9          51      4   79.7%
NOT_ENOUGH_INFO           48          22    288   80.4%
```

**73 of 125 SUPPORT pairs are predicted CONTRADICT.** I expected SUPPORT to collapse into NEI if
anything went wrong. It collapsed into CONTRADICT instead. CONTRADICT precision is 51/146 = 35%:
the model shouts CONTRADICT.

## 4. The threshold sweep, which made it much clearer

Claim level, retrieved@3, all 162 claims. τ is the confidence the most-decisive sentence must
reach before the model commits to a direction instead of abstaining.

```
  tau   accuracy   macro-rec    SUPPORT  CONTRADICT       NEI
  0.00     39.5%      47.3%      20.9%       88.2%      32.8%
  0.50     46.9%      51.1%       0.0%       64.7%      88.5%
  0.70     37.7%      33.3%       0.0%        0.0%     100.0%
  0.80+    37.7%      33.3%       0.0%        0.0%     100.0%

  majority baseline:  accuracy 41.4%,  macro-recall 33.3%
```

Two things fall out, and neither is visible in an accuracy number.

**The model is systematically under-confident.** At τ = 0.70 it abstains on *everything*. Not
one sentence in the entire evaluation reaches 0.70 confidence on a directional call. Its
probability mass never concentrates.

**SUPPORT is gone.** At τ = 0.50 — any confidence threshold at all — SUPPORT recall is exactly
**0.0%**. The model never confidently predicts SUPPORT. It has effectively learned a two-class
problem, CONTRADICT versus NOT_ENOUGH_INFO, and dropped the third class entirely.

## 5. Two causes, and the second is more likely mine

**Cause A: class weighting overshot.** Weights were SUPPORT 1.59, CONTRADICT 1.90, NEI 0.54.
CONTRADICT is the rarest class so it got the largest weight — which was the intent, to stop the
model ignoring it. It stopped ignoring it and started over-claiming it.

**Cause B: the negative sampling drowns SUPPORT — and this is a design flaw in `build_pairs`.**

For an evidence-bearing claim the builder emits:

- one positive: (claim, gold rationale sentence) → SUPPORT
- **two** negatives: (claim, other sentences **from the same abstract**) → NEI

So every SUPPORT example is accompanied by two near-identical examples — same claim, same
abstract, same topic, adjacent prose — labelled the opposite way, at a 2:1 ratio against it.
SUPPORT is being squeezed from both sides: hard to separate from NEI because the negatives come
from the same paragraph, and outweighed on the other side by a CONTRADICT class carrying more
loss weight.

CONTRADICT does not suffer this. A negated claim against its abstract has an explicit polarity
flip, which is a strong, learnable signal that no same-abstract negative mimics.

## 6. The honest bottom line

| condition | accuracy | macro-recall |
|---|---|---|
| majority (always SUPPORT), all claims | **41.4%** | 33.3% |
| cross-encoder, τ=0.5, all claims | 46.9% | **51.1%** |
| majority, evidence-bearing only | **66.3%** | 50.0% |
| cross-encoder, oracle sentences | 44.6% | 53.9% |

**On accuracy the fine-tuned model loses to a constant predictor on evidence-bearing claims**
(44.6% vs 66.3%). On macro-recall it wins, but at n=101–162 the interval is roughly ±8–10pp and
the evidence-bearing margin is 3.9pp — **not distinguishable from noise**.

So the defensible claim after Stage 3 is narrow: *fine-tuning a small cross-encoder solved the
polarity failure that defeated the lexical baseline, and introduced a new failure on SUPPORT.
It is not yet better than guessing.*

## 7. Two lessons worth more than the model

**Accuracy on imbalanced classes rewards the degenerate predictor.** "66.9% pair accuracy" was
the first number this model produced, against a 65.4% majority floor. It looked like a result.
Macro-recall showed the model had abandoned a class. Macro-recall is now required by
`docs/EVALUATION.md`.

**The aggregation rule was a second selection-bias bug, and I wrote it days after teaching the
concept.** Scoring ~24 candidate sentences and taking the most-decisive one is
`E[max of m noisy draws]` — Checkpoint 1's formula — applied to sentences instead of models.
With 24 draws, something always looks confident. Measured cost: identical model, NEI recall
**80.4% on single pairs, 32.8% at claim level.** Nothing changed but the combination rule.

The general form: *any time you take a maximum over many candidates, you have imported
selection bias, whether or not the candidates are models.*
