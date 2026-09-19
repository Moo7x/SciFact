# Stage 2 — Decomposing failure, and a verifier that fails backwards

*Measured 2026-09-19 on `train` (809 claims). Reproduce: `python scripts/run_stage2.py`.*

---

## 1. Why one run cannot answer the question

The project plan's Stage 2 asked to measure retrieval failure and reasoning failure
independently. From a single run over retrieved evidence that is impossible: a wrong verdict is
equally consistent with *"the evidence was never retrieved"* and *"the evidence was retrieved and
misread."* One signal, two hypotheses. This was logged as OQ-001 before Stage 1 was built,
because it changes what the Stage 1 harness has to support.

The fix is to vary one thing at a time:

| Condition | Evidence shown to the verifier | Errors still possible |
|---|---|---|
| **A. oracle sentences** | the gold rationale sentences | reasoning |
| **B. oracle abstract** | every sentence of the gold abstract | reasoning + sentence selection |
| **C. retrieved@k** | every sentence of BM25's top-k | reasoning + selection + retrieval |

Then `A − B` is the cost of finding the sentence, and `B − C` the cost of finding the abstract.

**An asymmetry worth stating rather than hiding.** A and B exist only for claims that *have*
evidence. For a NOT_ENOUGH_INFO claim there is no gold rationale: "perfect evidence" means
"nothing in the corpus settles this," which is not a set of sentences. Substituting its
`cited_doc_ids` would leak the answer key. So A and B are scored on the 505 SUPPORT/CONTRADICT
claims, and C is scored twice — once on that same subset for a like-for-like comparison, once on
all 809, which is the only number describing the real system.

## 2. The result

```
STAGE 2 - FAILURE DECOMPOSITION: lexical (nei_threshold=0.45)   split=train

  condition                                            accuracy       n
  A. oracle sentences                      27.5% [23.6%, 31.5%]     505
  B. oracle abstract                       35.4% [31.3%, 39.6%]     505
  C. retrieved@3 (evidence-bearing only)   42.4% [38.0%, 46.7%]     505
  C. retrieved@3 (all claims)              52.8% [49.3%, 56.2%]     809

  Decomposition (evidence-bearing claims):
    reasoning error, with perfect evidence        72.5%
    + cost of selecting the sentence              -7.9%
    + cost of retrieving the abstract             -6.9%
    = total pipeline error                        57.6%
```

**The costs are negative.** The verifier gets *more accurate as the evidence gets worse*:
27.5% given exactly the right sentence, 42.4% given three whole abstracts retrieved by BM25.

A decomposition whose intermediate costs come out negative is not a broken measurement. It is
the measurement working: **a negative cost means the model is not using evidence quality at
all**, and is instead exploiting something that happens to correlate with having more text.

## 3. What it is actually doing

The lexical verifier abstains when no candidate sentence shares at least 45% of the claim's
content words. Hand it one sentence and that bar is often missed, so it predicts NOT_ENOUGH_INFO
— which, in conditions A and B, is *always wrong* by construction. Hand it three abstracts
(~24 sentences) and something always clears the bar, so it predicts SUPPORT — correct for 66%
of evidence-bearing claims, since SUPPORT is the majority.

It is not a verifier. It is a *did-I-find-enough-words* detector, and more text means more words.

The per-class recalls confirm it:

```
condition                                SUPPORT  CONTRADICT  NOT_ENOUGH_INFO
A. oracle sentences                        36.1%      11.0%             n/a
C. retrieved@3 (all claims)                56.0%      16.2%           70.1%
```

**CONTRADICT recall is 11–16%.** The polarity heuristic — comparing negation and direction cue
words between claim and evidence — barely functions. That is the finding that matters, and it
was predictable: "treatment reduced mortality" and "treatment did not reduce mortality" share
almost every content word and every cue word. Polarity in scientific prose lives in syntax and
scope, not in vocabulary, and a bag of words has thrown that away before the comparison starts.

## 4. It loses to doing nothing

```
verifier                          accuracy on evidence-bearing claims
majority (always SUPPORT)                        65.7%
lexical, oracle sentences                        27.5%
```

Given *perfect* evidence, the lexical verifier scores **less than half** what a constant
prediction scores. Reported plainly because the plan requires it: a result where the cheaper
method wins is a real result.

This is not a reason to tune the thresholds. It is the argument for Stage 3. The failure is in
the representation — bag-of-words cannot encode "A caused B" versus "A did not cause B" — and no
threshold fixes a representation.

## 5. The abstention trade, visible early

```
NEI threshold sweep, condition C (retrieved@3, all 809 claims)

  threshold    accuracy   NEI recall   S/C recall
       0.00       43.6%         0.0%        58.4%
       0.30       51.2%        28.6%        54.4%
       0.45       52.8%        70.1%        36.1%
       0.60       47.3%        93.1%        17.0%
       0.90       38.4%       100.0%         1.1%
```

The two recall columns move in opposite directions across the whole range. There is no setting
that is good at both; every point is a choice about which error you prefer. That trade is the
Stage 5 calibration problem, already fully visible with a trivial model.

## 6. A methodological bug worth keeping

The threshold was first swept on condition A — which contains **no NOT_ENOUGH_INFO claims at
all**. There, abstaining can only ever be wrong, so the sweep dutifully selected "never abstain"
(threshold 0.0, the highest score) and reported it as optimal.

**A parameter must be tuned on a set where it can be right.** The sweep was not broken; it
answered exactly the question it was asked, and the question was wrong. This class of error —
an optimisation that succeeds against a mis-specified objective — produces confident, plausible,
useless numbers, and nothing in the output announces it.
