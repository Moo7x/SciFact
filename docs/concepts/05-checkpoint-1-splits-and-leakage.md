# Checkpoint 1 — Valid splits, dev/test separation, and leakage in retrieval

*2026-09-19. Run `python scripts/demonstrate_leakage.py` to reproduce every number here.*

---

## 1. What a split is actually for

You are not trying to measure how your system does on your data. You are trying to measure how
it does on data it has never seen. The split exists to make one an estimate of the other.

Formally: you want the **risk**

$$R(f) = \mathbb{E}_{(x,y)\sim D}\big[L(f(x), y)\big]$$

the expected loss of your system $f$ over the true distribution $D$. You cannot compute it — you
do not have $D$. So you compute the **empirical risk** on a finite sample $S$:

$$\hat{R}_S(f) = \frac{1}{n}\sum_{i=1}^{n} L(f(x_i), y_i)$$

And here is the entire content of this checkpoint, in one line:

> $\hat{R}_S(f)$ is an unbiased estimator of $R(f)$ **if and only if $f$ is independent of $S$.**

The instant you use $S$ to choose anything about $f$, that independence is gone and the estimate
is optimistic. Not "slightly biased in theory" — measurably, quantifiably optimistic, by an
amount computed below.

**The crucial part people miss:** "choose anything about $f$" is far broader than training.
Picking $k$ in recall@$k$. Picking BM25's $b$ and $k_1$. Picking an abstention threshold.
Deciding a dense retriever beats a lexical one. Looking at a number and deciding to try
something else. Every one of those spends the split.

## 2. Why three splits and not two

| Split | Fits | Reusable? |
|---|---|---|
| **train** | model parameters | yes, freely |
| **dev** | your *decisions* — hyperparameters, architecture, thresholds | yes, and every use costs you |
| **test** | nothing. It only estimates. | **once** |

Dev is training data. It trains *you* rather than the weights, and you are a far higher-capacity
optimiser than gradient descent — you can condition on everything you have ever seen about the
data. That is why the test set has to be touched once, at the end, after the protocol is frozen.

## 3. The price of reusing a split, computed

Suppose you evaluate $m$ variants on the same set, and — worst case for intuition — **all of
them are equally good**, with identical true recall $p$. Each observed score is

$$\hat{p}_j = p + \varepsilon_j, \qquad \varepsilon_j \approx \mathcal{N}(0, \sigma^2),
\qquad \sigma^2 = \frac{p(1-p)}{n}$$

You report the best one. But the expected maximum of $m$ zero-mean normals is not zero:

$$\mathbb{E}\Big[\max_j \varepsilon_j\Big] \approx \sigma\sqrt{2\ln m}$$

So the winner's reported score is inflated by roughly $\sigma\sqrt{2\ln m}$ — **entirely from
selecting on noise**, with no variant actually better than any other.

### On this project's numbers

With $p \approx 0.6$ and the dev split's $n = 300$:

$$\sigma = \sqrt{\frac{0.6 \times 0.4}{300}} = 0.0283 = 2.8\text{ percentage points}$$

Simulated over 4,000 runs (`demonstrate_leakage.py`):

```
 variants tried  best observed   inflation    theory
             1         60.0%      +0.0%    +0.0%
             2         61.6%      +1.6%    +3.3%
             5         63.3%      +3.3%    +5.1%
            10         64.3%      +4.3%    +6.1%
            20         65.2%      +5.2%    +6.9%
            50         66.3%      +6.3%    +7.9%
           100         67.0%      +7.0%    +8.6%
```

The `theory` column runs high because $\sigma\sqrt{2\ln m}$ is an upper bound. The refined
asymptotic for the expected maximum is

$$\sigma\left(\sqrt{2\ln m} - \frac{\ln\ln m + \ln 4\pi}{2\sqrt{2\ln m}}\right)$$

At $m = 20$ that gives $0.0283 \times 1.707 = 4.8$pp against a simulated $5.2$pp — which is the
agreement you want before trusting the formula on a case you have not simulated.

**Why this matters more than it looks.** Trying twenty variants — an ordinary amount for a
project with a BM25 baseline, a dense retriever, and a reranker — buys about **5 percentage
points of pure selection noise** on a 300-claim split. If Stage 3's real improvement over Stage 1
is in the 5–15 point range, selection bias is *the same order of magnitude as the entire effect
you are trying to measure*.

## 4. Leakage in a retrieval task specifically

### 4a. Annotation leakage — the one this dataset sets a trap for

`cited_doc_ids` sits in every claim record and looks like ordinary input metadata. It is not.
It is an **annotation artifact**: a record of which abstracts the annotator consulted. The
evidence is inside them.

Measured on train:

```
evidence-bearing docs that are also in cited_doc_ids:  564/564 (100.0%)
documents cited per claim:  mean=1.17   min=1   max=5
```

Every evidence document is in there, and there are barely more than one of them. So restricting
retrieval to `cited_doc_ids` shrinks the candidate set from 5,183 to about 1 — and that 1 is the
answer.

With one identical scoring function and nothing else changed:

```
Top-1 document accuracy, n=200:
  search all 5,183 abstracts         62.5%
  search only cited_doc_ids          99.5%   <- leaked
  difference                        +37.0%
```

**Nothing about the method changed between those rows.** The candidate set was built from the
answer key. The 99.5% is not a better retriever; it is not a retriever.

The general pattern: **a field that exists only because somebody annotated the data cannot be an
input to the system.** It is available at evaluation time and absent in deployment, which is the
definition of leakage. This class is dangerous precisely because such fields look like metadata.

### 4b. What is *not* leakage here

Worth stating, because over-correcting is its own error.

The 5,183-abstract corpus is **shared across train, dev and test by design**. The task is
retrieval over a fixed corpus; withholding documents would change the task. Computing IDF over
the whole corpus is fine — the corpus is legitimately given.

But this means you can never say "the test documents were unseen." They were not. The claims
are held out; the corpus is not.

### 4c. Statistic leakage

Anything *fitted* — an IDF table, a score normalisation, a calibration curve, an abstention
threshold — must be fitted on training data only. Fitting a threshold on dev and then reporting
dev performance at that threshold is exactly the $m$-variants problem above, with $m$ equal to
however many thresholds you scanned. Scanning 100 threshold values is trying 100 variants.

### 4d. Pretraining contamination

SciFact is a public benchmark from 2020. Any pretrained model you use has plausibly seen it.
This is not fixable and it is not fatal — it limits claims about *absolute* performance while
leaving *controlled comparisons* intact, because contamination affects both arms equally. It has
to be stated, not worked around.

## 5. What this means for OQ-002

The three options, with the standard error of a proportion estimate at $p = 0.6$:

| Option | Evaluation $n$ | $\sigma$ | 95% CI half-width |
|---|---|---|---|
| Official dev | 300 | 2.8pp | ±5.5pp |
| Holdout carved from train | ~200 | 3.5pp | ±6.8pp |
| 5-fold CV over train ∪ dev | 1,109 total | — see below | — |

**The CV row cannot be filled in naively.** It is tempting to write $\sigma = \sqrt{0.24/1109} =
1.5$pp, but that assumes 1,109 independent observations. They are not independent: the five fold
models share overlapping training data, so their errors are correlated. Bengio and Grandvalet
(2004), *No Unbiased Estimator of the Variance of K-Fold Cross-Validation*, proves there is no
universally unbiased variance estimator for exactly this reason. The true variance is larger
than the naive formula says, by an amount that depends on the data.

So the honest framing of the trilemma is:

1. **Official dev** — a clean holdout, but ±5.5pp, and it degrades with every decision made on it.
2. **5-fold CV** — the tightest point estimate and the best use of 1,109 claims, but its variance
   is not cleanly estimable, and it dissolves dev into training so **nothing labelled is left
   untouched** (test is unlabelled).
3. **Train-only holdout** — keeps dev pristine as a genuinely independent second check, at the
   cost of training data and a wider interval.

Nothing above chooses between them. The choice depends on what the project needs to be able to
*claim* at the end, which is the protocol question and belongs in `docs/EVALUATION.md`.

## 6. The one-sentence version

**A number is only meaningful if you can say what was decided before it was computed** — and the
only way to be able to say that is to write the protocol down first.
