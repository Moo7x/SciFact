# What the SciFact data actually looks like

*Measured 2026-09-16 on the **training split only** via `scripts/describe_data.py`.
Reproduce with `python scripts/describe_data.py`.*

This started from three observations Mounir made reading examples by hand. All three turned
out to be real; one was right in direction and wrong in its strong form, which is the most
useful kind of result a measurement can give you.

---

## The observations, and what the numbers say

### 1. "The evidence is on a specific numbered line"

Confirmed, and stronger than it sounds. Evidence is annotated at **sentence** granularity, and
a rationale is almost always **exactly one sentence**:

```
sentences per rationale   p50=1   p75=1   p90=1   p99=2
```

So the design question *"do I retrieve sentences, or retrieve abstracts and then locate
sentences?"* has a data-driven answer available: the unit of truth is a single sentence out of
a median of 8.

### 2. "Related lines get split apart"

Also real, and it has a consequence worth naming. The abstracts arrive **pre-segmented** into
sentences by an automatic splitter, and that splitter is mechanical — it breaks on punctuation,
not on meaning.

98.3% of rationales have contiguous indices, but since the median rationale is one sentence,
most of that contiguity is trivial. The thing to watch for is the opposite case: **if the
segmenter split one logical statement across two lines and only one of them is annotated, a
retriever that finds the annotated sentence still only has half the justification.** The
annotation is correct; the segmentation is what is lossy.

### 3. "The last sentence is the conclusion one"

Right in direction, wrong in the strong form — and the gap between those matters.

```
Where evidence sits in the abstract (0.0 = first sentence, 1.0 = last):
  0.0-0.1  #########...............................    47  ( 4.6%)
  0.1-0.2  ####....................................    21  ( 2.0%)
  0.2-0.3  #######.................................    40  ( 3.9%)
  0.3-0.4  ######..................................    34  ( 3.3%)
  0.4-0.5  ############............................    65  ( 6.3%)
  0.5-0.6  ########################................   134  (13.1%)
  0.6-0.7  #############################...........   162  (15.8%)
  0.7-0.8  #########################...............   140  (13.7%)
  0.8-0.9  #############################...........   162  (15.8%)
  0.9-1.0  ########################################   220  (21.5%)
```

Roughly **80% of evidence sentences sit in the back half** of the abstract. But only **15.4%**
of rationales include the literal final sentence. So: heavily back-weighted, not "always last."

**Why this is dangerous, and not merely interesting.** That skew is a *positional prior*. A
"retriever" that ignores the claim entirely and always returns the last two sentences of each
cited abstract would score far above chance — while performing no retrieval whatsoever.

That makes a positional baseline mandatory, not optional. Without one, a BM25 recall number is
uninterpretable: you cannot tell how much of it came from matching the claim and how much came
from the fact that conclusions live at the end of abstracts. **The purpose of a baseline is to
absorb the credit that does not belong to your method.**

---

## Three more measurements that shape the design

### Abstract length — the cross-encoder budget

```
sentences per abstract   p50=8    p75=11   p90=13   p95=15   p99=20
words per abstract       p50=192  p75=247  p90=295  p95=341  p99=451
longest abstract         1,524 words
```

A BERT-family cross-encoder takes 512 tokens total, shared between claim and abstract. Words
are not tokens: biomedical text runs roughly 1.3–1.6 tokens per word because rare terms get
split into word-pieces. At that rate the median abstract is roughly 250–300 tokens and p95 is
roughly 440–550 — **right at the boundary.**

Those token figures are an *estimate from a conversion factor*, not a measurement. Before any
Stage 3 decision depends on them they must be measured with the actual tokenizer.

### The abstention class is large

```
claims with NO evidence   304 / 809  (37.6%)
rationale labels          SUPPORT 616,  CONTRADICT 341
```

Over a third of training claims have no evidence at all against their cited abstracts. Whatever
"insufficient evidence" ends up meaning in Stage 5, it is not an edge case being handled for
completeness — it is more than a third of the data. Supporting evidence also outnumbers
contradicting roughly 2:1.

### Lexical overlap — BM25's ceiling, visible before BM25 exists

```
% of the claim's content words that also appear in its gold evidence:
  p10=0   p25=15   p50=29   p75=47   p90=65

rationales sharing ZERO content words with their claim:  106 / 957  (11.1%)
```

BM25 matches **words**. For the median claim, fewer than a third of its content words appear in
the sentence that justifies it. And for **11% of rationales there is no shared content word at
all** — nothing for a lexical method to match on, in principle, no matter how it is tuned.

That is a structural ceiling on lexical retrieval, known before writing a line of BM25. It does
not make BM25 a bad baseline — it makes it an *honest* one, and it predicts in advance where
the dense retriever in Stage 3 should earn its keep.

---

## The bug that produced all of this

Every number above was initially **zero**, because both scripts read the evidence with:

```python
rationale.get("sentence_indices", [])   # wrong key
```

The real key is `sentences`. `.get()` with a default turned a typo into an empty list, silently,
for all 957 rationales. The viewer showed 505 annotated claims with nothing marked and never
raised.

**`.get(key, default)` is correct when a key is genuinely optional. It is wrong when the schema
requires the key**, because then a misspelling becomes indistinguishable from a legitimately
absent value.

The fix is `src/scifact/data/schema.py`: a typed loader that raises `SchemaError` naming the
keys it actually found. `tests/test_schema.py` locks the behaviour in, using inline fixtures so
it runs in CI where the corpus is absent.
