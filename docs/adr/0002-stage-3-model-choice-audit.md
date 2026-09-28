# ADR-0002 — Stage 3 model choice: audited after the fact

- **Date:** 2026-09-28
- **Status:** Proposed — the choice of whether to switch models belongs to Mounir
- **Author:** Claude

## Context

Stage 3 fine-tuned `cross-encoder/ms-marco-MiniLM-L-6-v2`. **That model was chosen from memory,
with no survey of alternatives and no model cards read.** That broke the plan's
design-before-code rule, and it only surfaced when Mounir asked for the choice to be defended.
This record is the survey, done afterwards and labelled as such.

Measured by `scripts/compare_models.py` on this project's own data: the 547 `train_tune` pairs
behind the Stage 3 confusion matrix, all train claims and rationales, and an RTX 3050 Ti Laptop
with 4 GiB.

## Candidates

Surveyed on the Hugging Face Hub, 2026-09-28:

| model | params | trained for | head |
|---|---|---|---|
| **ms-marco-MiniLM-L-6-v2** (chosen) | 23M | web search relevance (MS MARCO) | 1 output: relevance score |
| nli-MiniLM2-L6-H768 | 82M | NLI (MNLI + SNLI) | contradiction / entailment / neutral |
| nli-deberta-v3-xsmall | 71M | NLI (MNLI + SNLI) | same |
| nli-deberta-v3-small | 142M | NLI (MNLI + SNLI) | same |
| PubMedBERT-MNLI-MedNLI | 109M | PubMed pretraining, then NLI | same |

**Rejected on principle, before measuring:** every Hub model already fine-tuned on SciFact
(15+ exist). They were trained on the dataset under evaluation, possibly including our `dev`
holdout.

## Evidence

### 1. Head fit — zero-shot, no SciFact training at all

NLI's `entailment / contradiction / neutral` maps directly onto `SUPPORT / CONTRADICT / NEI`.
Input order follows NLI convention: (evidence, claim).

| model | accuracy | SUPPORT | CONTRADICT | NEI | macro |
|---|---|---|---|---|---|
| majority (always NEI) | 65.4% | 0% | 0% | 100% | 33.3% |
| **our fine-tuned ms-marco** | — | 22.4% | 79.7% | 79.9% | 60.7% |
| nli-MiniLM2-L6-H768 | 67.5% [63.6, 71.3] | 22.4% | 53.1% | 85.8% | 53.8% |
| nli-deberta-v3-xsmall | 62.3% [58.1, 66.4] | 28.0% | 60.9% | 74.6% | 54.5% |
| nli-deberta-v3-small | 65.8% [61.6, 69.8] | 32.8% | 51.6% | 79.9% | 54.8% |
| **PubMedBERT-MNLI-MedNLI** | 56.3% [51.9, 60.1] | **80.0%** | **73.4%** | 45.0% | **66.1%** |

The general-domain NLI models land below our fine-tuned model on macro-recall. **With no SciFact
training at all, the biomedical NLI model scores above it.** It is also the only model measured
that gets **both** directional classes above 70%. The SUPPORT/CONTRADICT see-saw from OQ-010 does
not appear. Its weakness is the opposite one: it over-commits and recognises only 45% of NEI.

The macro gap (66.1 vs 60.7) is inside the likely interval at these class sizes (125 / 64 / 358).
**The qualitative difference is the stronger evidence**: both directional classes are high at
once, which none of our three trained arms achieved.

### 2. Vocabulary fit — OQ-012

Tokens per word over 1,834 train texts (40,679 words). The last four columns give how many
pieces each term becomes mid-sentence:

| tokenizer | tokens/word | erythrocyte | homozygous | thalassemia |
|---|---|---|---|---|
| **ms-marco-MiniLM (chosen)** | **1.68 (worst)** | 5 | 4 | 3 |
| nli-MiniLM2 (RoBERTa BPE) | 1.63 | 5 | 3 | 4 |
| nli-deberta-v3 | 1.46 | 1 | 1 | 2 |
| SciBERT | 1.47 | 1 | 1 | 3 |
| **PubMedBERT** | **1.41 (best)** | 1 | 1 | 1 |

The chosen tokenizer fragments SciFact text the most of the five. The OQ-012 hypothesis survives
its cheap test.

### 3. Hardware fit — one real AdamW step, batch 16, length 256, fp32

| model | peak memory | ms / step |
|---|---|---|
| **ms-marco-MiniLM (chosen)** | **0.93 GiB** | **177** |
| nli-MiniLM2-L6-H768 | 2.25 GiB | 552 |
| PubMedBERT-MNLI-MedNLI | 3.75 GiB | 6,033 |
| nli-deberta-v3-xsmall | 4.18 GiB | 4,414 |
| nli-deberta-v3-small | 5.00 GiB | 10,734 |

**The larger models did not crash; they slowed down 25-60x.** On Windows the NVIDIA driver
(since 536.40; this machine runs 546.80) can spill GPU memory into system RAM rather than raise
an out-of-memory error. The CUDA context and the desktop also take VRAM that
`max_memory_allocated` does not count. So "it ran" is not "it fits". A step that silently takes
6 s instead of 0.2 s is the symptom.

These figures are worst-case: padding fixed at 256, fp32. Dynamic padding (Lesson 2: mean batch
width 127) and mixed precision would both cut activation memory substantially. Whether
PubMedBERT then fits properly is **untested**.

## What the original choice gets right, and what it gets wrong

**Right, and defensible:** cost. It is 5x lighter and 3-34x faster than every alternative, and
it is the only candidate that trains on this GPU with no memory work at all. For fast iteration
on a 4 GiB card, that is a real reason.

**Wrong, and now measured:**

1. **Head mismatch.** It was trained to output one number (relevance to a query). That head was
   discarded and a 3-way head trained from scratch on 2,196 pairs. Relevance and entailment are
   different tasks: an abstract can be highly relevant to a claim and still contradict it.
2. **Vocabulary mismatch.** It is the worst tokenizer of five on this corpus.

## Consequence for Stage 3's conclusion

Stage 3 reported that the model "cannot reliably tell a claim from its own negation." The audit
shows **that is a finding about this model, not established as a finding about the task.** A
domain- and task-matched model shows no such see-saw without any training. The README has been
updated to say so.

## Caveat on the strongest candidate

PubMedBERT was pretrained on PubMed abstracts, and **SciFact's corpus is PubMed abstracts.** It
has very likely seen our evidence text during pretraining; not the labels, but the passages. That
is a different contamination from label leakage, and it favours this model specifically. Any
comparison involving it must say so.

## Options

| option | cost | what it buys |
|---|---|---|
| A. keep ms-marco, record this audit | none | honest, but Stage 3's conclusion stays model-specific |
| B. fine-tune nli-MiniLM2 (fits: 2.25 GiB) | one run | tests "does an NLI head help?" with nothing else changed that much |
| C. fine-tune PubMedBERT-MNLI-MedNLI | needs dynamic padding + mixed precision first | tests the strongest candidate, under a contamination caveat |

## Recommendation

Do not switch yet. Lessons 2-5 cover exactly what option C needs: dynamic padding, mixed
precision, device memory. Finish them, then run B and C as a controlled Stage 3b comparison.
Mounir decides.

## Revisit trigger

Any Stage 3b run that trains a candidate here under the same protocol.
