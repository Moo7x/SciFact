# Open questions

Questions asked that I could not answer, and decisions not yet made.
Per PART IX of the plan, an unanswered question here is the highest-value signal in the project.

Status key: **OPEN** / **ANSWERED** (with date and where the answer lives) / **DEFERRED** (with trigger)

---

## OQ-001 — Stage 2 cannot decompose failure as written — OPEN

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
