# Evaluation protocol

> **Author: Mounir. Claude must not write this file.**
>
> If you are an agent reading this in a later session: this document is authoritative over
> every measurement in the repository. You may *implement* what it specifies, *challenge* it,
> or *point out that it is silent* on something you need. You may not fill it in.

**Status: not yet written.**

This file is deliberately close to empty. A pre-structured template would quietly hand over
most of the design by fixing which questions get asked and in what order, which is the part
that matters.

---

## Gate

Nothing in Stage 1 gets built before this file exists, because the evaluation harness is
shaped by the protocol and not the other way round. Building the harness first and writing
the protocol to match it is how a project ends up measuring what was convenient to measure.

Prerequisites, in order:

1. Stage 0 — read real claims, abstracts and evidence annotations by hand. No code.
2. Checkpoint 1 teaching — valid splits, dev/test separation, what leakage looks like in a
   retrieval task specifically.
3. The actual claim and corpus counts from the downloaded data. `OQ-002` cannot be decided
   without them.

## The bar

Written well enough that a stranger with the repository, and no access to any conversation,
could reproduce every number in it and get the same answer — and could tell, from this file
alone, whether a given result is allowed to be called held-out.

## Open questions this file will have to settle

Tracked in `docs/OPEN_QUESTIONS.md`: OQ-001, OQ-002, OQ-003. They are recorded there as
questions, not as decisions, and they stay questions until Mounir answers them.
