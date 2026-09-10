# SciFact — Evidence Retrieval

Retrieve evidence for scientific claims from a fixed corpus of abstracts, identify which
sentences support or contradict each claim, and return an evidence-linked verdict — or
explicitly abstain when the evidence is insufficient.

**Research question:** when does iterative, adaptive evidence retrieval improve grounded
decisions enough to justify its extra cost and latency, compared to a fixed pipeline?

Dataset: [SciFact](https://github.com/allenai/scifact) (Wadden et al., 2020).
This is an implementation-and-comparison study over a public benchmark, not a novel method.
Prior work on SciFact is extensive and is cited rather than ignored.

## Status

Stage 0 — scaffolding. Nothing measured yet.

**No number appears in this repository until a run produces it.**

## Where to start

| If you are | Read |
|---|---|
| A returning session (human or agent) | [`docs/STATE.md`](docs/STATE.md) |
| New to the project | [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) |
| Looking for the evaluation rules | [`docs/EVALUATION.md`](docs/EVALUATION.md) |
| Looking for the code layout | [`docs/MAP.md`](docs/MAP.md) |
