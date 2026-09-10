# Architecture Decision Records

> **Author: Mounir.** Claude prompts when one is needed and may critique a draft.
> Claude does not write these. Per the project plan: *"These are my interview answers."*

## What an ADR is for

A code comment says *what* the code does. An ADR says *why this and not the obvious
alternative*, written at the moment the alternative was still live — before hindsight makes
the choice look inevitable. Six months later the code still shows the decision; only the ADR
still shows the reasoning, and the reasoning is what gets interrogated in an interview.

## Rules

- One file per decision: `NNNN-short-kebab-title.md`, numbered sequentially, never renumbered.
- An ADR is only warranted when there was a **real alternative**. If there was only one
  sensible option, it was not a decision and does not need a record.
- ADRs are **immutable once accepted**. A decision that changes gets a *new* ADR that
  supersedes the old one; the old file stays, marked `Superseded by NNNN`. The history of
  reversals is the valuable part — an ADR log with no reversals in it is usually a log
  that has been quietly edited.
- Where possible, state the **revisit trigger**: the observation that would make you change
  your mind. A decision that names its own expiry condition is worth far more than one
  written as though it were permanent.

## Pending

| # | Decision | Status |
|---|---|---|
| 0001 | Dev environment: Windows-native loop with Docker for Linux/CI parity | **awaiting Mounir** |
