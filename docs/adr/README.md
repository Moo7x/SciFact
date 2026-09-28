# Architecture Decision Records

> **Author: Claude.** Amended 2026-09-11 at Mounir's request — previously these were his to
> write. They are now written by Claude and explained in chat. Do not ask him to write one.

## What an ADR is for

A code comment says *what* the code does. An ADR says *why this and not the obvious
alternative*, written at the moment the alternative was still live — before hindsight makes
the choice look inevitable.

Six months later, the code still shows the decision. Only the ADR still shows the reasoning.

These exist so a cold session — human or agent — can find out why the repository is shaped
the way it is without re-deriving it or, worse, silently reversing it.

## Rules

- One file per decision: `NNNN-short-kebab-title.md`, numbered sequentially, never renumbered.
- An ADR is only warranted when there was a **real alternative**. If there was only one
  sensible option, it was not a decision and does not need a record.
- ADRs are **immutable once accepted**. A decision that changes gets a *new* ADR that
  supersedes the old one; the old file stays, marked `Superseded by NNNN`. The history of
  reversals is the valuable part — an ADR log with no reversals in it is usually a log that
  has been quietly edited.
- State the **revisit trigger**: the observation that would make you change your mind. A
  decision that names its own expiry condition is worth far more than one written as though
  it were permanent.

## Log

| # | Decision | Status |
|---|---|---|
| [0001](0001-windows-dev-loop-with-docker-parity.md) | Windows-native dev loop, Docker for Linux/CI parity | Accepted |
| [0002](0002-stage-3-model-choice-audit.md) | Stage 3 model choice, audited after the fact | Proposed |
