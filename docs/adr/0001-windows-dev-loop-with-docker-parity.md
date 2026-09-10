# ADR-0001 — Develop on Windows natively; get Linux parity from Docker, not from WSL

- **Date:** 2026-09-11
- **Status:** Accepted
- **Decider:** Claude, on Mounir's explicit delegation ("whichever is better and proper")

## Context

The project targets Linux: GitHub Actions runs `ubuntu-latest`, and Stage 6 of the plan
requires Docker containers, which are Linux. The development machine is Windows 10 with WSL2
(Ubuntu 24.04) already installed and in use for other projects.

Measured on 2026-09-11, before deciding:

| | |
|---|---|
| Physical disk | One volume, **29.7 GB free of 395 GB** |
| WSL Ubuntu VHDX | 19.5 GB on disk, 11 GB used inside (≈8 GB reclaimable by compaction) |
| Docker Desktop WSL data | 13 GB |
| WSL toolchain | git 2.43, Python 3.12.3, gcc, make, curl. **No** `gh`, `uv`, Node, or Claude Code |
| Windows toolchain | Python 3.12.3, git 2.51, gh 2.97, Docker 28.5.1, all authenticated |

Stages 0–5 (BM25, a classifier, a cross-encoder, calibration) are pure Python and PyTorch,
with no OS-specific behaviour.

## Decision

The development loop runs on Windows natively. Linux parity is obtained from Docker at Stage 6
rather than by relocating development into WSL2.

The underlying principle: there are two ways to get dev/prod parity — make the laptop *be* the
target OS, or make the *artifact carry its own environment*. This project takes the second,
because that is what containers are for and it is the one that transfers to production.

## Alternatives considered

**Full WSL2 development, repository in the WSL filesystem.** This was the initial position, and
it was reversed on evidence:

- The disk argument, which was the first reason to hesitate, turned out to be **neutral**: the
  WSL VHDX and Windows draw on the same 29.7 GB. "WSL costs you disk" was simply false.
- Neither Node nor Claude Code is installed in WSL. Adopting it meant installing both and
  abandoning the desktop app in active use — a cost falling entirely on the user's workflow.
- The parity benefit is near zero through Stage 5, since nothing in those stages touches
  OS-specific behaviour.

Ruled out because the cost was immediate and concrete while the benefit was deferred and small.

**Repository on the Windows filesystem, executed from WSL via `/mnt/c`.** Ruled out on
performance: `/mnt/c` goes through the 9p protocol translation layer, and virtualenv operations
over thousands of small files are substantially slower there. It also inherits the path-mangling
class of bug documented in `docs/concepts/02-three-bug-classes.md`.

## Consequences

**Easier.** No workflow change, no migration, no second toolchain to maintain. The repository
stays visible to Explorer and the desktop app.

**Harder.** The dev machine no longer matches CI. This is a real, accepted risk, not a
dismissed one.

**How the risk is instrumented.** CI runs a matrix on **both** `ubuntu-latest` and
`windows-latest`. A divergence bug therefore surfaces on a pull request rather than at Stage 6
when Docker arrives. Taking on a risk without instrumenting for it would be the actual mistake;
taking it on with a detector is a trade.

**Known gap in that instrumentation.** The CI test step runs `pytest -m "not needs_data"`,
because the corpus is gitignored and never present on a runner. So the matrix proves the code
imports, type-checks and passes unit tests on both operating systems. It does **not** exercise
any code path that touches the actual data, which is precisely where encoding, line-ending and
path-separator divergence would appear. This gap is recorded rather than hidden.

## Revisit trigger

**The first genuine OS-divergence bug** — a defect that reproduces on one operating system and
not the other. At that point the cost of Windows-native development has been demonstrated
rather than hypothesised, and the decision should be re-made with that evidence in hand.
