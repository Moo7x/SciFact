# STATE

> A cold session reads **this file first**, then `docs/PROJECT_PLAN.md`, then the current
> milestone file in `milestones/`.

**Last updated:** 2026-09-11
**Current stage:** Stage 0 — look at the data
**Branch:** `feat/stage-0-scaffold`

---

## Where things stand

Repository scaffolded. No data downloaded yet. No code that computes anything yet.
**No measured number exists in this repository.**

## Done

- [x] Repo initialised, remote `https://github.com/Moo7x/SciFact.git` (private), `main` pushed
- [x] Docs skeleton, ADR template, `.gitignore` (data excluded), `.gitattributes` (LF)
- [x] Environment decided: Windows-native dev loop, Docker for Linux/CI parity
      — ADR pending, **author: Mounir**
- [x] Toolchain green: `uv` venv, ruff, mypy `strict`, pytest — all passing locally
- [x] SciFact downloaded and checksummed. `data/MANIFEST.json` committed, data gitignored
- [x] `scripts/peek.py` — read claims and gold evidence by hand
- [x] PR #1 open, CI green on both `ubuntu-latest` and `windows-latest`

## GPU note (checked 2026-09-11, so it is not re-litigated later)

No CUDA toolkit is installed and **none is needed** — PyTorch wheels bundle their own CUDA
runtime; only the driver matters, and 546.80 / CUDA 12.3 already supports cu121 and cu124.
Existing `torch 2.11.0+cpu` is a CPU build. Stage 3 needs one `uv add` against the PyTorch
index, nothing more.

Torch is deferred not for disk reasons but because Stages 0–2 have no GPU consumer
(PART IV: a tool's milestone must demonstrate something specific).

**The real Stage 3 ceiling is 4 GB VRAM.** fp32 Adam costs ~16 bytes/parameter before
activations (4 weights + 4 grads + 8 Adam moments): ~22 M params ≈ 0.35 GB (comfortable),
~110–184 M ≈ 1.8–2.9 GB (needs AMP and gradient accumulation), ≥300 M ≈ 4.8 GB (over budget
before a single activation). To be measured, not trusted.

## Measured (the only numbers in this repo so far — all counts, no results)

| | |
|---|---|
| Corpus | 5,183 abstracts |
| Claims | 809 train / 300 dev / 300 test (**test unlabelled**) |
| CV split | 5 folds over `train ∪ dev` = 1,109 claims — **not** over `train` alone |
| Archive | sha256 `11c62128…76be`, 3.1 MB |

## Division of labour (amended 2026-09-11 — read this before assigning work)

Mounir prefers explanation in chat over producing written artifacts. **Claude writes all
documentation**, including ADRs and `docs/concepts/`. `docs/LEARNING_LOG.md` has been deleted.
This was raised, argued once, and decided — **do not re-propose it.**

Still Mounir's, unchanged: reading the data by hand, `docs/EVALUATION.md`, the metric
implementations, and the abstention threshold.

## Next

- [x] **ADR-0001** — dev environment decision (written by Claude)
- [ ] **Stage 0 (Mounir, by hand, no code):** read real claims, abstracts, evidence annotations
- [ ] **Checkpoint 1 teaching:** valid splits, dev/test separation, leakage in retrieval
- [ ] **`docs/EVALUATION.md` — Mounir writes.** Gate: nothing in Stage 1 is built before this
- [ ] **Checkpoint 2 teaching:** BM25 / lexical retrieval
- [ ] Stage 1: BM25 baseline (Claude implements) + metrics (Mounir implements)

## Open questions blocking progress

See `docs/OPEN_QUESTIONS.md`. Two are live and both belong to the evaluation protocol:

1. **Stage 2 needs a control condition.** As written in the plan, Stage 2 cannot separate
   retrieval failure from reasoning failure — one run over retrieved evidence yields one
   signal and two hypotheses. This has a consequence *upstream*: if the protocol requires
   gold-conditioned runs, the Stage 1 harness must support them from the start.
2. **Holdout vs the shipped 5-fold CV split.** Now unblocked — counts are measured. The CV
   split turns out to re-partition `train ∪ dev`, so adopting it dissolves the official dev
   set into training and leaves *nothing* labelled untouched (test is unlabelled). This makes
   the plan's "do not tune on dev then call it held-out" trap unavoidable rather than
   optional. See OQ-002 for the trilemma and the interval arithmetic to do first.

## Standing constraints (verified 2026-09-11, not assumed)

| | |
|---|---|
| Disk | One physical volume. **29.7 GB free of 395 GB.** ~8 GB reclaimable by compacting the WSL VHDX (19.5 GB on disk, 11 GB used inside). |
| GPU | RTX 3050 Ti Laptop, **4 GB VRAM**. Hard ceiling on Stage 3 reranker size. |
| Python | 3.12.3 (Anaconda). `uv.exe` in the user Scripts dir, not on PATH. |
| Torch | Install **CPU-only** until Stage 3 needs otherwise. The CUDA wheel is ~2.5 GB against 29.7 GB free. |
