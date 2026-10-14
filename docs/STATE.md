# STATE

> A cold session reads **this file first**, then `docs/PROJECT_PLAN.md`, then the current
> milestone file in `milestones/`.

**Last updated:** 2026-09-11
**Current stage:** Stage 0 — look at the data
**Branch:** `feat/stage-0-scaffold`

---

## RESUME HERE (2026-10-13) -- read this first after any context reset

**Where we are:** the plan's real Stage 3 (learned reranking vs BM25 vs off-the-shelf dense).
Part 1 (dense baseline) is DONE on branch `feat/stage-3-dense-retrieval`, **PR #3 open, not
merged**. Next is part 2: the cross-encoder reranker.

**Recent history, in order:**
1. 2026-10-07 Mounir rescoped Stage 3 back to the plan. The old "Stage 3" (fine-tuned verifier)
   is now "Stage 2, extended". Training better verifiers = optional enhancement, needs his yes.
2. Zero-shot verifier for Stage 4 measured (ADR-0002): PubMedBERT-MNLI-MedNLI, no training,
   condition A on the 505 evidence-bearing train claims = 80.6% [77.0, 84.0] vs majority 65.7%;
   SUPPORT 85% / CONTRADICT 72%, no see-saw. Weak at abstention: NEI recall 8.2% at claim level
   with no threshold. That is Stage 5, and the threshold is Mounir's decision.
3. PR #2 (head alignment, input order, bf16, logging) merged by Mounir. Two commits pushed after
   the merge were orphaned and recovered into PR #3.
4. Dense retrieval built and measured (train, n=505): BM25 hit@1 74.9 [70.9, 78.6] / hit@10 92.7;
   dense (all-MiniLM-L6-v2) hit@1 68.7 [64.6, 72.7] / hit@10 93.3. Complementary: only-BM25 22,
   only-dense 25, union 97.6% at top 10. FAISS ~8 ms vs numpy ~23-31 ms (warmed up, fair
   baseline). Not a speed necessity at 5,183 documents.
5. Lesson (visual + snippets) on dense vectors, FAISS, and bi- vs cross-encoder (Checkpoints 3
   and 4) delivered 2026-10-08. The encoder was NOT surveyed (it was already on disk); it needs an
   ADR-0002-style audit before the final Stage 3 comparison.

**Mounir's decisions, 2026-10-14 (Stage 3 part 2):**
- YES: core reranker (BM25 + dense shortlist, fine-tuned cross-encoder) + BOTH controls
  (rank fusion RRF; the off-the-shelf ms-marco cross-encoder zero-shot as a reranker).
- YES: enhancement C (cost per claim for every pipeline) and B (contamination audit of candidate
  models). HOLD: A (false-negative audit) until the reranker works.
- Before ANY Stage 3 code: teach dense retrieval and FAISS properly (why this embedding model,
  with model cards; what the index stores), then the reranker training concept.
- **Mounir writes the metric code and the cost-logging code himself** (Claude writes the spec and
  the tests, he writes the function, Claude corrects him).
- Teach the interview fundamentals (embeddings, chunk size, measuring retrieval quality) with
  this project's measured numbers.

**Next, in order:**
1. Stage 3 part 2 DESIGN in chat (design before code): reranker over the BM25 + dense
   shortlist; the cheap control first (rank fusion); an encoder/reranker survey with a
   contamination audit; training negatives (the plan's "unannotated != negative" trap).
   Mounir decides the options.
2. Build, run on his PC plugged in, report in chat with snippets and a visual.
3. Drift check vs PROJECT_PLAN.md Stage 3 at the end; 2-3 optional enhancements with cost,
   benefit and learning.

**Working rules (all in memory too):** Mounir does not write code or docs; every explanation
uses real code snippets in chat; important information goes in chat, not only in files; visuals
where possible, else Graphviz/Mermaid code; drift check at every milestone; vary teaching
formats; no unmeasured numbers; be ambitious in proposals, never silently add scope; no
minimising language ("small enough", "easy", "just").

## BUILDING IS PAUSED (2026-09-26) -- read before doing anything

Mounir flagged, correctly, that Stages 1-3 were built faster than he learned them. He could
explain the concepts but had never walked through the code: tokenization, Dataset/DataLoader,
model setup, the training step. The plan's rule applies: *"If you've built three stages and I
understand one, stop building."* Claude should have flagged this and did not.

**No Stage 4 work until the lesson track below is complete.** The OQ-010 ablation (already
running when this was decided) may finish and be analysed; nothing new gets built.

| # | Lesson | Method | Status |
|---|---|---|---|
| 1 | Tokenization -- what the model actually receives | predict, then run | done 2026-09-27 |
| 2 | Dataset / DataLoader / padding -- how pairs become batches | broken version first | done 2026-09-28 |
| 3 | Model setup -- what's inside, what the head is, tensor shapes | worked example, shapes traced | done 2026-09-29 |
| 4 | The training step: annotated snippet + bug hunt (4 mystery runs, 3 signals) | snippets + diagnose-from-symptoms (Mounir does not write code) | done 2026-10-01 |
| 5 | GPU, `.to(device)`, memory breakdown, bf16 mixed precision | predict, then run | ready, awaiting run |

Lessons live in `lessons/` as runnable scripts that pause between reveals.

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
