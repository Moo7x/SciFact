# Project 1 — Evidence Retrieval (SciFact)

> **This file is the handoff.** It is written to be read by an AI agent in a fresh session with
> no memory of the conversation that produced it. Read all of it before writing any code.
>
> Status: planning complete, nothing built yet. Created 2026-09-10.

---

## Context — why this project, and why this way

I am a data science undergraduate. I am building 2–3 portfolio projects to gain the engineering
skills that job postings ask for and university does not teach. This is the first of them.

The project was selected after a long comparison process: roughly thirty candidates, verified
against real published work and shipped products, cross-checked independently against two other
AI systems (Gemini and Codex), with several candidates killed on hard evidence. This is not a
first idea — it is what survived.

**The other two projects** (not built here, listed so a fresh agent knows the wider shape):
a streaming recommendation system on the EB-NeRD dataset, and a cross-generator synthetic-image
detection study. Those are separate. Do not pull their scope into this one.

**Why SciFact specifically:** it replaced an earlier idea ("Truename," sanctions-list name
matching) that required me to hand-label a golden evaluation set — weeks of work, and if the
labels were weak, every downstream result was void. SciFact ships gold evidence annotations,
and its "not enough information" class means abstention is part of the task rather than bolted
onto it.

---

## PART I — HOW I WANT TO LEARN (read this twice)

### The core problem

You are going to write most of the code. That is intended. But the historical route to
competence ran *through* typing code yourself, and that route is now optional.

**Something has to replace it deliberately, or I end up with a repository I cannot defend and
skills I do not have.** Your job is not only to build this. Your job is to make sure that by the
end, I could have.

### What I actually need to learn — and what I don't

In a job I will likely not be hand-writing most code either. So drilling syntax is close to
worthless. What carries value when AI writes the code:

| Worth learning deeply | Not worth drilling |
|---|---|
| What a component actually does, and why it exists | Memorizing library syntax |
| Why this choice over the obvious alternative | Typing boilerplate from memory |
| How I would know if the output is wrong | Reproducing code I can look up |
| What breaks, how it breaks, how to recognize it | Framework-specific trivia |
| When a decision matters and when it doesn't | — |

**The skill is judgment and verification, not production.** Teach toward that.

I have strong maths. Do not simplify mathematical explanations on the assumption that I can't
follow them — go to the real formulation.

### Your obligations

**Design before code.** Before building any component: what you're about to build, why, what the
main alternative was, why you rejected it. I should be able to argue with it before it exists.
If I agree too readily, push back — ask what the failure mode of my agreement would be.

**Stop at the checkpoints below and teach properly.** Not in passing. For each concept:
intuition in plain language first → the formal definition → why it matters *here specifically* →
a small worked example with real numbers or real data from this project.

**Vary the method.** Some concepts need a worked example. Some need you to show me the broken
version first so I see what the correct version prevents. Some need me to predict the outcome
before you run it, then compare. Some need you to write the tests and let me write the function.

**Suggest external material at the moment it becomes relevant** — a paper, a chapter, a talk, a
blog post — and say *why that source* and *what specifically to read in it*. Not a reading list
up front. And go beyond what I ask for: if there's something I should learn that I haven't
thought to ask about, say so.

**Check understanding rather than assuming it.** After anything non-trivial, ask me to explain it
back and actually wait. Ask a question that needs understanding, not recall. If my answer is
vague, find where the misunderstanding started and fix it there — don't just restate.

**Debugging is the best teaching moment available.** When something breaks: show me the symptom,
walk the diagnosis, name the *class* of bug, tell me how to recognise it next time. Don't
silently fix things.

**Tell me when something must be done manually.** Some things I have to do myself (list below).
When we reach one, stop and say so explicitly rather than doing it for me.

**Never let me merge code I can't explain.** If I say "just do it, I trust you" — that is exactly
the moment to slow down. Remind me of this line.

---

## PART II — WHAT I DO BY HAND

You may critique these afterwards. Do not generate them. If I ask you to, remind me why they're
on this list.

| Artifact | Why it must be mine |
|---|---|
| **Reading a real sample of the data** — actual claims, abstracts, evidence annotations | You cannot understand a dataset you have never looked at. Do this before any code. |
| **The evaluation protocol** — what counts as correct, what the splits are, what's held out | If you design my evaluation, I own nothing. This is the single most important artifact. |
| **The metric implementations** — recall@k, precision/recall on evidence, whatever else | If I can't compute my own metrics, I can't defend a single number. |
| **The abstention threshold decision** | The core judgment call of the project. |
| **Every ADR** (architecture decision record) | These are my interview answers. |
| **`docs/LEARNING_LOG.md`** | This becomes interview prep and blog drafts. |

**You own:** repo scaffolding, Docker, indexing plumbing, training-loop boilerplate, FastAPI
wiring, CI configuration, tracing setup. These get probed at concept level in interviews, not
line level.

---

## PART III — STOP-AND-TEACH CHECKPOINTS

Named explicitly so they don't get skipped. Stop at each. Do not proceed until I can explain it
back.

1. **Before the evaluation protocol exists** — what makes a split valid, why dev and test must
   stay separate, what leakage looks like in a retrieval task.
2. **BM25 / lexical retrieval** — what it actually computes, why term frequency alone fails, what
   the tuning parameters mean. This is the baseline everything is measured against; I need to
   understand it before I have opinions about beating it.
3. **Embeddings and dense retrieval** — what a vector actually represents, why cosine similarity,
   what "semantic" means concretely and where it fails.
4. **Bi-encoder vs cross-encoder** — why one can be indexed and the other can't, and why that
   single architectural fact dictates the two-stage design.
5. **The training loop** — what a batch is, what the loss is measuring, what an optimizer step
   does, what would make it diverge.
6. **Calibration and abstention** — what a score threshold means, why raw model outputs aren't
   probabilities, how cost asymmetry changes where the threshold goes.
7. **Agent control flow** — the difference between the graph deciding control flow and the model
   deciding content, and why unbounded agents are dangerous.
8. **Tracing** — what a span is, what you can diagnose with traces that you cannot with logs.

---

## PART IV — STANDING RULES (carried from prior work, learned the hard way)

**Never state an unmeasured number as a result.** Five separate AI systems did this during the
selection process — "F1 dropped from 92% to 41%", "1.8× faster", "88% patch success rate" — none
of it measured. Describe *what is measured*; leave the value blank until a real run produces it.

**Prior art does not disqualify an idea.** SciFact has been worked on extensively. That changes
how the project is *pitched* — cite prior work, position honestly as implementation/comparison,
never claim to be first — it does not change whether it's worth building. I am not trying to
invent anything.

**A tool may be added to learn it — but its milestone must demonstrate something specific.**
Adding Kafka is fine if the milestone demonstrates replay, duplicates, partitioning, consumer
recovery. Adding it to have it on a list is not. I do not need to prove commercial-scale
necessity first; I do need to show what I learned.

**Do not convert my questions into settled decisions.** If I ask "should we do X?", that is a
question. Answer it; don't record it as decided.

**Hardware:** Windows 10 + WSL2, laptop RTX 3050, free Colab, willing to rent cloud GPU. This
project starts on CPU. Check actual memory use before assuming something fits — don't reason
"it's small, therefore fine."

**Timeline is open-ended.** Do not trim scope for deadline pressure, and don't invent budget
constraints I haven't given you.

---

## PART V — THE PROJECT

### The question

*When does iterative, adaptive evidence retrieval improve grounded decisions enough to justify
its extra cost and latency, compared to a fixed pipeline?*

### What it does

Given a scientific claim, find relevant evidence in a fixed corpus of abstracts, identify which
sentences support or contradict it, and return an evidence-linked verdict — or explicitly say
the evidence is insufficient.

### Verified foundation

[SciFact](https://github.com/allenai/scifact) (AllenAI, Wadden et al. 2020). Confirmed
2026-09-10:
- Claims labelled **supported / contradicted / not-inferable**, with sentence-level rationale
  annotations
- **Train and dev are labelled; test is unlabelled** (leaderboard submission only)
- A 5-fold cross-validation split is provided
- Download via `script/download-data.sh` or an S3 tarball
- Reference implementations and pretrained models exist in-repo

⚠️ Their setup instructions specify **Python 3.7 / Anaconda**, which is end-of-life. Treat their
environment files as reference, not as something to install directly.

### Build stages

Each stage ships something working. Do not start the next until the current one is understood,
not merely running.

**Stage 0 — Look at the data.** I read real claims, real abstracts, real evidence annotations by
hand. No code. Understand what an example actually looks like before deciding anything.

**Stage 1 — Evaluation protocol and BM25 baseline.** I define the protocol; you implement BM25
retrieval. Measure how much gold evidence a purely lexical method recovers. Inspect what it
misses and why — this shapes everything after.

**Stage 2 — Separate retrieval failure from reasoning failure.** Add a simple fixed classifier
over retrieved evidence. Critically: measure these two failure modes *independently*. A wrong
verdict because the evidence was never retrieved is a different bug from a wrong verdict on
correct evidence.

**Stage 3 — Learned reranking.** Fine-tune a compact cross-encoder on the training evidence.
Compare against lexical and against off-the-shelf dense retrieval. Train **one** component
properly rather than three badly.

**Stage 4 — Fixed pipeline vs bounded agent.** Same corpus, same call/token budget for both. The
agent may reformulate a query, search again, read an abstract, or stop. Measure whether adaptive
retrieval earns its cost. **A result showing the cheaper fixed pipeline wins is a real finding —
report it.**

**Stage 5 — Abstention and calibration.** Make "insufficient evidence" a calibrated decision with
a defended threshold, not a fallback for confusion. I choose the operating point.

**Stage 6 — Reliability.** Structured evidence IDs with deterministic schema validation. Traces
over retrieval, tool calls, and validators. Timeout/retry, fallback path, caching, regression
tests. Inject hostile instructions into test documents and measure what happens — read-only
tools, no write access.

### Stack, and what each piece is for

| Tool | Why it's here |
|---|---|
| PyTorch + Transformers / Sentence-Transformers | Fine-tuning the reranker |
| A lexical index (BM25) | The baseline that must be beaten honestly |
| FAISS or one vector store | Dense retrieval comparison — **one**, not both |
| FastAPI | The service surface |
| One orchestration framework | The bounded agent in Stage 4 — one, not several |
| OpenTelemetry | Traces must explain a real failure, not decorate |
| MLflow | Experiment tracking once there are variants worth comparing |
| Docker + GitHub Actions | Reproducibility and an evaluation regression gate |

LoRA is an optional later experiment if there's a question it answers. Not for the acronym.

### Evaluation traps — these will silently invalidate results

- **Do not tune on dev and then report it as held-out.** Tune within training data; keep a
  labelled holdout untouched.
- **The public test set is unlabelled.** Never call a dev-split number a test result.
- **Do not hand retrieval the annotation-only cited-document IDs.** That leaks the answer.
- **Unannotated retrieved documents are not guaranteed negatives** — they may be relevant but
  unlabelled. Treating them as negatives corrupts training and evaluation both.
- **SciFact is a public benchmark; foundation models have likely seen it.** This limits claims
  about absolute performance. Use controlled component-vs-component comparisons under matched
  conditions, where contamination affects both sides equally.
- **SciFact is small.** It limits how strong a conclusion fine-tuning can support. Say so.

---

## PART VI — CONTINUITY

I work across many sessions with different agents. **State lives in the repository, not in a
conversation.** Set this up in session one and maintain it.

```
docs/
  STATE.md            # current stage, what's done, what's next, open questions
  LEARNING_LOG.md     # one entry per session: built, broke, learned
  OPEN_QUESTIONS.md   # things you asked me that I couldn't answer — the gap list
  ARCHITECTURE.md     # the system as it currently is
  EVALUATION.md       # the protocol — mine, authoritative
  MAP.md              # module → purpose → key files, for cold sessions
  concepts/           # every explanation you give me, saved for revision
  adr/                # one file per real decision, written by me
milestones/
  stage-N-*.md        # goal, definition of done, what changed from this plan
```

- Update `STATE.md` at the end of every session — enough that a cold agent can resume from it.
- Save every concept explanation into `docs/concepts/`. I re-read these before interviews.
- Write the stage file when a stage completes, **including where reality differed from this
  plan.** This plan is a plan, not a record.
- A fresh session reads `STATE.md` first, then this file, then the current stage file.

---

## PART VII — ENGINEERING PRACTICE

Run this the way a real team runs, because learning the working practices matters as much to me
as the code. Apply these from the start, and **teach the reasoning behind each one as it comes
up** rather than just doing it.

**Version control**
- No direct commits to `main`. Branch per stage or fix (`feat/stage-1-bm25-baseline`,
  `fix/eval-split-leak`).
- Conventional commit messages (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `chore:`).
- Open a PR even though I'm solo. I review the diff before merge — that review is part of the
  learning, not a formality.
- PR description states: what changed, why, what alternative was considered, how it was tested.
- Tag releases at stage boundaries.
- **Never commit** secrets, API keys, or bulk downloaded data. Provide `.env.example` and a
  reproducible download script instead. SciFact data goes in `.gitignore`.

**Quality gates**
- CI on every PR: lint, type-check, tests. Red build blocks merge.
- Pre-commit hooks so formatting never reaches review.
- Pinned dependencies with a lockfile.
- Seeds fixed and recorded anywhere randomness affects a result.

**Documentation as part of done**
- A stage isn't done until its docs are updated — not "later."
- ADRs for every decision with a real alternative. Written by me; prompt me when one is needed.
- A runbook for anything operational: how to run it, how it fails, how to recover.

**Working discipline**
- Enforce the definition of done. If scope drifts, say so explicitly and let me decide — don't
  let it slide silently.
- Small vertical slices that work end to end, not big-bang builds.
- Flag technical debt when we take it on, with the reason and the cost of paying it later.

Explain the *why* of these as we hit them — why branch protection exists, why conventional
commits matter, why an ADR beats a code comment, why small PRs get better review. I want to
understand the practice, not just follow it.

---

## PART VIII — DAY ONE

1. Repo skeleton, `docs/` structure, `STATE.md` initialised.
2. Download SciFact (the script, in a modern Python environment — not their 3.7 spec).
3. **I read real examples by hand** — claims, abstracts, annotations. No code yet.
4. BM25 baseline. Measure gold-evidence recall.
5. **Look at what it misses**, together. That conversation is the actual first lesson.

---

## PART IX — THINGS I DIDN'T ASK FOR BUT SHOULD KNOW

Added because they were requested if I hadn't thought of them.

**Reading an explanation feels like understanding. Explaining it back reveals the gap.** The
teach-back isn't ceremony — it's the only reliable test. Expect me to be worse at explaining than
I expect to be, and treat that as information rather than failure.

**Keep a "questions I couldn't answer" list in the learning log.** When you ask something and I
can't answer, that's the highest-value signal in the entire project. Those gaps are what to
target next.

**Don't get ahead of me.** If you've built three stages and I understand one, stop building. A
repository that runs is not the goal; a repository I can defend is. Flag it if you notice the gap
widening.

**Ask "why not the simpler thing?" every time a component is added.** What's the simpler version,
and what specifically does it fail to do? That question is how judgment gets built instead of
tools getting accumulated.

**From roughly Stage 3, run a periodic hostile interview.** Play an interviewer with access only
to the repo, and push on whatever's weakest. Twenty minutes. It finds gaps nothing else does.

**Watch for yourself doing my thinking.** The most dangerous help is you proposing the evaluation
design, because that's the one thing I most need to own. If you catch yourself drifting there,
stop and hand it back.

**One finished, understood project beats two half-built ones.** If scope needs to shrink, shrink
it — but say so out loud as a decision, rather than letting stages quietly go unfinished.
