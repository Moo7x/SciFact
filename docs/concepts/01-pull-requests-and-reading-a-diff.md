# Pull requests, and how to actually read a diff

*Written 2026-09-11, for PR #1. Re-readable: the method generalises, the examples do not.*

---

## 1. What a pull request actually is

Git gives you two things: **branches** and **merge**. That is all. A "pull request" is not a git
feature at all — it is a GitHub feature layered on top.

Mechanically, a PR says: *"here is branch X; I propose merging it into branch Y."* On top of
that proposal GitHub attaches four things git does not give you:

| | |
|---|---|
| A **diff view** | branch X compared against where it diverged from Y |
| A **comment thread** | anchored to specific lines |
| **CI results** | attached to the proposal, before the merge |
| A **merge button** | which records who decided, and when |

The name is historical. You are asking a maintainer to *pull* your changes into their branch.

## 2. Why bother when you are the only developer?

Three reasons, none of them ceremony.

**Writing and reviewing are different cognitive modes.** While writing, you hold your intent in
your head, and the code looks like what you meant — because you are reading your intention, not
the text. A diff strips the intent away and shows only what changed. You catch a different class
of bug in each mode. This is not a metaphor: on this project, three bugs were found by *running*
the code and a different set is findable by *reading* it.

**The gate becomes mechanical instead of disciplinary.** CI runs on the PR before the merge
button works. You cannot forget to run the tests, because forgetting is not an available action.
Discipline you have to remember is discipline you will eventually skip.

**It is a dated record of reasoning.** In six months, `git log` tells you *what* changed. The PR
body tells you *why*, and what you rejected, written while the alternative was still live.
That is the difference between "I chose Windows" and an answer to *"why not WSL?"* in an
interview.

## 3. How to read a diff: four passes, in order

Do not mix the passes. Each one asks a different question, and trying to ask all four at once is
why code review feels like staring.

### Pass 1 — Scope

Read the PR description. Then look **only at the file list**, not the contents.

> **Question:** does every file here belong to what the description claims?

A file that surprises you is either scope creep or a mistake. Both matter. This pass takes
sixty seconds and catches the most expensive category of problem.

### Pass 2 — Dangerous files first

GitHub lists files alphabetically. That is the wrong order. Read in order of **what could hurt
you**, which usually means: things that control what gets committed, then things that control
what gets verified, then code, then docs, then generated files.

Generated files (`uv.lock`, `package-lock.json`) are **skimmed, not read**. The only question is
whether they look machine-generated and plausible. Reading 343 lines of resolved dependency
hashes teaches you nothing and exhausts the attention you need for the 16 lines that matter.

### Pass 3 — "What input makes this wrong?"

For each real code file, do **not** ask "is this correct?" That question has no traction — you
will read the code, it will look reasonable, and you will learn nothing.

Ask instead: **what input, state, or timing makes this produce the wrong answer?** That question
forces you to simulate rather than skim. It is also exactly how a bug gets found.

### Pass 4 — What is missing?

The hardest pass, because absence is invisible in a diff. Tests that should exist and do not.
Docs that describe the old behaviour. Error cases silently unhandled. A diff can only show you
what is there.

## 4. Merge strategies

| Strategy | What lands on `main` | Cost |
|---|---|---|
| **Squash** | All commits collapse into one. PR title + body becomes the commit message. | Individual commits no longer appear in `main`'s history (they stay in the PR forever). |
| **Merge commit** | Every commit, plus an extra "merge" commit. | Full granularity, noisier history, harder to read at a glance. |
| **Rebase** | Every commit replayed onto `main`, no merge commit. | Linear history, but commit hashes are rewritten. |

**Default to squash** when the PR body carries the reasoning, which it should. Main's history
then reads as one line per unit of work, and the detailed commits remain visible in the PR.

```bash
gh pr merge 1 --squash --delete-branch
```

`--delete-branch` removes the branch after merging. It is already merged; keeping it around
just accumulates clutter.
