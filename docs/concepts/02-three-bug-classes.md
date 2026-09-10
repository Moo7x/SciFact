# Three bug classes, found while scaffolding

*2026-09-11. These are worth knowing by **class**, not by their specific fix — the fix is
one line each, but the class recurs for the rest of your career.*

---

## 1. The variable was expanded by the wrong shell

**What happened.** Running a probe inside WSL from Git Bash:

```bash
wsl.exe -d Ubuntu -- bash -lc 'for c in git gh uv; do $c --version; done'
```

Every iteration printed `--version: command not found`. Nine identical errors.

**Why.** The *outer* shell consumed `$c` before WSL ever saw the string. By the time the inner
bash ran, the loop body was a bare `--version`.

**The class.** Any time a command crosses an interpreter boundary — shell to shell, shell to
SSH, shell to Docker, a shell string inside a Python `subprocess` call — there are now two
things that want to expand your `$variables`, and only one of them should.

**How to recognise it.** A loop producing *identical* output for every iteration, or a variable
that is mysteriously empty inside a nested command. If the same code works when you paste it
directly into the inner shell, this is your bug.

---

## 2. Path mangling at a POSIX/Windows boundary

**What happened.** Passing a Linux path to WSL from Git Bash:

```
bash: C:/Program Files/Git/mnt/c/Users/.../probe.sh: No such file or directory
```

**Why.** Git Bash runs on MSYS2, which helpfully rewrites anything that looks like a Unix path
into a Windows path before handing it to a native `.exe`. It saw `/mnt/c/...` and "helped".

**The class.** A translation layer applying a transformation you did not ask for, at a boundary
you did not know was a boundary.

**How to recognise it.** A path in an error message carries a prefix **nobody wrote**. That
prefix names the layer that did it to you.

**The fix.** `MSYS_NO_PATHCONV=1` before the command.

---

## 3. Silent wrong-shape extraction — the most instructive one

**What happened.** `download_data.py` flattened the downloaded archive with this logic:

> *if there is exactly one top-level entry and it is a directory, descend into it*

The SciFact tarball was built on a Mac. Its top level holds `data/` **and** `._data` — a
220-byte AppleDouble resource fork, macOS metadata that got swept into the tar. So the count was
**two**, the condition went false, and everything landed one level too deep as `data/data/`.

**Why this one matters most.** It did not crash. It did not warn. It produced a perfectly valid
directory tree — just the wrong one. Had a path been hardcoded downstream, the symptom would
have appeared days later as a missing file and been misdiagnosed as a failed download.

**The class.** *A heuristic that encodes an assumption about an artifact's shape, validated
against the shape such artifacts usually have rather than against the actual artifact.*

**How to recognise it.** A path with one more level than you expected. Or, more generally: a
step that reports success and produces output that is structurally right and semantically wrong.

**The fix, and the better habit.** Filter the macOS metadata, *and* make the code print when it
declines to flatten. An assumption that announces itself when it stops holding is worth more
than one that merely happens to be correct today.

---

## 4. Bonus: an unanchored `.gitignore` rule

Not one of the three, but the same family and the most dangerous of the lot.

`.gitignore` contained `data/` — intended to exclude the downloaded corpus. But an unanchored
pattern matches a directory of that name **at any depth**, so it also silently excluded
`src/scifact/data/` — my own source package.

The failure mode: a source file present on your disk, absent from the repository. Everything
passes locally forever. A fresh clone in CI dies with an `ImportError` pointing nowhere near
the cause.

Two rules came out of it:

- **`/data/` not `data/`** — the leading slash anchors the pattern to the repo root.
- **`/data/*` not `/data/`** — git never descends into an excluded *directory*, so a negation
  like `!/data/MANIFEST.json` inside one can never fire. Exclude the *contents* to keep the
  directory traversable.

**The general lesson across all four:** every one of these failed *quietly*. None threw an
exception. This is why "it ran without errors" is not evidence that it worked, and why the
download script now prints its record counts — so a wrong result has somewhere to be visible.
