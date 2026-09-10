# Learning log

One entry per session: what got built, what broke, what I learned, what I could not explain.

**The "could not explain" section is the most valuable part of this file.** A question I
couldn't answer is a targeted gap. A question I answered is just a checkpoint passed.

---

## 2026-09-11 — Session 1: scaffolding

### Built

Repository skeleton, docs structure, ADR template, CI config, data download script.

### Broke

Two shell bugs while probing WSL from Git Bash. Both are worth recognising by *class*, not by
their specific fix:

- **Nested-quoting variable expansion.** `wsl.exe -- bash -lc '... $c ...'` — the *outer* shell
  consumed `$c` before WSL ever saw it, so a nine-iteration loop ran a bare `--version` nine
  times. Class: *the variable was expanded by the wrong shell.* Recognise it when a loop
  produces identical output for every iteration, or when a variable is mysteriously empty
  inside a command that crosses an interpreter boundary.

- **MSYS path translation.** Git Bash rewrote `/mnt/c/...` into `C:/Program Files/Git/mnt/c/...`
  before handing it to `wsl.exe`. Class: *path mangling at a POSIX/Windows boundary.*
  Recognise it when a path in an error message carries a prefix nobody wrote.
  Fix: `MSYS_NO_PATHCONV=1`.

- **Heredoc parse failure in a large multi-file shell block.** bash parses the entire `-c`
  string before executing any of it, so a single quoting error meant *nothing* ran — including
  the `mkdir` at the top. Class: *parse-time vs run-time failure.* Recognise it by the absence
  of partial side effects: if the first command's output is missing too, the script never
  started. Lesson: large generated shell blocks trade atomicity for a wide blast radius.

- **Silent wrong-shape extraction — the most instructive one.** `download_data.py` flattened
  the archive with "if there is exactly one top-level entry and it is a directory, descend
  into it". The tarball was built on macOS, so its top level held `data/` *and* `._data`
  (a 220-byte AppleDouble resource fork). The count was two, the condition went false, and
  everything landed one level too deep as `data/data/`.

  Class: *a heuristic encoding an assumption about an artifact's shape, validated against the
  common case rather than the actual artifact.* What makes this class dangerous is not that it
  fails, but that it fails **plausibly** — it produced a perfectly valid directory tree, just
  the wrong one. No exception, no warning. Had a path been hardcoded downstream, the symptom
  would have appeared days later as a missing file and been misdiagnosed as a bad download.

  Recognise it when a path in an error message has one more level than you expected, or when
  a "successful" step produces output that is structurally right and semantically wrong.
  The fix now also *prints* when it declines to flatten — an assumption that announces itself
  when it stops holding is worth more than one that is merely correct today.

### Learned

- There are two routes to dev/prod parity: make the laptop *be* the target OS, or make the
  *artifact carry its environment*. The second is what containers are for and the one that
  transfers to production.
- WSL's `df` reports the **virtual** disk size (1007 G), not the physical backing store. The
  real figure was 29.7 GB free. Checking it reversed a decision — a live instance of the
  plan's own "check, don't assume" rule, caught by following the rule.

- The shipped 5-fold CV split is **not** what it looks like. It re-partitions `train ∪ dev`
  (1,109 claims), so adopting it dissolves the official dev set into training and leaves no
  untouched labelled data anywhere, because test is unlabelled. Verified by claim-ID set
  operations rather than inferred from file sizes — the arithmetic (887 + 222 = 809 + 300)
  *suggested* it, but suggestive arithmetic is not a check. See OQ-002.

- Related habit worth keeping: two files can have matching record counts for reasons that have
  nothing to do with containing the same records. Comparing identifiers is a check; comparing
  counts is a coincidence detector.

### Could not explain / did not know

_To fill in as they come up. Do not leave this section empty out of politeness — an empty
section here across several sessions means the questions being asked are too easy._
