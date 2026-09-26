"""OQ-010: which of my two mistakes collapsed SUPPORT?

Stage 3 fixed CONTRADICT (11-16% -> 84-88%) and destroyed SUPPORT (21.6%, and 0.0% at any
confidence threshold). Two candidate causes, both mine, and they are separable by changing one
thing at a time:

  A. class weighting overshot   -- CONTRADICT got the largest weight (1.90) for being rarest,
                                   and went from ignored to over-claimed (precision 35%).
  B. negative sampling drowns SUPPORT -- every SUPPORT example ships with TWO near-identical
                                   counterexamples drawn from the SAME abstract, labelled NEI,
                                   2:1 against it. CONTRADICT escapes this because a negated
                                   claim carries an explicit polarity flip no same-abstract
                                   negative mimics.

Three arms, each differing from the baseline in exactly one variable:

  baseline   weights ON,  2 negatives   (already run; re-run here so all three share a seed)
  arm A      weights OFF, 2 negatives
  arm B      weights ON,  1 negative

Three configurations, not twenty-five. At sigma ~= 4pp on n=162 the expected selection
inflation at m=3 is about 4pp -- so a difference has to be large to mean anything, and that
limit is stated up front rather than discovered afterwards.

**Prediction on record, made before running, so it can be wrong: arm B moves SUPPORT more than
arm A.** Weighting shifts a decision boundary, and the tau sweep already showed no threshold
could recover SUPPORT -- which is evidence against A being the whole story.

Usage::

    python scripts/run_oq010.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN = REPO_ROOT / "scripts" / "train_crossencoder.py"

ARMS: list[tuple[str, list[str], str]] = [
    ("baseline", [], "weights ON, 2 negatives -- the run that collapsed SUPPORT"),
    ("no_weights", ["--no-class-weights"], "arm A: isolates class weighting"),
    ("one_neg", ["--max-negatives", "1"], "arm B: isolates negative sampling"),
]


def main() -> int:
    print("=" * 74)
    print("OQ-010 ABLATION -- three arms, one variable changed each")
    print("=" * 74)
    for tag, extra, why in ARMS:
        print(f"  {tag:<12} {' '.join(extra) or '(defaults)':<28} {why}")
    print("\n  ~3 min per arm on GPU (~40 on CPU). All three arms re-run so they share a device:")
    print("  the same seed on CPU vs GPU gives slightly different numbers, which would otherwise")
    print("  be a second variable hiding inside the comparison.")
    print("  Each arm writes outputs/<tag>/ and outputs/train_history_<tag>.json\n")

    t0 = time.perf_counter()
    for i, (tag, extra, _why) in enumerate(ARMS, start=1):
        print("=" * 74)
        print(f"ARM {i}/{len(ARMS)}: {tag}")
        print("=" * 74, flush=True)
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(TRAIN), "--tag", tag, *extra],
            check=False,
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            print(f"\n  ARM {tag} FAILED (exit {result.returncode}). Stopping.")
            return result.returncode
        print(f"\n  arm {tag} done at {(time.perf_counter() - t0) / 60:.1f} min\n", flush=True)

    print("=" * 74)
    print(f"ALL ARMS COMPLETE in {(time.perf_counter() - t0) / 60:.1f} min")
    print("=" * 74)
    print("  Next: python scripts/compare_oq010.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
