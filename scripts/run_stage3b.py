"""OPTIONAL ENHANCEMENT, not a stage: train better-matched verifiers (ADR-0002).

Not run without Mounir's explicit yes (decision 2026-10-07). Kept so the option is ready.

Does a better-matched model fix what the verifier fine-tune broke?

Three arms. Everything is identical except the model: same data, seed, epochs, learning rate,
class weights, negatives, per-batch padding and bf16. The baseline is re-run rather than reused,
because the padding fix changed dropout's random draws (Lesson 4 debrief), and runs on
different code are not comparable.

  baseline   ms-marco-MiniLM-L-6-v2   relevance head discarded, new 3-way head, claim-first
  nli_small  nli-MiniLM2-L6-H768      pretrained NLI head, aligned, evidence-first
  nli_bio    PubMedBERT-MNLI-MedNLI   pretrained NLI head, aligned, evidence-first

The questions, fixed before running:
  1. Does a pretrained NLI head remove the SUPPORT / CONTRADICT see-saw (OQ-010)?
  2. Does biomedical pretraining add anything on top of the NLI head?
  Caveat stated in advance: nli_bio's pretraining corpus is PubMed, the source of SciFact's
  abstracts, so it has plausibly read our evidence text (ADR-0002).

Run on the same machine, plugged in. All three arms: about 10 minutes on the RTX 3050 Ti.

    python scripts/run_stage3b.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN = REPO_ROOT / "scripts" / "train_crossencoder.py"

ARMS: list[tuple[str, str]] = [
    ("baseline", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
    ("nli_small", "cross-encoder/nli-MiniLM2-L6-H768"),
    ("nli_bio", "pritamdeka/PubMedBERT-MNLI-MedNLI"),
]


def main() -> int:
    t0 = time.perf_counter()
    for i, (tag, model) in enumerate(ARMS, start=1):
        print(
            "=" * 74 + f"\nSTAGE 3b  ARM {i}/{len(ARMS)}: {tag}  ({model})\n" + "=" * 74, flush=True
        )
        result = subprocess.run(  # noqa: S603 - fixed argument list, no user input
            [sys.executable, str(TRAIN), "--model", model, "--tag", tag, "--bf16"],
            check=False,
            cwd=REPO_ROOT,
        )
        if result.returncode != 0:
            print(f"\n  arm {tag} FAILED (exit {result.returncode}). Stopping.")
            return result.returncode
        print(f"\n  arm {tag} done at {(time.perf_counter() - t0) / 60:.1f} min\n", flush=True)
    print(f"ALL ARMS DONE in {(time.perf_counter() - t0) / 60:.1f} min.")
    print("Histories: outputs/train_history_{baseline,nli_small,nli_bio}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
