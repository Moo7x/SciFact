"""Stage 2: decompose pipeline error into reasoning error and retrieval error.

Usage::

    python scripts/run_stage2.py
    python scripts/run_stage2.py --k 5 --threshold-sweep
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scifact.data.schema import load_claims, load_corpus  # noqa: E402
from scifact.eval.decomposition import (  # noqa: E402
    condition_a_oracle_sentences,
    condition_b_oracle_abstract,
    condition_c_retrieved,
    print_decomposition,
)
from scifact.retrieval.bm25 import build_index  # noqa: E402
from scifact.verify.baselines import LexicalVerifier, MajorityVerifier  # noqa: E402

DATA_DIR = REPO_ROOT / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="train", choices=["train", "dev"])
    parser.add_argument("--k", type=int, default=3, help="abstracts retrieved per claim")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--threshold-sweep", action="store_true")
    args = parser.parse_args()

    if args.split == "dev":
        print("*** HELD-OUT dev split. Log this in docs/EVALUATION.md section 2. ***")

    corpus = load_corpus(DATA_DIR / "corpus.jsonl")
    claims = load_claims(DATA_DIR / f"claims_{args.split}.jsonl")
    doc_ids = sorted(corpus)
    index = build_index(
        doc_ids, [corpus[i].title + " " + " ".join(corpus[i].abstract) for i in doc_ids]
    )

    if args.threshold_sweep:
        if args.split == "dev":
            raise SystemExit("Refusing to sweep on dev.")
        # Swept on condition C over ALL claims, not condition A. The threshold controls
        # abstention, and condition A contains no NOT_ENOUGH_INFO claims at all -- there,
        # abstaining can only ever be wrong, so the sweep trivially selects "never abstain"
        # and tells you nothing. A parameter must be tuned on a set where it can be right.
        print(
            f"\nNEI threshold sweep on {args.split}, condition C (retrieved@{args.k}, all claims)"
        )
        print(f"\n  {'threshold':>10}   {'accuracy':>9}   {'NEI recall':>10}   {'S/C recall':>10}")
        print(f"  {'-' * 10}   {'-' * 9}   {'-' * 10}   {'-' * 10}")
        for t in (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90):
            res = condition_c_retrieved(
                LexicalVerifier(t), claims, corpus, index, args.k, evidence_only=False
            )
            rec = res.per_class_recall()
            sc = (rec["SUPPORT"] + rec["CONTRADICT"]) / 2
            print(
                f"  {t:>10.2f}   {res.accuracy.value:>9.1%}   "
                f"{rec['NOT_ENOUGH_INFO']:>10.1%}   {sc:>10.1%}"
            )
        print()
        print("  The two recall columns move in opposite directions. That trade is the")
        print("  abstention problem; picking a point on it is Stage 5.")
        return 0

    for verifier in (MajorityVerifier(), LexicalVerifier(args.threshold)):
        print()
        print("=" * 78)
        print(f"STAGE 2 - FAILURE DECOMPOSITION: {verifier.name}   split={args.split}")
        print("=" * 78)
        results = [
            condition_a_oracle_sentences(verifier, claims, corpus),
            condition_b_oracle_abstract(verifier, claims, corpus),
            condition_c_retrieved(verifier, claims, corpus, index, args.k, evidence_only=True),
            condition_c_retrieved(verifier, claims, corpus, index, args.k, evidence_only=False),
        ]
        print_decomposition(results)

        a, b, c = results[0].accuracy.value, results[1].accuracy.value, results[2].accuracy.value
        print()
        print("  Decomposition (evidence-bearing claims, like for like):")
        print(f"    reasoning error, with perfect evidence      {1 - a:>7.1%}")
        print(f"    + cost of selecting the sentence            {a - b:>+7.1%}")
        print(f"    + cost of retrieving the abstract           {b - c:>+7.1%}")
        print(f"    = total pipeline error                      {1 - c:>7.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
