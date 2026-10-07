"""Evaluate the fine-tuned cross-encoder against Stage 2's baselines, like for like.

Two views, because they answer different questions:

* **Pair level** -- one (claim, sentence) at a time, with a full confusion matrix. This is what
  the model was trained on and the only place a class-vs-class error pattern is visible.
* **Claim level** -- the Stage 2 three-condition decomposition, so the cross-encoder can be
  compared directly against the majority and lexical verifiers on identical inputs.

`train` split only. Dev is untouched and blocked on OQ-009.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scifact.data.schema import load_claims, load_corpus  # noqa: E402
from scifact.data.splits import split_train  # noqa: E402
from scifact.eval.decomposition import (  # noqa: E402
    condition_a_oracle_sentences,
    condition_b_oracle_abstract,
    condition_c_retrieved,
    print_decomposition,
)
from scifact.retrieval.bm25 import build_index  # noqa: E402
from scifact.verify.baselines import LexicalVerifier, MajorityVerifier  # noqa: E402
from scifact.verify.crossencoder import CrossEncoderVerifier  # noqa: E402
from scifact.verify.dataset import build_pairs  # noqa: E402
from scifact.verify.labels import VERDICTS  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
OUTPUTS = REPO_ROOT / "outputs"


def pair_level(verifier: CrossEncoderVerifier, pairs: list) -> None:
    """Confusion matrix on single (claim, sentence) pairs."""
    confusion: dict[tuple[str, str], int] = {}
    for p in pairs:
        pred = verifier.predict(p.claim, [p.evidence])
        key = (p.label, pred)
        confusion[key] = confusion.get(key, 0) + 1

    print()
    print("=" * 78)
    print(f"PAIR-LEVEL CONFUSION  (n={len(pairs):,} single claim/sentence pairs)")
    print("=" * 78)
    short = {v: v[:9] for v in VERDICTS}
    print(
        f"  {'gold \\ predicted':<20} "
        + "  ".join(f"{short[v]:>9}" for v in VERDICTS)
        + "   recall"
    )
    print(f"  {'-' * 20} " + "  ".join("-" * 9 for _ in VERDICTS) + "   ------")
    for gold in VERDICTS:
        row = [confusion.get((gold, pred), 0) for pred in VERDICTS]
        total = sum(row)
        recall = row[VERDICTS.index(gold)] / total if total else float("nan")
        print(f"  {gold:<20} " + "  ".join(f"{c:>9,}" for c in row) + f"   {recall:>6.1%}")

    correct = sum(confusion.get((v, v), 0) for v in VERDICTS)
    print(f"\n  overall pair accuracy: {correct / len(pairs):.1%}")
    print("\n  Read the ROWS: of everything whose true label was X, where did it go?")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--skip-claim-level", action="store_true")
    parser.add_argument("--tag", default="crossencoder", help="which outputs/<tag>/ model")
    parser.add_argument(
        "--model",
        default=None,
        help="a Hugging Face model id to evaluate ZERO-SHOT instead of a trained outputs/<tag>",
    )
    parser.add_argument(
        "--split",
        choices=["train_tune", "train"],
        default="train_tune",
        help="train = all 809 train claims. Valid only for a model never trained on SciFact "
        "(zero-shot): for it, every train claim is unseen. A model trained on train_fit must "
        "use train_tune, or it is scored on its own training data.",
    )
    args = parser.parse_args()

    if args.model:
        source: Path | str = args.model
    else:
        trained = OUTPUTS / args.tag
        if not trained.exists():
            raise SystemExit(f"{trained} not found. Run scripts/train_crossencoder.py first.")
        source = trained
        if args.split == "train":
            raise SystemExit("Refusing: a trained model scored on --split train is in-sample.")

    corpus = load_corpus(DATA_DIR / "corpus.jsonl")
    train = load_claims(DATA_DIR / "claims_train.jsonl")
    _fit, tune = split_train(train)
    tune = train if args.split == "train" else tune
    verifier = CrossEncoderVerifier(source)

    print(
        f"model: {source}   zero-shot NLI: {verifier.zero_shot_nli}   "
        f"order: {'evidence-first' if verifier.evidence_first else 'claim-first'}   "
        f"device: {verifier.device.type}"
    )
    print(f"split: {args.split} ({len(tune)} claims). Dev untouched.")

    pair_level(verifier, build_pairs(tune, corpus, seed=20260919))

    if args.skip_claim_level:
        return 0

    index = build_index(
        sorted(corpus),
        [corpus[i].title + " " + " ".join(corpus[i].abstract) for i in sorted(corpus)],
    )

    for v in (MajorityVerifier(), LexicalVerifier(0.45), verifier):
        print()
        print("=" * 78)
        print(f"CLAIM-LEVEL DECOMPOSITION: {v.name}")
        print("=" * 78)
        results = [
            condition_a_oracle_sentences(v, tune, corpus),
            condition_b_oracle_abstract(v, tune, corpus),
            condition_c_retrieved(v, tune, corpus, index, args.k, evidence_only=True),
            condition_c_retrieved(v, tune, corpus, index, args.k, evidence_only=False),
        ]
        print_decomposition(results)
        a, b, c = (r.accuracy.value for r in results[:3])
        print()
        print("  Decomposition (evidence-bearing claims):")
        print(f"    reasoning error, with perfect evidence      {1 - a:>7.1%}")
        print(f"    + cost of selecting the sentence            {a - b:>+7.1%}")
        print(f"    + cost of retrieving the abstract           {b - c:>+7.1%}")
        print(f"    = total pipeline error                      {1 - c:>7.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
