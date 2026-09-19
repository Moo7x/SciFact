"""Stage 1: BM25 document retrieval against its controls.

Defaults to `train_tune`. Evaluating on `dev` requires `--split dev` to be typed explicitly,
because the holdout should never be spent by a command someone ran without meaning to.

Usage::

    python scripts/run_stage1.py                 # train_tune
    python scripts/run_stage1.py --sweep         # tune k1/b on train_tune
    python scripts/run_stage1.py --split dev     # spends the holdout
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from scifact.data.schema import Claim, load_claims, load_corpus  # noqa: E402
from scifact.data.splits import split_train  # noqa: E402
from scifact.eval.harness import DEFAULT_KS, evaluate_document_retrieval, print_table  # noqa: E402
from scifact.retrieval.baselines import PopularityRetriever, RandomRetriever  # noqa: E402
from scifact.retrieval.bm25 import DEFAULT_B, DEFAULT_K1, build_index, tokenize  # noqa: E402

DATA_DIR = REPO_ROOT / "data"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # `train` is the right tuning set for Stage 1. The train_fit/train_tune division exists to
    # stop a model being tuned on data it was also fitted on -- and BM25 fits nothing. With no
    # learned parameters, all 809 train claims are legitimately available, which roughly
    # quintuples n and halves the interval. From Stage 3 onward, when something is actually
    # trained, train_tune becomes the correct choice again.
    parser.add_argument(
        "--split", default="train", choices=["train", "train_tune", "train_fit", "dev"]
    )
    parser.add_argument("--k1", type=float, default=DEFAULT_K1)
    parser.add_argument("--b", type=float, default=DEFAULT_B)
    parser.add_argument("--sweep", action="store_true", help="grid search k1/b on this split")
    args = parser.parse_args()

    corpus = load_corpus(DATA_DIR / "corpus.jsonl")
    doc_ids = sorted(corpus)
    texts = [corpus[i].title + " " + " ".join(corpus[i].abstract) for i in doc_ids]

    if args.split == "dev":
        claims = load_claims(DATA_DIR / "claims_dev.jsonl")
        print("*** Evaluating on the HELD-OUT dev split. Record this in docs/EVALUATION.md. ***")
    else:
        train = load_claims(DATA_DIR / "claims_train.jsonl")
        if args.split == "train":
            claims = train
        else:
            fit, tune = split_train(train)
            claims = tune if args.split == "train_tune" else fit

    if args.sweep:
        return sweep(doc_ids, texts, claims, args.split)

    t0 = time.perf_counter()
    index = build_index(doc_ids, texts, k1=args.k1, b=args.b)
    build_s = time.perf_counter() - t0

    results = [
        evaluate_document_retrieval(RandomRetriever(doc_ids), claims, args.split, "random (floor)"),
        evaluate_document_retrieval(
            PopularityRetriever(doc_ids, [len(tokenize(t)) for t in texts]),
            claims,
            args.split,
            "longest-abstract (control)",
        ),
        evaluate_document_retrieval(index, claims, args.split, f"BM25 (k1={args.k1}, b={args.b})"),
    ]

    print()
    print("=" * 74)
    print(f"STAGE 1 - DOCUMENT RETRIEVAL over {len(doc_ids):,} abstracts")
    print("=" * 74)
    print_table(results, DEFAULT_KS)
    print()
    print(f"  index build: {build_s:.2f}s")
    return 0


def sweep(doc_ids: list[int], texts: list[str], claims: list[Claim], split: str) -> int:
    if split == "dev":
        raise SystemExit("Refusing to sweep on dev. Tuning on the holdout is the thing to avoid.")

    print(f"Grid search on {split} (hit@10). This spends {split}, which is what it is for.")
    print()
    print(f"  {'k1':>5} {'b':>6}   {'hit@10':>8}")
    print(f"  {'-' * 5} {'-' * 6}   {'-' * 8}")

    best = (0.0, DEFAULT_K1, DEFAULT_B)
    tried = 0
    for k1 in (0.6, 0.9, 1.2, 1.5, 2.0):
        for b in (0.0, 0.3, 0.5, 0.75, 1.0):
            index = build_index(doc_ids, texts, k1=k1, b=b)
            res = evaluate_document_retrieval(index, claims, split, "bm25")
            score = res.hit_at[10].value
            tried += 1
            marker = "  <- best so far" if score > best[0] else ""
            print(f"  {k1:>5.1f} {b:>6.2f}   {score:>8.1%}{marker}")
            if score > best[0]:
                best = (score, k1, b)

    print()
    print(f"  best: k1={best[1]}, b={best[2]}  ->  hit@10 {best[0]:.1%}")
    print(f"  {tried} configurations tried on {len(claims)} claims.")
    print("  Selection bias applies: see docs/concepts/05. The winner's margin over the")
    print("  default is not evidence of improvement until it survives the holdout.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
