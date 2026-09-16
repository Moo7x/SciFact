"""Display SciFact claims with their gold evidence marked, for reading by hand.

Stage 0 of the project plan is *read real examples by hand, no code*. Raw JSONL is not
readable by a human, so this exists to remove the transcription barrier and nothing else.
It displays; it does not summarise, aggregate, or characterise. Those are the reader's job,
and a tool that did them would quietly replace the exercise.

Usage::

    python scripts/peek.py                        # one random train claim
    python scripts/peek.py -n 5                   # five of them
    python scripts/peek.py --label NOT_ENOUGH_INFO   # only abstention cases
    python scripts/peek.py --claim-id 42          # one specific claim
    python scripts/peek.py --split dev --seed 7   # reproducible sample from dev
"""

from __future__ import annotations

import argparse
import json
import random
import textwrap
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
WIDTH = 88


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python scripts/download_data.py")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def wrap(text: str, indent: str = "") -> str:
    return textwrap.fill(
        text, width=WIDTH, initial_indent=indent, subsequent_indent=indent + "    "
    )


def show(claim: dict[str, Any], corpus: dict[int, dict[str, Any]]) -> None:
    print("=" * WIDTH)
    print(wrap(f"CLAIM {claim['id']}  |  {claim['claim']}"))

    evidence: dict[str, list[dict[str, Any]]] = claim.get("evidence", {})
    cited: list[int] = claim.get("cited_doc_ids", [])

    if not evidence:
        print()
        print(wrap("EVIDENCE: none. This claim is NOT_ENOUGH_INFO against its cited abstracts."))
    print(wrap(f"cited_doc_ids: {cited}"))
    print(wrap(f"abstracts carrying annotated evidence: {sorted(evidence) or 'none'}"))

    # Show every cited abstract, annotated or not. Seeing an abstract that was cited but
    # carries no rationale is exactly the distinction Stage 0 is meant to surface.
    for doc_id in cited:
        doc = corpus.get(doc_id)
        if doc is None:
            print(f"\n  [doc {doc_id} not in corpus]")
            continue

        annotations = evidence.get(str(doc_id), [])
        marked: dict[int, list[str]] = {}
        for ann in annotations:
            for idx in ann.get("sentence_indices", []):
                marked.setdefault(idx, []).append(ann.get("label", "?"))

        status = "ANNOTATED" if annotations else "cited, no rationale annotated"
        print(f"\n  {'-' * (WIDTH - 4)}")
        print(wrap(f"ABSTRACT {doc_id}  [{status}]", indent="  "))
        print(wrap(doc["title"], indent="  "))
        print()

        for i, sentence in enumerate(doc["abstract"]):
            labels = marked.get(i)
            marker = f"  <== {'/'.join(labels)}" if labels else ""
            print(wrap(f"[{i:>2}] {sentence.strip()}{marker}", indent="  "))
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-n", type=int, default=1, help="how many claims to show")
    parser.add_argument("--split", default="train", choices=["train", "dev"])
    parser.add_argument("--claim-id", type=int, default=None, help="show one specific claim")
    parser.add_argument(
        "--label",
        default=None,
        help="only claims whose evidence carries this label, "
        "or NOT_ENOUGH_INFO for claims with no evidence at all",
    )
    parser.add_argument("--seed", type=int, default=None, help="reproducible sampling")
    args = parser.parse_args()

    corpus = {doc["doc_id"]: doc for doc in load_jsonl(DATA_DIR / "corpus.jsonl")}
    claims = load_jsonl(DATA_DIR / f"claims_{args.split}.jsonl")

    if args.claim_id is not None:
        selected = [c for c in claims if c["id"] == args.claim_id]
        if not selected:
            raise SystemExit(f"No claim with id {args.claim_id} in {args.split}.")
    else:
        pool = claims
        if args.label:
            wanted = args.label.upper()
            if wanted == "NOT_ENOUGH_INFO":
                pool = [c for c in claims if not c.get("evidence")]
            else:
                pool = [
                    c
                    for c in claims
                    if any(
                        ann.get("label") == wanted
                        for anns in c.get("evidence", {}).values()
                        for ann in anns
                    )
                ]
            if not pool:
                raise SystemExit(f"No claims matching label {wanted} in {args.split}.")
        rng = random.Random(args.seed)  # noqa: S311 - sampling for display, not security
        selected = rng.sample(pool, min(args.n, len(pool)))

    for claim in selected:
        show(claim, corpus)

    print(f"({len(selected)} of {len(claims)} claims in {args.split})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
