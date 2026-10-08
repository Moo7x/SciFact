"""Stage 3, part 1: off-the-shelf dense retrieval vs BM25, same harness, same claims.

The dense encoder is used off the shelf: it is trained on nothing here, so (as with BM25 in
Stage 1) all 809 train claims are legitimately available. Dev stays untouched.

Also measures two things the lesson needs:
  * complementarity -- per claim, which of the two methods finds the gold abstract;
  * the cost argument for FAISS -- exact search with FAISS vs plain numpy at this corpus size.

    python scripts/run_dense.py
    python scripts/run_dense.py --export path/to/widget.json   # data for the visual
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections.abc import Callable
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from scifact.data.schema import Claim, Document, load_claims, load_corpus  # noqa: E402
from scifact.eval.harness import DEFAULT_KS, evaluate_document_retrieval, print_table  # noqa: E402
from scifact.retrieval.bm25 import BM25, build_index  # noqa: E402
from scifact.retrieval.dense import DEFAULT_ENCODER, DenseIndex, Encoder  # noqa: E402

DATA = REPO_ROOT / "data"
CACHE = REPO_ROOT / "outputs"


def corpus_vectors(encoder: Encoder, texts: list[str]) -> np.ndarray:
    """Encode the corpus once and cache it: abstracts never change, claims do."""
    path = CACHE / f"corpus_vectors_{encoder.name.replace('/', '__')}.npy"
    if path.exists():
        vecs = np.load(path)
        if vecs.shape[0] == len(texts):
            print(f"  corpus vectors: loaded from cache {path.name}  shape {vecs.shape}")
            return vecs
    t0 = time.perf_counter()
    vecs = encoder.encode(texts)
    CACHE.mkdir(exist_ok=True)
    np.save(path, vecs)
    print(f"  corpus vectors: encoded {vecs.shape} in {time.perf_counter() - t0:.1f}s, cached")
    return vecs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--export", type=Path, default=None)
    args = parser.parse_args()

    corpus = load_corpus(DATA / "corpus.jsonl")
    claims = load_claims(DATA / "claims_train.jsonl")
    doc_ids = sorted(corpus)
    texts = [corpus[i].title + " " + " ".join(corpus[i].abstract) for i in doc_ids]

    encoder = Encoder(args.encoder)
    print(f"encoder: {args.encoder}   device: {encoder.device.type}")
    vecs = corpus_vectors(encoder, texts)
    dense = DenseIndex(encoder, doc_ids, vecs)
    bm25 = build_index(doc_ids, texts)

    # ------------------------------------------------------------------ head to head
    results = [
        evaluate_document_retrieval(bm25, claims, "train", "BM25 (Stage 1)"),
        evaluate_document_retrieval(dense, claims, "train", "dense (all-MiniLM-L6-v2)"),
    ]
    print(
        "\n" + "=" * 74 + "\nSTAGE 3 PART 1 -- lexical vs dense, over 5,183 abstracts\n" + "=" * 74
    )
    print_table(results, DEFAULT_KS)

    # ------------------------------------------------------------------ complementarity
    scored = [c for c in claims if c.has_evidence]
    claim_vecs = encoder.encode([c.claim for c in scored])
    _s, dense_pos = dense.search_vectors(claim_vecs, 10)
    both = only_bm25 = only_dense = neither = 0
    per_claim = []
    for c, row in zip(scored, dense_pos, strict=True):
        gold = set(c.evidence)
        b = any(d in gold for d in bm25.rank(c.claim, k=10))
        d = any(doc_ids[p] in gold for p in row)
        both += b and d
        only_bm25 += b and not d
        only_dense += d and not b
        neither += not b and not d
        per_claim.append((c, b, d))
    n = len(scored)
    print(f"\n  Who finds the gold abstract in the top 10?  (n={n} claims with evidence)")
    for label, v in (
        ("both", both),
        ("only BM25", only_bm25),
        ("only dense", only_dense),
        ("neither", neither),
    ):
        print(f"    {label:<11} {v:>4}  ({v / n:.1%})")
    print(f"    either one  {n - neither:>4}  ({(n - neither) / n:.1%})   <- ceiling if combined")

    # ------------------------------------------------------------------ why FAISS
    def timed(fn: Callable[[], object], reps: int = 5) -> float:
        """Mean wall time in seconds over `reps` runs, after one discarded warm-up run.

        A single cold run includes one-off costs (memory allocation, caches, thread pools)
        that say nothing about steady-state speed: one cold numpy run measured ~85 ms
        against ~23 ms warmed up.
        """
        fn()
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t0) / reps

    t_faiss = timed(lambda: dense.search_vectors(claim_vecs, 10))
    # Fair numpy baseline: argpartition finds the top 10 without fully sorting all 5,183.
    t_numpy = timed(lambda: np.argpartition(-(claim_vecs @ vecs.T), 10, axis=1)[:, :10])
    print(f"\n  exact top-10 for {n} claims over {len(doc_ids):,} abstracts:")
    print(f"    FAISS IndexFlatIP   {t_faiss * 1000:7.1f} ms")
    print(f"    numpy argpartition  {t_numpy * 1000:7.1f} ms")

    if args.export:
        export(args.export, corpus, doc_ids, vecs, encoder, bm25, dense, per_claim)
    return 0


def export(
    path: Path,
    corpus: dict[int, Document],
    doc_ids: list[int],
    vecs: np.ndarray,
    encoder: Encoder,
    bm25: BM25,
    dense: DenseIndex,
    per_claim: list[tuple[Claim, bool, bool]],
) -> None:
    """2D projection + neighbour lists for the visual. PCA is fitted on the corpus only."""
    mean = vecs.mean(axis=0)
    _u, _s, vt = np.linalg.svd(vecs - mean, full_matrices=False)
    axes = vt[:2].T  # (hidden, 2)
    explained = float((_s[:2] ** 2).sum() / (_s**2).sum())

    rng = random.Random(0)  # noqa: S311
    pick = (
        [x for x in per_claim if x[2] and not x[1]][:3]
        + [x for x in per_claim if x[1] and not x[2]][:3]
        + [x for x in per_claim if x[1] and x[2]][:2]
    )
    special = {c.id for c, _, _ in pick}
    for c, b, d in per_claim:  # make sure the Lesson 1 "SMA vs anemia" claim is in
        if c.id == 41 and c.id not in special:
            pick.append((c, b, d))
    show_docs: set[int] = set(rng.sample(doc_ids, 350))
    claims_out = []
    for c, b, d in pick:
        qv = encoder.encode([c.claim])
        dpos = dense.search_vectors(qv, 5)[1][0]
        dtop = [doc_ids[p] for p in dpos]
        btop = bm25.rank(c.claim, k=5)
        gold = sorted(c.evidence)
        show_docs |= set(dtop) | set(btop) | set(gold)
        xy = ((qv[0] - mean) @ axes).tolist()
        claims_out.append(
            {
                "id": c.id,
                "text": c.claim,
                "xy": xy,
                "gold": gold,
                "dense": dtop,
                "bm25": btop,
                "bm25_hit10": b,
                "dense_hit10": d,
            }
        )
    pos = {d: i for i, d in enumerate(doc_ids)}
    docs_out = [
        {"id": d, "title": corpus[d].title[:110], "xy": ((vecs[pos[d]] - mean) @ axes).tolist()}
        for d in sorted(show_docs)
    ]
    path.write_text(
        json.dumps({"explained": explained, "claims": claims_out, "docs": docs_out}),
        encoding="utf-8",
    )
    print(f"\n  exported {len(docs_out)} abstracts + {len(claims_out)} claims -> {path}")
    print(f"  2D projection keeps {explained:.1%} of the variance of the 384-D vectors")


if __name__ == "__main__":
    raise SystemExit(main())
