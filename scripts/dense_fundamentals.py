"""Measure the interview-grade fundamentals of dense retrieval on SciFact.

Part A -- what an index stores, and what approximate search trades away.
  Flat, IVF, HNSW and PQ indexes over the same 5,183 vectors: bytes stored, build time, query
  time (warmed up, averaged), how much of the EXACT top-10 each one recovers, and the end metric
  (gold abstract in top 10).

Part B -- chunk size and truncation.
  * Truncation: an encoder has a maximum input length. How many abstracts exceed it, and how
    many gold evidence sentences sit partly or wholly past the cut, unseen by the encoder?
  * Granularity: one vector per abstract vs one per sentence vs one per 3-sentence window,
    scored with the same harness. A document's score is the best score of its chunks.

Train split only. Dev untouched.

    python scripts/dense_fundamentals.py
"""

from __future__ import annotations

import sys
import tempfile
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import faiss  # noqa: E402
import numpy as np  # noqa: E402

from scifact.data.schema import load_claims, load_corpus  # noqa: E402
from scifact.eval.harness import evaluate_document_retrieval  # noqa: E402
from scifact.retrieval.dense import SPECS, Encoder  # noqa: E402

DATA = REPO_ROOT / "data"
OUT = REPO_ROOT / "outputs"
KEY = "all-MiniLM-L6-v2"


def timed(fn: Callable[[], object], reps: int = 5) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000


def index_bytes(index: faiss.Index) -> int:
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "i.faiss")
        faiss.write_index(index, path)
        return Path(path).stat().st_size


class ChunkRetriever:
    """Rank documents by their BEST-matching chunk."""

    def __init__(self, enc: Encoder, chunk_vecs: np.ndarray, chunk_doc: list[int]) -> None:
        self.enc, self.chunk_doc = enc, chunk_doc
        self.index = faiss.IndexFlatIP(chunk_vecs.shape[1])
        self.index.add(chunk_vecs)

    def rank(self, query: str, k: int | None = None) -> list[int]:
        qv = self.enc.encode([query], "query")
        _s, pos = self.index.search(qv, min(400, self.index.ntotal))
        seen: list[int] = []
        for p in pos[0]:
            d = self.chunk_doc[p]
            if d not in seen:
                seen.append(d)
            if k is not None and len(seen) >= k:
                break
        return seen


def main() -> int:
    corpus = load_corpus(DATA / "corpus.jsonl")
    claims = load_claims(DATA / "claims_train.jsonl")
    doc_ids = sorted(corpus)
    enc = Encoder(spec=SPECS[KEY])
    vecs = np.load(OUT / f"corpus_vectors_{KEY}.npy")
    scored = [c for c in claims if c.has_evidence]
    q = enc.encode([c.claim for c in scored], "query")
    gold = [set(c.evidence) for c in scored]
    n, d = vecs.shape

    # ================================================================== PART A
    print("=" * 96)
    print(f"PART A -- what each FAISS index stores ({n:,} vectors x {d} dims, {KEY})")
    print("=" * 96)
    flat = faiss.IndexFlatIP(d)
    flat.add(vecs)
    _s, exact = flat.search(q, 10)
    print(f"  raw vectors alone: {n:,} x {d} x 4 bytes = {n * d * 4 / 1e6:.2f} MB")
    v0 = flat.reconstruct(0)
    print(
        f"  Flat stores the vectors verbatim: reconstruct(0) == vector 0 ? "
        f"{bool(np.array_equal(v0, vecs[0]))}"
    )

    def build_ivf(nprobe: int) -> faiss.Index:
        quant = faiss.IndexFlatIP(d)
        ivf = faiss.IndexIVFFlat(quant, d, 64, faiss.METRIC_INNER_PRODUCT)
        ivf.train(vecs)
        ivf.add(vecs)
        ivf.nprobe = nprobe
        return ivf

    def build_hnsw() -> faiss.Index:
        h = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
        h.add(vecs)
        return h

    def build_pq() -> faiss.Index:
        pq = faiss.IndexPQ(d, 48, 8, faiss.METRIC_INNER_PRODUCT)
        pq.train(vecs)
        pq.add(vecs)
        return pq

    builders: list[tuple[str, Callable[[], faiss.Index], str]] = [
        ("Flat (exact)", lambda: flat, "every vector, uncompressed"),
        ("IVF, probe 1/64", lambda: build_ivf(1), "vectors grouped under 64 k-means centroids"),
        ("IVF, probe 8/64", lambda: build_ivf(8), "same, searches 8 nearest groups"),
        ("HNSW (M=32)", build_hnsw, "vectors + a navigable neighbour graph"),
        ("PQ (48 x 8 bit)", build_pq, "48-byte compressed codes, NOT the vectors"),
    ]
    print(
        f"\n  {'index':<17} {'stored':>9} {'build':>8} {'query':>8} {'exact top-10 kept':>18} "
        f"{'gold in top 10':>15}   what it stores"
    )
    for name, build, what in builders:
        t0 = time.perf_counter()
        idx = build()
        build_ms = (time.perf_counter() - t0) * 1000
        q_ms = timed(partial(idx.search, q, 10))
        _s, got = idx.search(q, 10)
        overlap = np.mean([len(set(a) & set(b)) / 10 for a, b in zip(got, exact, strict=True)])
        hit = np.mean(
            [any(doc_ids[p] in g for p in row if p >= 0) for row, g in zip(got, gold, strict=True)]
        )
        print(
            f"  {name:<17} {index_bytes(idx) / 1e6:>7.2f}MB {build_ms:>6.0f}ms {q_ms:>6.1f}ms "
            f"{overlap:>17.1%} {hit:>15.1%}   {what}"
        )

    # ================================================================== PART B
    print("\n" + "=" * 96)
    print("PART B -- truncation and chunk size")
    print("=" * 96)
    tok = enc.tokenizer
    limit = SPECS[KEY].doc_max_length
    lengths, cut_evidence, total_evidence = [], 0, 0
    for doc_id in doc_ids:
        doc = corpus[doc_id]
        lengths.append(len(tok(doc.title + " " + " ".join(doc.abstract))["input_ids"]))
    for c in scored:
        for doc_id, rats in c.evidence.items():
            doc = corpus[doc_id]
            for r in rats:
                last = max(r.sentences)
                prefix = doc.title + " " + " ".join(doc.abstract[: last + 1])
                total_evidence += 1
                cut_evidence += len(tok(prefix)["input_ids"]) > limit
    over = sum(limit < L for L in lengths)
    print(f"  encoder max length: {limit} tokens")
    print(
        f"  abstracts longer than that: {over:,} of {n:,} ({over / n:.1%})  "
        f"median length {int(np.median(lengths))}, p95 {int(np.percentile(lengths, 95))}"
    )
    print(
        f"  gold rationales that END past the cut (encoder never sees them whole): "
        f"{cut_evidence} of {total_evidence} ({cut_evidence / total_evidence:.1%})"
    )

    def chunks(size: int) -> tuple[list[str], list[int]]:
        texts, owners = [], []
        for doc_id in doc_ids:
            doc = corpus[doc_id]
            sents = list(doc.abstract)
            if size == 0:
                texts.append(doc.title + " " + " ".join(sents))
                owners.append(doc_id)
                continue
            for i in range(0, max(1, len(sents) - size + 1)):
                texts.append(doc.title + ". " + " ".join(sents[i : i + size]))
                owners.append(doc_id)
        return texts, owners

    print(f"\n  {'granularity':<28} {'chunks':>7} {'hit@1':>22} {'hit@10':>7} {'MRR':>6}")
    for label, size in (
        ("whole abstract", 0),
        ("1 sentence (+title)", 1),
        ("3-sentence window", 3),
    ):
        texts, owners = chunks(size)
        t0 = time.perf_counter()
        cv = vecs if size == 0 else enc.encode(texts, "doc", batch_size=128)
        enc_s = time.perf_counter() - t0
        res = evaluate_document_retrieval(ChunkRetriever(enc, cv, owners), claims, "train", label)
        print(
            f"  {label:<28} {len(texts):>7,} {res.hit_at[1]!s:>22} "
            f"{res.hit_at[10].value:>7.1%} {res.mrr.value:>6.3f}   (encode {enc_s:.0f}s)"
        )
    print("\n  Each sentence chunk carries the title, so it keeps the abstract's topic. A chunk")
    print("  that loses its context is a different, worse experiment, not 'sentence retrieval'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
