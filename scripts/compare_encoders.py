"""Choose the dense encoder on evidence: every candidate, same harness, TRAIN ONLY.

Each candidate runs with its own model-card conventions (`dense.SPECS`). None is trained here,
so all 809 train claims are usable; dev is never touched. Choosing among m candidates on one
split inflates the winner by roughly sigma * sqrt(2 ln m) (docs/concepts/05), so the report
states m and the interval with every number.

Contamination (OQ-014): MTEB's "SciFact test" set IS our dev set. A model whose card reports an
MTEB SciFact score is flagged. It can still be measured here on train, but its numbers carry
that flag into every later decision.

    python scripts/compare_encoders.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from scifact.data.schema import load_claims, load_corpus  # noqa: E402
from scifact.eval.harness import RetrievalResult, evaluate_document_retrieval  # noqa: E402
from scifact.retrieval.bm25 import build_index  # noqa: E402
from scifact.retrieval.dense import SPECS, DenseIndex, Encoder  # noqa: E402

DATA = REPO_ROOT / "data"
OUT = REPO_ROOT / "outputs"

# From scripts/survey_encoders.py, 2026-10-14: the card reports a score on MTEB SciFact (= our dev).
MTEB_SCIFACT_IN_CARD = {"bge-small-en-v1.5", "e5-base-v2"}
DOMAIN = {"S-PubMedBert-MS-MARCO": "biomedical", "MedCPT": "biomedical"}


def main() -> int:
    corpus = load_corpus(DATA / "corpus.jsonl")
    claims = load_claims(DATA / "claims_train.jsonl")
    doc_ids = sorted(corpus)
    docs: list[str | tuple[str, str]] = [
        (corpus[i].title, " ".join(corpus[i].abstract)) for i in doc_ids
    ]
    OUT.mkdir(exist_ok=True)

    bm25 = build_index(
        doc_ids, [f"{corpus[i].title} {' '.join(corpus[i].abstract)}" for i in doc_ids]
    )
    rows: list[tuple[str, RetrievalResult, int | None, int | None, float | None]] = [
        (
            "BM25 (reference)",
            evaluate_document_retrieval(bm25, claims, "train", "BM25"),
            None,
            None,
            None,
        )
    ]

    for key, spec in SPECS.items():
        print(f"\n--- {key}", flush=True)
        enc = Encoder(spec=spec)
        cache = OUT / f"corpus_vectors_{key}.npy"
        t0 = time.perf_counter()
        if cache.exists():
            vecs = np.load(cache)
        else:
            vecs = enc.encode(docs, "doc", batch_size=32)
            np.save(cache, vecs)
        enc_s = time.perf_counter() - t0
        index = DenseIndex(enc, doc_ids, vecs)
        res = evaluate_document_retrieval(index, claims, "train", key)
        n_params = sum(p.numel() for p in enc.model.parameters())
        if spec.doc_model:
            n_params += sum(p.numel() for p in enc.doc_encoder.parameters())
        rows.append((key, res, vecs.shape[1], n_params, enc_s))
        print(
            f"    hit@1 {res.hit_at[1]}   hit@10 {res.hit_at[10].value:.1%}   "
            f"MRR {res.mrr.value:.3f}   dim {vecs.shape[1]}   corpus encode {enc_s:.0f}s"
        )
        del enc, index
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print(
        f"ENCODER COMPARISON -- train, n={rows[0][1].n_claims} claims with evidence, "
        f"5,183 abstracts, m={len(SPECS)} candidates"
    )
    print("=" * 100)
    print(f"  {'model':<24} {'hit@1':>22} {'hit@10':>7} {'MRR':>6} {'dim':>4} {'params':>7}  flags")
    out = []
    for name, res, dim, params, secs in rows:
        flags = []
        if name in MTEB_SCIFACT_IN_CARD:
            flags.append("CARD REPORTS MTEB SciFact (= our dev)")
        if name in DOMAIN:
            flags.append(DOMAIN[name])
        print(
            f"  {name:<24} {res.hit_at[1]!s:>22} {res.hit_at[10].value:>7.1%} "
            f"{res.mrr.value:>6.3f} {dim or '-':>4} "
            f"{(f'{params / 1e6:.0f}M' if params else '-'):>7}  {'; '.join(flags)}"
        )
        out.append(
            {
                "model": name,
                "hit1": res.hit_at[1].value,
                "hit1_lo": res.hit_at[1].low,
                "hit1_hi": res.hit_at[1].high,
                "hit10": res.hit_at[10].value,
                "mrr": res.mrr.value,
                "dim": dim,
                "params": params,
                "encode_s": secs,
                "flags": flags,
            }
        )
    (OUT / "encoder_comparison.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  Selection inflation at m={len(SPECS)}: roughly sigma*sqrt(2 ln m) on the winner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
