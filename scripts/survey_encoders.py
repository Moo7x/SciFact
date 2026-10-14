"""Survey candidate dense encoders: model cards, training data, and contamination risk.

Enhancement B (approved 2026-10-14). The Stage 3 dense baseline used all-MiniLM-L6-v2 because it
was already on disk, which is the same mistake ADR-0002 recorded for the verifier. This is the
survey that should have come first.

For each candidate it pulls Hub metadata and the model card, and searches the card for:
  * what the model was trained on (the datasets it names);
  * any mention of SciFact, BEIR or MTEB. SciFact is one of BEIR's datasets, and BEIR is part of
    MTEB, so a model *tuned* against those leaderboards may have been selected -- or trained --
    with SciFact in view;
  * input conventions (query/passage prefixes) that change results if ignored.

A card that does not mention SciFact is NOT evidence the model never saw it. It is the absence of
a statement, and the report says "not stated" rather than "clean".

    python scripts/survey_encoders.py
"""

from __future__ import annotations

import json
import re
import urllib.request
from typing import Any

CANDIDATES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "intfloat/e5-base-v2",
    "pritamdeka/S-PubMedBert-MS-MARCO",
    "NeuML/pubmedbert-base-embeddings",
    "ncbi/MedCPT-Query-Encoder",
    "allenai/specter2_base",
]

KEYWORDS = {
    "SciFact": r"sci[-_ ]?fact",
    "BEIR": r"\bbeir\b",
    "MTEB": r"\bmteb\b",
    "S2ORC": r"s2orc",
    "PubMed": r"pubmed",
    "MS MARCO": r"ms[ _-]?marco",
}


def fetch(url: str) -> str:
    try:
        with urllib.request.urlopen(url, timeout=30) as r:  # noqa: S310 - fixed Hub URLs
            return r.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return f"__ERROR__ {exc}"


def context(text: str, pattern: str, width: int = 70) -> str:
    m = re.search(pattern, text, re.IGNORECASE)
    if not m:
        return ""
    s = max(0, m.start() - width)
    snippet = text[s : m.end() + width].replace("\n", " ")
    return re.sub(r"\s+", " ", snippet)


def main() -> int:
    rows: list[dict[str, Any]] = []
    for name in CANDIDATES:
        meta_raw = fetch(f"https://huggingface.co/api/models/{name}")
        card = fetch(f"https://huggingface.co/{name}/raw/main/README.md")
        cfg_raw = fetch(f"https://huggingface.co/{name}/resolve/main/config.json")
        meta = json.loads(meta_raw) if not meta_raw.startswith("__ERROR__") else {}
        cfg = json.loads(cfg_raw) if not cfg_raw.startswith("__ERROR__") else {}
        params = (meta.get("safetensors") or {}).get("total")
        datasets = (meta.get("cardData") or {}).get("datasets") or []
        hits = {k: bool(re.search(p, card, re.IGNORECASE)) for k, p in KEYWORDS.items()}
        rows.append(
            {
                "model": name,
                "params_m": round(params / 1e6) if params else None,
                "dim": cfg.get("hidden_size"),
                "max_pos": cfg.get("max_position_embeddings"),
                "vocab": cfg.get("vocab_size"),
                "downloads": meta.get("downloads"),
                "license": (meta.get("cardData") or {}).get("license"),
                "card_datasets": len(datasets),
                "mentions": hits,
                "scifact_context": context(card, KEYWORDS["SciFact"]),
                "prefix_hint": context(card, r"(query: |passage: |Represent this sentence)", 40),
            }
        )

    print(
        f"{'model':<42} {'params':>6} {'dim':>4} {'vocab':>6} {'license':<12} "
        + " ".join(f"{k[:7]:>7}" for k in KEYWORDS)
    )
    for r in rows:
        flags = " ".join(f"{'yes' if r['mentions'][k] else '-':>7}" for k in KEYWORDS)
        print(
            f"{r['model']:<42} {str(r['params_m']) + 'M':>6} {r['dim']!s:>4} {r['vocab']!s:>6} "
            f"{str(r['license'])[:12]:<12} {flags}"
        )
    print("\nWhere a card mentions SciFact, the surrounding text:")
    for r in rows:
        if r["scifact_context"]:
            print(f"  {r['model']}:\n    ...{r['scifact_context']}...")
    print("\nInput conventions found in the cards:")
    for r in rows:
        if r["prefix_hint"]:
            print(f"  {r['model']}: ...{r['prefix_hint']}...")
    print("\nAlso listed in card metadata, number of training datasets declared:")
    for r in rows:
        print(f"  {r['model']:<42} {r['card_datasets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
