"""Dense retrieval: text -> one vector, then nearest-neighbour search with FAISS.

Two pieces, kept separate because they answer different questions:

* `Encoder` turns text into a vector. A bi-encoder: claim and abstract are encoded
  SEPARATELY, never seeing each other. That is the property that makes indexing possible --
  every abstract can be encoded once, ahead of time, before any claim exists.
* `DenseIndex` finds the stored vectors closest to a query vector. FAISS does only this.

Every embedding model ships with input CONVENTIONS, and ignoring them silently handicaps it:
how token vectors are pooled into one (mean vs the first [CLS] token), whether vectors are
length-normalised, a maximum length, text prefixes such as "query: " / "passage: ", and
sometimes a separate model for documents than for queries (asymmetric encoders such as MedCPT).
`EncoderSpec` holds those conventions; the presets in `SPECS` are taken from each model card.

Same `rank(query, k)` interface as BM25, so the Stage 1 harness scores both identically.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"

Text = str | tuple[str, str]  # a tuple is (title, body), for models that take them as a pair


@dataclass(frozen=True)
class EncoderSpec:
    """One model's input conventions, as stated in its model card."""

    query_model: str
    doc_model: str | None = None  # None = same model for both (symmetric)
    pooling: Literal["mean", "cls"] = "mean"
    normalize: bool = True  # unit length, so a dot product IS cosine similarity
    query_max_length: int = 256
    doc_max_length: int = 256
    query_prefix: str = ""
    doc_prefix: str = ""
    doc_as_pair: bool = False  # feed (title, abstract) as two segments instead of one string


SPECS: dict[str, EncoderSpec] = {
    "all-MiniLM-L6-v2": EncoderSpec("sentence-transformers/all-MiniLM-L6-v2"),
    "all-mpnet-base-v2": EncoderSpec(
        "sentence-transformers/all-mpnet-base-v2", query_max_length=384, doc_max_length=384
    ),
    "bge-small-en-v1.5": EncoderSpec(
        "BAAI/bge-small-en-v1.5",
        pooling="cls",
        query_max_length=512,
        doc_max_length=512,
        query_prefix="Represent this sentence for searching relevant passages: ",
    ),
    "e5-base-v2": EncoderSpec(
        "intfloat/e5-base-v2",
        query_max_length=512,
        doc_max_length=512,
        query_prefix="query: ",
        doc_prefix="passage: ",
    ),
    "S-PubMedBert-MS-MARCO": EncoderSpec(
        "pritamdeka/S-PubMedBert-MS-MARCO", query_max_length=350, doc_max_length=350
    ),
    "MedCPT": EncoderSpec(
        "ncbi/MedCPT-Query-Encoder",
        doc_model="ncbi/MedCPT-Article-Encoder",
        pooling="cls",
        normalize=False,  # the card scores with a raw dot product of [CLS] vectors
        query_max_length=64,
        doc_max_length=512,
        doc_as_pair=True,
    ),
}


class Encoder:
    """Pooled transformer embeddings, following one model's conventions."""

    def __init__(
        self,
        name: str = DEFAULT_ENCODER,
        device: str | None = None,
        spec: EncoderSpec | None = None,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.spec = spec or EncoderSpec(name)
        self.name = self.spec.query_model
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.query_model)
        self.model = AutoModel.from_pretrained(self.spec.query_model).to(self.device).eval()
        if self.spec.doc_model:
            self.doc_tokenizer = AutoTokenizer.from_pretrained(self.spec.doc_model)
            self.doc_encoder = AutoModel.from_pretrained(self.spec.doc_model).to(self.device).eval()
        else:
            self.doc_tokenizer, self.doc_encoder = self.tokenizer, self.model

    @torch.no_grad()
    def encode(
        self, texts: Sequence[Text], kind: Literal["query", "doc"] = "query", batch_size: int = 64
    ) -> np.ndarray:
        """Return a (len(texts), hidden) float32 array, one row per text."""
        is_doc = kind == "doc"
        tok = self.doc_tokenizer if is_doc else self.tokenizer
        model = self.doc_encoder if is_doc else self.model
        prefix = self.spec.doc_prefix if is_doc else self.spec.query_prefix
        max_length = self.spec.doc_max_length if is_doc else self.spec.query_max_length

        rows: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start : start + batch_size])
            if is_doc and self.spec.doc_as_pair:
                firsts = [t[0] if isinstance(t, tuple) else t for t in chunk]
                seconds = [t[1] if isinstance(t, tuple) else "" for t in chunk]
                enc = tok(
                    firsts,
                    seconds,
                    truncation="only_second",
                    max_length=max_length,
                    padding=True,
                    return_tensors="pt",
                )
            else:
                flat = [prefix + (" ".join(t) if isinstance(t, tuple) else t) for t in chunk]
                enc = tok(
                    flat, truncation=True, max_length=max_length, padding=True, return_tensors="pt"
                )
            enc = enc.to(self.device)
            tokens = model(**enc).last_hidden_state  # (batch, seq_len, hidden)
            if self.spec.pooling == "cls":
                pooled = tokens[:, 0]  # the [CLS] position (Lesson 3)
            else:
                # Mean pooling over REAL tokens only: padding is zeroed by the mask so it cannot
                # drag the average.
                mask = enc["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)
                pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            if self.spec.normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            rows.append(pooled.float().cpu().numpy())
        return np.concatenate(rows).astype(np.float32)


class DenseIndex:
    """Exact inner-product search. Over unit vectors that is cosine similarity."""

    def __init__(self, encoder: Encoder, doc_ids: Sequence[int], doc_vectors: np.ndarray) -> None:
        if len(doc_ids) != doc_vectors.shape[0]:
            raise ValueError("one vector per document id")
        self.encoder = encoder
        self.doc_ids = list(doc_ids)
        # IndexFlatIP = "flat" (compare against every stored vector, no approximation) +
        # "IP" (inner product). It stores the raw float32 vectors and nothing else -- no text,
        # no ids: results come back as row positions, which `doc_ids` maps back to documents.
        self.index = faiss.IndexFlatIP(doc_vectors.shape[1])
        self.index.add(doc_vectors)

    def search_vectors(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(scores, positions), each shaped (n_queries, k), best first."""
        scores, positions = self.index.search(queries, k)
        return scores, positions

    def rank(self, query: str, k: int | None = None) -> list[int]:
        k = len(self.doc_ids) if k is None else k
        _scores, positions = self.search_vectors(self.encoder.encode([query], "query"), k)
        return [self.doc_ids[p] for p in positions[0]]
