"""Dense retrieval: text -> one vector, then nearest-neighbour search with FAISS.

Two pieces, kept separate because they answer different questions:

* `Encoder` turns text into a vector. A bi-encoder: claim and abstract are encoded
  SEPARATELY, never seeing each other. That is the property that makes indexing possible --
  every abstract can be encoded once, ahead of time, before any claim exists.
* `DenseIndex` finds the stored vectors closest to a query vector. FAISS does only this.

Same `rank(query, k)` interface as BM25, so the Stage 1 harness scores both identically.
"""

from __future__ import annotations

from collections.abc import Sequence

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


class Encoder:
    """Mean-pooled transformer embeddings, L2-normalised."""

    def __init__(self, name: str = DEFAULT_ENCODER, device: str | None = None) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.name = name

    @torch.no_grad()
    def encode(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        """Return a (len(texts), hidden) float32 array, one unit-length row per text."""
        rows: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            enc = self.tokenizer(
                list(texts[start : start + batch_size]),
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            tokens = self.model(**enc).last_hidden_state  # (batch, seq_len, hidden)
            # Mean pooling: average the token vectors, but only over REAL tokens. Padding
            # positions are zeroed by the mask so they cannot drag the average.
            mask = enc["attention_mask"].unsqueeze(-1).float()  # (batch, seq_len, 1)
            summed = (tokens * mask).sum(dim=1)  # (batch, hidden)
            mean = summed / mask.sum(dim=1).clamp(min=1.0)
            # Unit length, so that a dot product IS the cosine similarity.
            unit = torch.nn.functional.normalize(mean, p=2, dim=1)
            rows.append(unit.float().cpu().numpy())
        return np.concatenate(rows).astype(np.float32)


class DenseIndex:
    """Exact inner-product search over unit vectors (= cosine similarity)."""

    def __init__(self, encoder: Encoder, doc_ids: Sequence[int], doc_vectors: np.ndarray) -> None:
        if len(doc_ids) != doc_vectors.shape[0]:
            raise ValueError("one vector per document id")
        self.encoder = encoder
        self.doc_ids = list(doc_ids)
        # IndexFlatIP = "flat" (compare against every stored vector, no approximation) +
        # "IP" (inner product). Exact search over 5,183 vectors takes milliseconds; the
        # approximate indexes FAISS is famous for only pay off at millions of vectors.
        self.index = faiss.IndexFlatIP(doc_vectors.shape[1])
        self.index.add(doc_vectors)

    def search_vectors(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """(scores, positions), each shaped (n_queries, k), best first."""
        scores, positions = self.index.search(queries, k)
        return scores, positions

    def rank(self, query: str, k: int | None = None) -> list[int]:
        k = len(self.doc_ids) if k is None else k
        _scores, positions = self.search_vectors(self.encoder.encode([query]), k)
        return [self.doc_ids[p] for p in positions[0]]
