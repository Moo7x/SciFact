"""Wrap the fine-tuned cross-encoder as a `Verifier`, so Stage 2's decomposition applies to it.

The model scores one (claim, sentence) pair at a time. A claim-level verdict needs an
aggregation rule over the candidate sentences, and the rule is a real design choice:

* **Most-decisive sentence** (used here): score every candidate, then take the verdict of the
  sentence with the highest max(P(SUPPORT), P(CONTRADICT)). Rationale: evidence is a claim about
  a *particular* sentence, and a single decisive sentence is exactly what a rationale is. Median
  rationale length in this dataset is one sentence, so this matches the annotation unit.

* **Mean pooling** (rejected): averaging probabilities across all candidates lets 20 irrelevant
  sentences drown one decisive one. With retrieved@3 supplying ~24 sentences, that is the
  common case rather than the exception.

The rule is stated here because it is invisible in the metrics and changes them substantially.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from scifact.verify.dataset import ID_TO_LABEL
from scifact.verify.labels import NEI, Verdict

MAX_LENGTH = 256
BATCH = 32


class CrossEncoderVerifier:
    """Fine-tuned cross-encoder, aggregated to a claim-level verdict."""

    def __init__(
        self,
        model_dir: Path,
        device: str | None = None,
        decision_threshold: float = 0.0,
    ) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device).eval()
        self.decision_threshold = decision_threshold
        self._name = f"cross-encoder (tau={decision_threshold:.2f})"

    @property
    def name(self) -> str:
        return self._name

    @torch.no_grad()
    def score(self, claim: str, sentences: Sequence[str]) -> torch.Tensor:
        """Return an (n_sentences, 3) tensor of class probabilities."""
        out: list[torch.Tensor] = []
        for start in range(0, len(sentences), BATCH):
            chunk = list(sentences[start : start + BATCH])
            enc = self.tokenizer(
                [claim] * len(chunk),
                chunk,
                truncation=True,
                max_length=MAX_LENGTH,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            logits = self.model(**enc).logits
            out.append(torch.softmax(logits, dim=-1).cpu())
        return torch.cat(out) if out else torch.empty(0, 3)

    def predict(self, claim: str, sentences: Sequence[str]) -> Verdict:
        if not sentences:
            return NEI
        probs = self.score(claim, sentences)
        # Most-decisive sentence: the one most confident about *something other than* NEI.
        scores = probs[:, :2].max(dim=-1).values
        decisive = int(scores.argmax().item())

        # A threshold is required here, and its absence was a real bug. Taking a max over ~24
        # candidate sentences is the selection-bias problem from docs/concepts/05 wearing a
        # different hat: with 24 noisy draws, at least one will look confident by chance, so an
        # unconditional argmax almost never abstains. Measured effect: pair-level NEI recall
        # 80.4% collapsed to 32.8% at claim level purely from this aggregation.
        if float(scores[decisive]) < self.decision_threshold:
            return NEI
        return ID_TO_LABEL[int(probs[decisive].argmax().item())]
