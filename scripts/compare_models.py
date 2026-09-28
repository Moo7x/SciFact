"""Audit the Stage 3 model choice against the alternatives, on this project's own data.

The original choice (cross-encoder/ms-marco-MiniLM-L-6-v2) was made from memory, without a
survey. This script is the survey, done afterwards and reported as such. It measures three
things a model card cannot tell you about THIS task:

1. Head fit      -- zero-shot on the 547 train_tune pairs used for Stage 3's pair-level
                    confusion matrix. NLI models already have a 3-way head
                    (entailment / contradiction / neutral) that maps onto
                    SUPPORT / CONTRADICT / NOT_ENOUGH_INFO with no training at all.
2. Vocabulary    -- tokens per word on SciFact text (the OQ-012 test), and how specific
                    biomedical terms are split.
3. Hardware      -- peak GPU memory and time for ONE full training step (forward, backward,
                    AdamW update) at the worst-case length of 256, batch 16. Measured, not
                    estimated from parameter counts.

NLI convention: the model reads (premise, hypothesis) = (evidence, claim). Stage 3 fed
(claim, evidence). Order matters to a model trained on one convention.

Usage::

    python scripts/compare_models.py
"""

from __future__ import annotations

import gc
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from scifact.data.schema import load_claims, load_corpus  # noqa: E402
from scifact.data.splits import split_train  # noqa: E402
from scifact.eval.metrics import bootstrap  # noqa: E402
from scifact.verify.dataset import build_pairs  # noqa: E402

hf_logging.set_verbosity_error()
DATA = REPO_ROOT / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NLI_TO_OURS = {"entailment": "SUPPORT", "contradiction": "CONTRADICT", "neutral": "NOT_ENOUGH_INFO"}
CLASSES = ("SUPPORT", "CONTRADICT", "NOT_ENOUGH_INFO")

ZERO_SHOT = [
    "cross-encoder/nli-MiniLM2-L6-H768",
    "cross-encoder/nli-deberta-v3-xsmall",
    "cross-encoder/nli-deberta-v3-small",
    "pritamdeka/PubMedBERT-MNLI-MedNLI",
]
TOKENIZERS = {
    "ms-marco-MiniLM (current)": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "nli-MiniLM2 (RoBERTa BPE)": "cross-encoder/nli-MiniLM2-L6-H768",
    "nli-deberta-v3 (SentencePiece)": "cross-encoder/nli-deberta-v3-xsmall",
    "PubMedBERT (biomedical)": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
    "SciBERT (scientific)": "allenai/scibert_scivocab_uncased",
}
TRAINABLE = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "cross-encoder/nli-MiniLM2-L6-H768",
    "cross-encoder/nli-deberta-v3-xsmall",
    "cross-encoder/nli-deberta-v3-small",
    "pritamdeka/PubMedBERT-MNLI-MedNLI",
]
TERMS = ("microerythrocyte", "erythrocyte", "homozygous", "thalassemia", "haematological")


def free() -> None:
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()


@torch.no_grad()
def zero_shot(name: str, pairs: list) -> dict:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name).to(DEVICE).eval()
    id2label = {int(k): v.lower() for k, v in model.config.id2label.items()}
    preds: list[str] = []
    for i in range(0, len(pairs), 32):
        chunk = pairs[i : i + 32]
        enc = tok(
            [p.evidence for p in chunk],  # premise
            [p.claim for p in chunk],  # hypothesis
            truncation=True,
            max_length=256,
            padding=True,
            return_tensors="pt",
        ).to(DEVICE)
        ids = model(**enc).logits.argmax(-1).tolist()
        preds += [NLI_TO_OURS[id2label[j]] for j in ids]
    del model
    free()
    gold = [p.label for p in pairs]
    rec = {}
    for c in CLASSES:
        idx = [i for i, g in enumerate(gold) if g == c]
        rec[c] = sum(preds[i] == c for i in idx) / len(idx)
    acc = bootstrap([float(p == g) for p, g in zip(preds, gold, strict=True)])
    return {"acc": acc, "rec": rec, "macro": statistics.mean(rec.values())}


def pieces(tok: Any, word: str) -> int:
    """Pieces a word takes mid-sentence. Tokenizing it alone mis-states BPE/SentencePiece."""
    return len(tok.tokenize("the " + word)) - len(tok.tokenize("the"))


def train_step_cost(name: str, pairs: list) -> tuple[float, float] | str:
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=3, ignore_mismatched_sizes=True
    ).to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    batch = tok(
        [p.claim for p in pairs[:16]],
        [p.evidence for p in pairs[:16]],
        truncation=True,
        max_length=256,
        padding="max_length",
        return_tensors="pt",
    ).to(DEVICE)
    labels = torch.tensor([p.label_id for p in pairs[:16]], device=DEVICE)
    try:
        for step in range(3):  # step 0 warms up kernels; steps 1-2 are timed
            if step == 1:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
            loss = torch.nn.functional.cross_entropy(model(**batch).logits, labels)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 2 * 1000
        peak = torch.cuda.max_memory_allocated() / 2**30
        return peak, ms
    except torch.cuda.OutOfMemoryError:
        return "OUT OF MEMORY"
    finally:
        del model, opt
        free()


def main() -> int:
    corpus = load_corpus(DATA / "corpus.jsonl")
    train = load_claims(DATA / "claims_train.jsonl")
    _fit, tune = split_train(train)
    pairs = build_pairs(tune, corpus, max_negatives_per_claim=2, seed=20260919)
    n = {c: sum(p.label == c for p in pairs) for c in CLASSES}
    print(f"device={DEVICE.type}   tune pairs={len(pairs)}   {n}")

    # ------------------------------------------------------------------ 1. head fit
    print("\n" + "=" * 92)
    print("1. ZERO-SHOT -- no training at all, same 547 train_tune pairs as the Stage 3 matrix")
    print("=" * 92)
    print(f"  {'model':<40} {'accuracy':>22}  {'SUPPORT':>8} {'CONTRA':>8} {'NEI':>8} {'macro':>7}")
    print(
        f"  {'majority (always NEI on pairs)':<40} {n['NOT_ENOUGH_INFO'] / len(pairs):>21.1%}  "
        f"{0:>8.1%} {0:>8.1%} {1:>8.1%} {1 / 3:>7.1%}"
    )
    print(
        f"  {'Stage 3 fine-tuned ms-marco (baseline)':<40} {'--':>22}  "
        f"{0.224:>8.1%} {0.797:>8.1%} {0.799:>8.1%} {0.607:>7.1%}   <- after training"
    )
    for name in ZERO_SHOT:
        r = zero_shot(name, pairs)
        print(
            f"  {name.split('/')[-1]:<40} {r['acc']!s:>22}  {r['rec']['SUPPORT']:>8.1%} "
            f"{r['rec']['CONTRADICT']:>8.1%} {r['rec']['NOT_ENOUGH_INFO']:>8.1%} {r['macro']:>7.1%}"
        )
    print(
        "  cross-encoder/ms-marco-MiniLM-L-6-v2 cannot be scored zero-shot: its head has one output"
        " (a relevance score), not three."
    )

    # ------------------------------------------------------------------ 2. vocabulary
    print("\n" + "=" * 92)
    print("2. VOCABULARY -- how each tokenizer breaks up SciFact text (OQ-012)")
    print("=" * 92)
    texts = [c.claim for c in train]
    for c in train:
        for d, rs in c.evidence.items():
            for rat in rs:
                texts += [
                    corpus[d].abstract[i] for i in rat.sentences if i < len(corpus[d].abstract)
                ]
    words = sum(len(t.split()) for t in texts)
    print(f"  {len(texts):,} texts (train claims + gold rationale sentences), {words:,} words\n")
    print(
        f"  {'tokenizer':<32} {'vocab':>7} {'tokens/word':>12}   "
        + "  ".join(f"{t[:11]:>11}" for t in TERMS)
    )
    for label, name in TOKENIZERS.items():
        tok = AutoTokenizer.from_pretrained(name)
        tpw = sum(len(tok.tokenize(t)) for t in texts) / words
        per = "  ".join(f"{pieces(tok, t):>11}" for t in TERMS)
        print(f"  {label:<32} {len(tok):>7,} {tpw:>12.2f}   {per}")
    print("\n  Per-term columns: pieces the word becomes mid-sentence. 1 = kept whole.")

    # ------------------------------------------------------------------ 3. hardware
    print("\n" + "=" * 92)
    print(f"3. ONE TRAINING STEP -- batch 16, length 256 (worst case), AdamW, {DEVICE.type}")
    print("=" * 92)
    if DEVICE.type != "cuda":
        print("  skipped: no GPU")
        return 0
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"  GPU memory available: {total:.2f} GiB\n")
    print(f"  {'model':<40} {'peak memory':>12} {'ms / step':>10}")
    for name in TRAINABLE:
        cost = train_step_cost(name, pairs)
        if isinstance(cost, str):
            print(f"  {name.split('/')[-1]:<40} {cost:>12}")
        else:
            print(f"  {name.split('/')[-1]:<40} {cost[0]:>9.2f} GiB {cost[1]:>10.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
