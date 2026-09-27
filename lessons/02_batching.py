"""Lesson 2 -- Dataset, DataLoader, padding: how (claim, evidence) pairs become batches.

Method: broken version first. Each part shows something failing or wasting work *before*
explaining it, and pauses so you can think about why before the explanation appears.

    .venv/Scripts/python.exe lessons/02_batching.py

Uses the same pairs, tokenizer and seed as scripts/train_crossencoder.py.
"""

from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from scifact.data.schema import load_claims, load_corpus  # noqa: E402
from scifact.data.splits import split_train  # noqa: E402
from scifact.verify.dataset import Pair, build_pairs  # noqa: E402

hf_logging.set_verbosity_error()

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 256
SEED = 20260919
BATCH = 16


def pause(prompt: str) -> None:
    input(f"\n  >>> {prompt}\n  --- press Enter to continue ---")


def header(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# --------------------------------------------------------------------------- datasets


class UnpaddedDataset(Dataset[dict[str, torch.Tensor]]):
    """The broken version: tokenizes each pair to its natural length. No padding."""

    def __init__(self, pairs: list[Pair], tok: AutoTokenizer) -> None:
        self.pairs, self.tok = pairs, tok

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        p = self.pairs[i]
        enc = self.tok(p.claim, p.evidence, truncation=True, max_length=MAX_LENGTH)
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(p.label_id)
        return item


class FixedPadDataset(UnpaddedDataset):
    """Fix 1: pad every pair to 256. Exactly what train_crossencoder.py does."""

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        p = self.pairs[i]
        enc = self.tok(
            p.claim, p.evidence, truncation=True, max_length=MAX_LENGTH, padding="max_length"
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(p.label_id)
        return item


def make_dynamic_collate(tok: AutoTokenizer):
    """Fix 2: leave items unpadded, and pad each BATCH to its own longest member."""

    def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        labels = torch.stack([it.pop("labels") for it in items])
        batch = tok.pad(
            [{k: v.tolist() for k, v in it.items()} for it in items], return_tensors="pt"
        )
        batch["labels"] = labels
        return dict(batch)

    return collate


# --------------------------------------------------------------------------- timing


def time_step(
    model: torch.nn.Module, batch: dict[str, torch.Tensor], device: torch.device
) -> float:
    """One forward + backward pass, the expensive part of a training step. Returns ms."""
    batch = {k: v.to(device) for k, v in batch.items()}
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model(**batch)
    out.loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1000
    model.zero_grad(set_to_none=True)
    return ms


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)
    corpus = load_corpus(REPO_ROOT / "data" / "corpus.jsonl")
    fit, _tune = split_train(load_claims(REPO_ROOT / "data" / "claims_train.jsonl"))
    pairs = build_pairs(fit, corpus, seed=SEED)

    # ======================================================================= PART 1
    header("PART 1 -- A Dataset is just 'give me item number i'")
    ds = UnpaddedDataset(pairs, tok)
    print(f"  len(ds) = {len(ds):,}   (one item per training pair)")
    for i in (0, 1, 2):
        item = ds[i]
        shapes = {k: tuple(v.shape) for k, v in item.items()}
        print(f"  ds[{i}] -> {shapes}")
    print(
        "\n  Two methods and nothing else: __len__ (how many) and __getitem__ (fetch one).\n"
        "  Each item is a dict of tensors. Look at the input_ids shapes above."
    )

    # ======================================================================= PART 2
    header("PART 2 -- Ask a DataLoader for a batch of 4 unpadded items")
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    try:
        next(iter(loader))
        print("  (no error -- unexpected)")
    except RuntimeError as exc:
        msg = str(exc).splitlines()[0]
        print(f"  RuntimeError: {msg}")

    pause("Why did it crash? Look at the shapes from Part 1.")
    print(
        "  A batch is not a list of items. It is ONE tensor of shape (batch_size, seq_len):\n"
        "  a rectangle. The DataLoader builds it with torch.stack, which lines rows up on top\n"
        "  of each other -- and rows of different lengths do not form a rectangle.\n\n"
        "  Why insist on a rectangle? Because the GPU's speed comes from doing one big matrix\n"
        "  multiply over the whole batch at once. A ragged list would mean one small multiply\n"
        "  per item, which throws away the reason the GPU is fast.\n\n"
        "  So every row in a batch must be padded to the same length. The only question is:\n"
        "  the same length as WHAT?"
    )

    # ======================================================================= PART 3
    header("PART 3 -- Fix 1: pad every pair to 256 (what my training script does)")
    fixed = DataLoader(FixedPadDataset(pairs, tok), batch_size=BATCH, shuffle=False)
    b = next(iter(fixed))
    print(f"  batch input_ids shape: {tuple(b['input_ids'].shape)}   -- works")

    lengths = [len(tok(p.claim, p.evidence)["input_ids"]) for p in pairs]
    real = sum(min(n, MAX_LENGTH) for n in lengths)
    total = len(lengths) * MAX_LENGTH
    over = sum(n > MAX_LENGTH for n in lengths)
    q = statistics.quantiles(lengths, n=20)
    print(f"\n  Real length of all {len(lengths):,} training pairs (tokens):")
    print(f"    median {statistics.median(lengths):.0f}   p95 {q[18]:.0f}   max {max(lengths)}")
    print(f"    pairs longer than 256 (get truncated): {over}")
    print(f"\n  Positions computed per epoch:   {total:,}")
    print(f"  ...of which are real text:      {real:,}  ({real / total:.0%})")
    print(f"  ...of which are padding:        {total - real:,}  ({1 - real / total:.0%})")

    pause("Fix 1 works. What does it cost? Compare the median with 256.")
    print(
        "  Every pair is stretched to 256 because 256 is the worst case. But the worst case\n"
        "  almost never happens -- look at the p95. Most of what the GPU computes each epoch\n"
        "  is multiplication by padding that attention_mask then tells the model to ignore."
    )

    # ======================================================================= PART 4
    header("PART 4 -- Fix 2: pad each batch only to ITS OWN longest pair")
    dyn = DataLoader(
        UnpaddedDataset(pairs, tok),
        batch_size=BATCH,
        shuffle=False,
        collate_fn=make_dynamic_collate(tok),
    )
    widths = [batch["input_ids"].shape[1] for batch in dyn]
    print(f"  first five batch shapes: {[(BATCH, w) for w in widths[:5]]}")
    print(f"  mean batch width: {statistics.mean(widths):.0f} tokens   (Fix 1 always uses 256)")

    # Timing on the same 20 batches of items, padded both ways.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=3, ignore_mismatched_sizes=True
    ).to(device)
    model.train()

    rng = random.Random(SEED)  # noqa: S311
    order = list(range(len(pairs)))
    rng.shuffle(order)
    groups = [order[i : i + BATCH] for i in range(0, 20 * BATCH, BATCH)]

    fixed_ds, unp_ds = FixedPadDataset(pairs, tok), UnpaddedDataset(pairs, tok)
    collate = make_dynamic_collate(tok)
    stack = torch.utils.data.default_collate

    time_step(model, stack([fixed_ds[i] for i in groups[0]]), device)  # warm-up, discarded
    t_fixed = [time_step(model, stack([fixed_ds[i] for i in g]), device) for g in groups]
    t_dyn = [time_step(model, collate([unp_ds[i] for i in g]), device) for g in groups]

    mf, md = statistics.mean(t_fixed), statistics.mean(t_dyn)
    print(f"\n  forward + backward on {device.type}, same 20 batches padded both ways:")
    print(f"    Fix 1, pad to 256:        {mf:6.1f} ms / batch")
    print(f"    Fix 2, pad to batch max:  {md:6.1f} ms / batch")
    print(f"    speed-up:                 {mf / md:.1f}x")

    pause("Same data, same model, same answer. Why is one faster?")
    print(
        "  attention_mask makes the RESULT identical: padded positions contribute nothing.\n"
        "  It does not make the WORK identical. The GPU still multiplies every position in\n"
        "  the rectangle. Fix 2 simply builds a smaller rectangle.\n\n"
        "  The general lesson: correctness and cost are separate properties. Code can be\n"
        "  exactly right and still spend most of its time on work that is thrown away."
    )

    header("END OF LESSON 2 -- one question, answer it in chat")
    print(
        "  Open scripts/train_crossencoder.py and find where padding happens.\n"
        "  1. Which line would you change, and to what?\n"
        "  2. Something else in the script then has to change too, or it will crash.\n"
        "     What is it? (Hint: Part 2.)"
    )


if __name__ == "__main__":
    main()
