"""Stage 3: fine-tune a cross-encoder to classify (claim, evidence) -> SUPPORT/CONTRADICT/NEI.

The training loop is written out rather than delegated to `Trainer`, because the loop is the
thing being learned here (Checkpoint 5). Every step is visible:

    for each epoch:
        for each batch:
            logits = model(batch)                 # forward
            loss   = cross_entropy(logits, y)     # how wrong, in nats
            loss.backward()                       # d(loss)/d(every parameter)
            clip_grad_norm_(...)                  # cap the step size
            optimizer.step()                      # move parameters downhill
            scheduler.step()                      # decay the learning rate
            optimizer.zero_grad()                 # gradients accumulate; clear them

Usage::

    python scripts/train_crossencoder.py --smoke          # 20 steps, verifies plumbing
    python scripts/train_crossencoder.py                  # the real run
    python scripts/train_crossencoder.py --epochs 4 --lr 3e-5
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from scifact.data.schema import load_claims, load_corpus  # noqa: E402
from scifact.data.splits import split_train  # noqa: E402
from scifact.verify.dataset import ID_TO_LABEL, Pair, build_pairs, label_counts  # noqa: E402

DATA_DIR = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "outputs"

# 22M params. At ~16 bytes/param for fp32 Adam that is ~0.35 GB before activations,
# comfortable on a 4 GB card and fine on CPU. Already trained for relevance ranking,
# so it starts closer to the task than a raw BERT would.
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 256  # claim + one evidence sentence; p95 of that pair fits well inside 256


class PairDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, pairs: list[Pair], tokenizer, max_length: int) -> None:
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        p = self.pairs[i]
        enc = self.tokenizer(
            p.claim,
            p.evidence,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(p.label_id, dtype=torch.long)
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def class_weights(pairs: list[Pair], device: torch.device) -> torch.Tensor:
    """Inverse-frequency weights, w_c = N / (K * n_c).

    Needed because 65% of training pairs are NOT_ENOUGH_INFO and only 12.6% are CONTRADICT.
    An unweighted cross-entropy on that distribution has an easy local optimum -- predict NEI
    always, score 65% -- and CONTRADICT contributes so little gradient that the class the
    project actually cares about (OQ-008) gets ignored.

    Weighting multiplies each example's loss by w_c, so a CONTRADICT mistake costs ~5x an NEI
    mistake. The alternative is resampling (repeat rare classes until balanced); weighting is
    chosen here because it keeps every example seen exactly once per epoch, which keeps the
    epoch boundary meaningful and avoids overfitting to duplicated rare examples on a dataset
    this small.

    The cost: the model's output probabilities are no longer calibrated to the true class
    priors. That matters in Stage 5 and is recorded there rather than discovered there.
    """
    counts = label_counts(pairs)
    total = len(pairs)
    k = len(counts)
    weights = [total / (k * max(1, counts[ID_TO_LABEL[i]])) for i in range(k)]
    return torch.tensor(weights, dtype=torch.float, device=device)


@torch.no_grad()
def evaluate(
    model, loader: DataLoader, device: torch.device, weights: torch.Tensor | None
) -> tuple[float, float, dict]:
    """Return (mean loss, accuracy, per-class recall). No gradients: eval must not train."""
    model.eval()
    total_loss, correct, seen = 0.0, 0, 0
    per_class: dict[int, list[int]] = {0: [0, 0], 1: [0, 0], 2: [0, 0]}  # [correct, total]

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        labels = batch.pop("labels")
        out = model(**batch)
        loss = torch.nn.functional.cross_entropy(out.logits, labels, weight=weights)
        total_loss += loss.item() * labels.size(0)
        batch["labels"] = labels
        preds = out.logits.argmax(dim=-1)
        for p, y in zip(preds.tolist(), batch["labels"].tolist(), strict=True):
            per_class[y][1] += 1
            per_class[y][0] += int(p == y)
            correct += int(p == y)
            seen += 1

    recalls = {
        ID_TO_LABEL[k]: (v[0] / v[1] if v[1] else float("nan")) for k, v in per_class.items()
    }
    model.train()
    return total_loss / seen, correct / seen, recalls


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup-frac", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--smoke", action="store_true", help="20 steps, to verify plumbing")
    parser.add_argument(
        "--no-class-weights", action="store_true", help="disable inverse-frequency weighting"
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    corpus = load_corpus(DATA_DIR / "corpus.jsonl")
    fit, tune = split_train(load_claims(DATA_DIR / "claims_train.jsonl"))
    train_pairs = build_pairs(fit, corpus, seed=args.seed)
    tune_pairs = build_pairs(tune, corpus, seed=args.seed)

    if args.smoke:
        train_pairs, tune_pairs = train_pairs[:320], tune_pairs[:96]

    print("=" * 74)
    print(f"STAGE 3 - cross-encoder fine-tune     device={device.type}")
    print("=" * 74)
    print(f"  model            {args.model}")
    print(f"  train pairs      {len(train_pairs):,}   {label_counts(train_pairs)}")
    print(f"  tune pairs       {len(tune_pairs):,}   {label_counts(tune_pairs)}")
    print(f"  epochs {args.epochs}  batch {args.batch_size}  lr {args.lr}  seed {args.seed}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, num_labels=3, ignore_mismatched_sizes=True
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters       {n_params / 1e6:.1f}M")

    train_loader = DataLoader(
        PairDataset(train_pairs, tokenizer, MAX_LENGTH),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    tune_loader = DataLoader(
        PairDataset(tune_pairs, tokenizer, MAX_LENGTH), batch_size=args.batch_size
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = max(1, int(total_steps * args.warmup_frac))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_at(step: int) -> float:
        """Linear warmup then linear decay.

        Warmup exists because Adam's second-moment estimate is near-garbage for the first few
        steps -- it has almost no history -- so a full-size step taken then can wreck pretrained
        weights before training has learned anything. Decay exists so late steps refine rather
        than bounce around the minimum.
        """
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    chance = 1 / 3
    print(
        f"\n  Random-guess loss = ln(3) = {math.log(3):.4f}. "
        f"A loss stuck there means nothing is being learned."
    )
    print(
        f"  Majority-class accuracy on tune = "
        f"{max(label_counts(tune_pairs).values()) / len(tune_pairs):.1%} "
        f"(chance = {chance:.1%})\n"
    )

    OUT_DIR.mkdir(exist_ok=True)
    history: list[dict[str, float]] = []
    step = 0
    t0 = time.perf_counter()

    weights = None if args.no_class_weights else class_weights(train_pairs, device)
    if weights is not None:
        pretty = "  ".join(f"{ID_TO_LABEL[i]}={weights[i]:.2f}" for i in range(len(weights)))
        print(f"  class weights    {pretty}")

    loss0, acc0, _ = evaluate(model, tune_loader, device, weights)
    print(f"\n  before training:  tune loss {loss0:.4f}   acc {acc0:.1%}")

    for epoch in range(1, args.epochs + 1):
        running: list[float] = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            labels = batch.pop("labels")
            outputs = model(**batch)  # forward: token ids -> 3 logits per example
            # Loss computed explicitly rather than letting the model do it, so the
            # weighting is visible and so the loss is not a black box in the one place
            # this script exists to make legible.
            loss = torch.nn.functional.cross_entropy(outputs.logits, labels, weight=weights)

            loss.backward()  # backward: fills p.grad for every parameter
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.max_grad_norm
            ).item()
            optimizer.step()  # apply the update
            scheduler.step()  # advance the learning-rate schedule
            optimizer.zero_grad(set_to_none=True)  # grads accumulate; clear before next batch

            running.append(loss.item())
            step += 1

            if step % args.log_every == 0:
                window = sum(running[-args.log_every :]) / len(running[-args.log_every :])
                print(
                    f"  epoch {epoch}  step {step:>4}/{total_steps}  "
                    f"loss {window:.4f}  grad_norm {grad_norm:5.2f}  "
                    f"lr {scheduler.get_last_lr()[0]:.2e}"
                )
                history.append({"step": step, "train_loss": window, "grad_norm": grad_norm})

            if args.smoke and step >= 20:
                break
        if args.smoke and step >= 20:
            break

        tune_loss, tune_acc, recalls = evaluate(model, tune_loader, device, weights)
        train_loss = sum(running) / len(running)
        print(
            f"\n  >> epoch {epoch}:  train {train_loss:.4f}   tune {tune_loss:.4f}   "
            f"acc {tune_acc:.1%}"
        )
        print(
            "     per-class recall: " + "  ".join(f"{k}={v:.1%}" for k, v in recalls.items()) + "\n"
        )
        history.append(
            {
                "step": step,
                "epoch": epoch,
                "train_loss": train_loss,
                "tune_loss": tune_loss,
                "tune_acc": tune_acc,
                **{f"recall_{k}": v for k, v in recalls.items()},
            }
        )

    elapsed = time.perf_counter() - t0
    print(f"  {step} steps in {elapsed:.1f}s ({elapsed / max(1, step):.2f}s/step)")

    if not args.smoke:
        out = OUT_DIR / "crossencoder"
        model.save_pretrained(out)
        tokenizer.save_pretrained(out)
        (OUT_DIR / "train_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"\n  saved -> {out}")
        print(f"  history -> {OUT_DIR / 'train_history.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
