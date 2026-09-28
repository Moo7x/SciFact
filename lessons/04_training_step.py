"""Lesson 4 -- The training step: what each line does, and how each one fails.

Method: annotated snippet, then a bug hunt. Four "mystery" runs each have one line of the
training step secretly broken. You see only their symptoms and match each to its bug -- which is
how debugging actually works: you never see the bug, only what it does.

Every run uses the fastest sanity check for training code: can the model memorise ONE fixed
batch of 16 pairs? A working loop drives loss towards zero within a few dozen steps. If it can't
do that, nothing about the full run can be trusted.

    .venv/Scripts/python.exe lessons/04_training_step.py

Uses the GPU if one is available (seconds), otherwise CPU (about 3 minutes).
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from transformers.utils import logging as hf_logging  # noqa: E402

from scifact.data.schema import load_claims, load_corpus  # noqa: E402
from scifact.data.splits import split_train  # noqa: E402
from scifact.verify.dataset import build_pairs  # noqa: E402

hf_logging.set_verbosity_error()
MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
STEPS = 40
SHOW = (0, 5, 10, 20, 30, 39)

SNIPPET = """
    for batch in train_loader:
        labels  = batch.pop("labels")                     # 1
        logits  = model(**batch).logits                   # 2  forward
        loss    = cross_entropy(logits, labels)           # 3  how wrong, one number
        loss.backward()                                   # 4  fills p.grad for every weight
        clip_grad_norm_(model.parameters(), 1.0)          # 5  cap the step size
        optimizer.step()                                  # 6  move every weight downhill
        scheduler.step()                                  # 7  advance the learning rate
        optimizer.zero_grad()                             # 8  clear p.grad for next batch
"""

WHY = [
    (
        "1",
        "labels out of the batch",
        "The model must not see the answers as input. They go to the loss, not the model.",
    ),
    (
        "2",
        "forward",
        "Token ids -> 3 logits per pair. Also records every operation, so line 4 can go backwards.",
    ),
    ("3", "loss", "Collapses 16 predictions into ONE number: the mean of -log p(correct class)."),
    (
        "4",
        "backward",
        "Chain rule from that one number back through every operation. Computes d(loss)/d(w) for\n"
        "        all 22.7M weights and stores it in w.grad. Changes no weight.",
    ),
    (
        "5",
        "clip",
        "If the combined gradient is longer than 1.0, shrink it to 1.0. One bad batch cannot\n"
        "        throw the weights somewhere wild.",
    ),
    (
        "6",
        "step",
        "The ONLY line that changes the model. Reads w.grad and moves each weight against it.",
    ),
    (
        "7",
        "schedule",
        "Warmup, then decay: small steps while Adam has no history, smaller steps late on.",
    ),
    (
        "8",
        "zero_grad",
        "PyTorch ADDS new gradients to whatever is already in w.grad. Without this, every batch's\n"
        "        gradient is piled onto all the previous ones.",
    ),
]

BUGS = {
    "no_zero_grad": "line 8 deleted  (optimizer.zero_grad)",
    "no_step": "line 6 deleted  (optimizer.step)",
    "no_backward": "line 4 deleted  (loss.backward)",
    "lr_high": "learning rate set 50x too high (5e-3 instead of 1e-4)",
}

EXPLAIN = {
    "no_zero_grad": (
        "Loss falls, then STALLS well above zero. Gradient size climbs without limit (2 -> 100+).\n"
        "  Each step uses the running SUM of every gradient so far -- including gradients\n"
        "  computed at weights the model has since moved away from, which now point the wrong\n"
        "  way. The stale ones outvote the current one. Adam partly hides this: it divides by\n"
        "  gradient size,\n"
        "  so the steps don't explode -- which is why the LOSS looks almost fine. The gradient\n"
        "  size is the giveaway. RECOGNISE IT: grad norm rising steadily while loss plateaus."
    ),
    "no_step": (
        "Loss never moves -- it only wobbles. The wobble is dropout, which is still switched on\n"
        "  in training mode and gives slightly different outputs each call. Gradients ARE\n"
        "  computed (grad norm ~2) but never applied, so weight change is exactly zero.\n"
        "  RECOGNISE IT: healthy gradients, zero weight change."
    ),
    "no_backward": (
        "The loss curve is IDENTICAL to the step-less run -- same numbers, digit for digit.\n"
        "  From the loss alone these two bugs cannot be told apart. The gradient norm can:\n"
        "  here it is exactly 0, because nothing ever computed a gradient. optimizer.step()\n"
        "  runs, but with nothing to read it moves nothing.\n"
        "  RECOGNISE IT: grad norm exactly zero."
    ),
    "lr_high": (
        "Loss goes UP at first -- above where it started -- then hovers near ln(3) = 1.099,\n"
        "  the random-guess value. The steps are so large they overshoot every good region and\n"
        "  knock the pretrained weights off their useful values. Weight change is huge compared\n"
        "  with the correct run, yet the model learns nothing.\n"
        "  RECOGNISE IT: loss rises above its start, big weight change, no progress."
    ),
}


def pause(prompt: str) -> None:
    input(f"\n  >>> {prompt}\n  --- press Enter to continue ---")


def header(title: str) -> None:
    print("\n" + "=" * 86 + f"\n{title}\n" + "=" * 86)


def grad_norm(model: torch.nn.Module) -> float:
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    return math.sqrt(sum(float(g.norm()) ** 2 for g in grads)) if grads else 0.0


def weight_change(model: torch.nn.Module, start: list[torch.Tensor]) -> float:
    return math.sqrt(
        sum(
            float((p.detach() - s).norm()) ** 2
            for p, s in zip(model.parameters(), start, strict=True)
        )
    )


def run(variant: str, enc: dict, labels: torch.Tensor) -> dict:
    torch.manual_seed(0)  # same starting head and same dropout draws in every run
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=3, ignore_mismatched_sizes=True
    ).to(DEVICE)
    model.train()
    start = [p.detach().clone() for p in model.parameters()]
    lr = 5e-3 if variant == "lr_high" else 1e-4
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    loss_t, grad_t, move_t = [], [], []
    for _ in range(STEPS):
        loss = torch.nn.functional.cross_entropy(model(**enc).logits, labels)
        if variant != "no_backward":
            loss.backward()
        grad_t.append(grad_norm(model))
        if variant != "no_step":
            opt.step()
        if variant != "no_zero_grad":
            opt.zero_grad(set_to_none=True)
        loss_t.append(loss.item())
        move_t.append(weight_change(model, start))

    with torch.no_grad():
        model.eval()
        preds = model(**enc).logits.argmax(-1).tolist()
    del model, opt, start
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"loss": loss_t, "grad": grad_t, "move": move_t, "preds": preds}


def show(name: str, r: dict, gold: list[int]) -> None:
    correct = sum(p == g for p, g in zip(r["preds"], gold, strict=True))
    print(f"\n  RUN {name}")
    print("    step          " + "".join(f"{s:>9}" for s in SHOW))
    print("    loss          " + "".join(f"{r['loss'][s]:>9.3f}" for s in SHOW))
    print("    grad size     " + "".join(f"{r['grad'][s]:>9.2f}" for s in SHOW))
    print("    weight moved  " + "".join(f"{r['move'][s]:>9.3f}" for s in SHOW))
    print(f"    after training, memorised {correct}/16 of the batch")


def main() -> None:
    corpus = load_corpus(REPO_ROOT / "data" / "corpus.jsonl")
    fit, _ = split_train(load_claims(REPO_ROOT / "data" / "claims_train.jsonl"))
    pairs = build_pairs(fit, corpus, seed=20260919)
    by_label: dict[int, list] = {0: [], 1: [], 2: []}
    for p in pairs:
        by_label[p.label_id].append(p)
    batch = by_label[0][:6] + by_label[1][:5] + by_label[2][:5]

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(
        [p.claim for p in batch],
        [p.evidence for p in batch],
        truncation=True,
        max_length=256,
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)
    labels = torch.tensor([p.label_id for p in batch], device=DEVICE)
    gold = labels.tolist()

    # ======================================================================= PART 1
    header("PART 1 -- The training step, one line at a time")
    print(SNIPPET)
    for num, name, why in WHY:
        print(f"  {num}  {name:<24} {why}")
    print(
        "\n  Only line 6 changes the model. Lines 2-4 compute what to change; 5, 7 and 8 keep\n"
        "  the change safe and the next step clean."
    )

    # ======================================================================= PART 2
    header(f"PART 2 -- Sanity check: can a correct loop memorise one batch of 16? ({DEVICE.type})")
    print(
        "  Three signals, not one:\n"
        "    loss          how wrong the model is                   (ln 3 = 1.099 is guessing)\n"
        "    grad size     how big the computed gradient is        (0 = nothing computed)\n"
        "    weight moved  distance the weights have travelled from where they started"
    )
    good = run("correct", enc, labels)
    show("CORRECT (nothing broken)", good, gold)
    print(
        "\n  Loss falls towards zero. Gradient size rises, then falls as there is less left to\n"
        "  fix. Weights move steadily. That shape is the reference for everything below."
    )

    # ======================================================================= PART 3
    header("PART 3 -- Four mystery runs. Each has ONE line broken.")
    print("  The possible bugs:")
    for i, desc in enumerate(BUGS.values(), start=1):
        print(f"    ({i}) {desc}")
    variants = list(BUGS)
    order = variants[:]
    random.Random(4).shuffle(order)  # noqa: S311 - fixed shuffle so run letters are stable
    names = "ABCD"
    results = {v: run(v, enc, labels) for v in order}
    for name, v in zip(names, order, strict=True):
        show(name, results[v], gold)

    pause(
        "Match runs A-D to bugs (1)-(4). Compare each with the CORRECT run in Part 2. "
        "Two runs have identical loss -- which signal separates them?"
    )

    for name, v in zip(names, order, strict=True):
        print(f"\n  RUN {name} = {BUGS[v]}")
        print(f"  {EXPLAIN[v]}")

    header("THE RULE THIS LESSON EXISTS FOR")
    print(
        "  Two different bugs produced the same loss curve to three decimal places. Anyone who\n"
        "  logs only the loss cannot tell them apart. Log at least three things: the loss (is it\n"
        "  learning?), the gradient size (is anything being computed?), and some measure of\n"
        "  change (is anything being applied?). Each bug above shows up in a different one."
    )

    # ======================================================================= PART 4
    header("END OF LESSON 4 -- one question, answer it in chat")
    hist = REPO_ROOT / "outputs" / "train_history_baseline.json"
    if hist.exists():
        rows = [r for r in json.loads(hist.read_text(encoding="utf-8")) if "grad_norm" in r]
        pick = rows[:: max(1, len(rows) // 8)][:8]
        print("  Your real Stage 3 baseline run logged this:")
        print("    step       " + "".join(f"{r['step']:>8}" for r in pick))
        print("    loss       " + "".join(f"{r['train_loss']:>8.3f}" for r in pick))
        print("    grad size  " + "".join(f"{r['grad_norm']:>8.2f}" for r in pick))
    print(
        "\n  1. Using the signatures above, which of the four bugs can you RULE OUT from\n"
        "     that log?\n"
        "  2. The Stage 3 script logged loss and gradient size but not weight change. Is there\n"
        "     any bug from this lesson that its log could not have caught? Why or why not?"
    )


if __name__ == "__main__":
    main()
