"""Lesson 5 -- GPU, .to(device), memory and mixed precision.

Method: predict, then run. Each part asks for a prediction before showing the answer.

Needs a CUDA GPU. Part 4 deliberately runs one configuration that does not fit and spills into
system RAM, so it takes about a minute; everything else takes seconds.

    .venv/Scripts/python.exe lessons/05_gpu_and_precision.py
"""

from __future__ import annotations

import random
import sys
import time
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
SMALL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BIG = "pritamdeka/PubMedBERT-MNLI-MedNLI"
MB = 2**20
GB = 2**30


def pause(prompt: str) -> None:
    input(f"\n  >>> PREDICT: {prompt}\n  --- press Enter to reveal ---")


def header(title: str) -> None:
    print("\n" + "=" * 80 + f"\n{title}\n" + "=" * 80)


def load(name: str, device: torch.device) -> torch.nn.Module:
    return AutoModelForSequenceClassification.from_pretrained(
        name, num_labels=3, ignore_mismatched_sizes=True
    ).to(device)


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("This lesson needs a CUDA GPU.")
    gpu = torch.device("cuda")

    corpus = load_corpus(REPO_ROOT / "data" / "corpus.jsonl")
    fit, _ = split_train(load_claims(REPO_ROOT / "data" / "claims_train.jsonl"))
    pairs = build_pairs(fit, corpus, seed=20260919)
    random.Random(0).shuffle(pairs)  # noqa: S311 - fixed sample of 16 real pairs
    batch_pairs = pairs[:16]
    claims = [p.claim for p in batch_pairs]
    evidence = [p.evidence for p in batch_pairs]
    labels_cpu = torch.tensor([p.label_id for p in batch_pairs])

    # ======================================================================= PART 1
    header("PART 1 -- Where tensors live")
    tok = AutoTokenizer.from_pretrained(SMALL)
    model = load(SMALL, gpu)
    enc = tok(claims, evidence, truncation=True, padding=True, return_tensors="pt")
    print(f"  model weights are on: {next(model.parameters()).device}")
    print(f"  the tokenized batch is on: {enc['input_ids'].device}")

    pause("The model is on the GPU, the batch is still on the CPU. What happens on model(**enc)?")
    try:
        model(**enc)
    except RuntimeError as exc:
        print(f"  RuntimeError: {str(exc).splitlines()[0][:150]}")
    print(
        "\n  A GPU can only compute on numbers in its own memory. Nothing moves automatically:\n"
        "  every tensor has to be sent there explicitly with .to(device)."
    )

    pause("So we write  enc['input_ids'].to(gpu)  -- no assignment. Does model(**enc) work now?")
    enc["input_ids"].to(gpu)  # deliberately not assigned
    print(f"  enc['input_ids'] is on: {enc['input_ids'].device}   -- still the CPU")
    print(
        "\n  tensor.to(...) returns a NEW tensor and leaves the original where it was. The line\n"
        "  above created a GPU copy and threw it away. The correct form is\n"
        "      batch = {k: v.to(device) for k, v in batch.items()}\n\n"
        "  The trap: model.to(device) DOES work without assignment, because a module moves its\n"
        "  own parameters in place. Same method name, two different behaviours. Bug class:\n"
        "  'in-place vs returns-a-copy' -- the same shape as list.sort() vs sorted(list)."
    )
    del model
    torch.cuda.empty_cache()

    # ======================================================================= PART 2
    header(f"PART 2 -- Where GPU memory goes during one training step ({BIG.split('/')[-1]})")
    tok = AutoTokenizer.from_pretrained(BIG)
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    model = load(BIG, gpu)
    model.train()
    n = sum(p.numel() for p in model.parameters())
    weights = torch.cuda.memory_allocated() - base
    print(f"  {n / 1e6:.1f}M parameters x 4 bytes (fp32) = {n * 4 / MB:.0f} MB")
    print(f"  after loading onto the GPU:  {weights / MB:6.0f} MB")

    pause(
        "After ONE full training step (forward, backward, AdamW update), how many times the "
        "weight memory is in use?"
    )
    batch = {
        k: v.to(gpu)
        for k, v in tok(
            claims, evidence, truncation=True, padding=True, return_tensors="pt"
        ).items()
    }
    y = labels_cpu.to(gpu)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    loss = torch.nn.functional.cross_entropy(model(**batch).logits, y)
    after_fwd = torch.cuda.memory_allocated() - base
    loss.backward()
    after_bwd = torch.cuda.memory_allocated() - base
    opt.step()
    after_step = torch.cuda.memory_allocated() - base
    peak = torch.cuda.max_memory_allocated() - base
    rows = [
        ("loaded", weights, "weights"),
        ("after forward", after_fwd, "weights + ACTIVATIONS kept for the backward pass"),
        ("after backward", after_bwd, "activations freed, GRADIENTS added (one per weight)"),
        ("after optimizer step", after_step, "AdamW's two running averages per weight added"),
    ]
    for name, val, what in rows:
        print(f"  {name:<22} {val / MB:6.0f} MB  = {val / weights:4.1f}x weights   {what}")
    print(f"  {'peak during the step':<22} {peak / MB:6.0f} MB")
    print(
        "\n  Steady state is 4x the weights: weights + gradients + two Adam averages, 4 bytes\n"
        "  each = the '16 bytes per parameter' rule from STATE.md, now measured, not assumed.\n"
        "  The PEAK is higher: during backward, activations and gradients exist at the same time.\n"
        "  Activations are the only part that grows with batch size and sequence length -- so\n"
        "  they are the only part padding and precision can shrink."
    )
    del model, opt, loss, batch
    torch.cuda.empty_cache()

    # ======================================================================= PART 3
    header("PART 3 -- Number formats: what 16-bit gives up")
    print("  fp32: 8 exponent bits, 23 mantissa bits   (the default)")
    print("  fp16: 5 exponent bits, 10 mantissa bits")
    print("  bf16: 8 exponent bits,  7 mantissa bits   (same RANGE as fp32, less PRECISION)")
    pause("A gradient of 1e-8 is stored in fp16 and in bf16. Which one turns it into zero?")
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        small = torch.tensor(1e-8, dtype=dt).item()
        near_one = torch.tensor(1.001, dtype=dt).item()
        print(f"  {dt!s:<16} 1e-8 -> {small:<12.3e} 1.001 -> {near_one:.6f}")
    print(
        "\n  fp16 cannot represent 1e-8 at all: it becomes 0. Small gradients vanish, so fp16\n"
        "  training needs a GradScaler that multiplies the loss up before backward and divides\n"
        "  back down afterwards. bf16 keeps fp32's range, so 1e-8 survives -- but it rounds 1.001\n"
        "  to 1.0: fewer digits of precision. For training, range matters more than precision,\n"
        f"  and this GPU supports bf16 (torch.cuda.is_bf16_supported() = "
        f"{torch.cuda.is_bf16_supported()}), so bf16 is the choice and no scaler is needed."
    )

    # ======================================================================= PART 4
    header(f"PART 4 -- Does {BIG.split('/')[-1]} train on this 4 GB GPU?")
    print(
        "  torch.autocast runs the heavy matrix multiplies in bf16. Weights, gradients and Adam\n"
        "  state stay fp32, so only ACTIVATIONS shrink -- and matmuls get faster on tensor cores."
    )
    pause(
        "Which of the four configurations below spills into system RAM, and how much slower is it?"
    )
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()  # what the DRIVER says is free, not what PyTorch sees
    print(f"  GPU memory: {total / GB:.2f} GiB total, {free / GB:.2f} GiB actually free right now")
    print("  (the rest is held by Windows, other apps, and this process's own CUDA context)\n")
    print(f"  {'padding':<22} {'precision':<10} {'peak memory':>12} {'time / step':>13}")
    for label, pad in (("fixed to 256", "max_length"), ("per batch (Lesson 2)", True)):
        for amp in (False, True):
            model = load(BIG, gpu)
            model.train()
            opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
            enc = tok(
                claims, evidence, truncation=True, max_length=256, padding=pad, return_tensors="pt"
            )
            batch = {k: v.to(gpu) for k, v in enc.items()}
            try:
                for step in range(4):
                    if step == 1:
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        t0 = time.perf_counter()
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                        loss = torch.nn.functional.cross_entropy(model(**batch).logits, y)
                    loss.backward()
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t0) / 3 * 1000
                pk = torch.cuda.max_memory_allocated() / GB
                print(
                    f"  {label:<22} {'bf16' if amp else 'fp32':<10} {pk:>9.2f} GiB {ms:>10.0f} ms"
                )
            except torch.cuda.OutOfMemoryError:
                print(f"  {label:<22} {'bf16' if amp else 'fp32':<10} {'OUT OF MEMORY':>12}")
            del model, opt, batch
            torch.cuda.empty_cache()

    print(
        "\n  The fp32 / 256 row reports LESS than 4 GiB yet runs roughly 20-40x slower (the exact\n"
        "  factor varies run to run). Its peak is above the 'actually free' figure printed at the\n"
        "  top of this part: Windows and other apps already hold part of the card, and PyTorch's\n"
        "  peak-memory number does not count them. Past that line the Windows driver spills into\n"
        "  system RAM instead of raising an error. Nothing crashes; everything just gets slower.\n"
        "  The real budget is 'free', not 'total'.\n\n"
        "  bf16 at the worst-case length fits. Per-batch padding plus bf16 is the fastest and the\n"
        "  smallest. That settles ADR-0002's open question: the better-matched model CAN be\n"
        "  trained here, provided both lessons' fixes are on."
    )

    header("END OF LESSON 5 -- one question, answer it in chat")
    print(
        "  In Part 4, bf16 saved much more memory at length 256 than with per-batch padding.\n"
        "  Using Part 2's breakdown, explain why -- which part of memory does autocast shrink,\n"
        "  and why is that part small when the padding is small?"
    )


if __name__ == "__main__":
    main()
