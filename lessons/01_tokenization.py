"""Lesson 1 -- Tokenization: what the model actually receives.

Run AFTER you have written down your predictions in chat. Each section pauses for Enter so you
can compare one prediction at a time instead of seeing everything at once.

    .venv/Scripts/python.exe lessons/01_tokenization.py

Uses the exact tokenizer and settings from scripts/train_crossencoder.py.
"""

from __future__ import annotations

from transformers import AutoTokenizer

MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # same as the training script
MAX_LENGTH = 256  # same as the training script

CLAIM = (
    "A high microerythrocyte count protects against severe anemia in homozygous "
    "alpha (+)- thalassemia trait subjects."
)
EVIDENCE = (
    "CONCLUSIONS The increased erythrocyte count and microcytosis in children homozygous for "
    "alpha(+)-thalassaemia may contribute substantially to their protection against SMA."
)


def pause(label: str) -> None:
    input(f"\n--- press Enter to reveal: {label} ---")


def main() -> None:
    tok = AutoTokenizer.from_pretrained(MODEL)

    print("=" * 76)
    print("CLAIM:   ", CLAIM)
    print("EVIDENCE:", EVIDENCE)
    print("=" * 76)

    # ---------------------------------------------------------------- Q1
    pause("Q1  words vs tokens in the claim")
    words = CLAIM.split()
    pieces = tok.tokenize(CLAIM)
    print(f"  whitespace words: {len(words)}")
    print(f"  tokens:           {len(pieces)}")
    print(f"  tokens: {pieces}")

    # ---------------------------------------------------------------- Q2
    pause("Q2  how rare biomedical words are split")
    for word in ("microerythrocyte", "thalassemia", "thalassaemia", "anemia", "count"):
        print(f"  {word:<18} -> {tok.tokenize(word)}")
    print(
        "\n  '##' means 'glued to the previous piece'. The vocabulary has ~30,000 entries; any\n"
        "  word not in it is built from smaller pieces that are. Nothing is ever 'unknown'."
    )

    # ---------------------------------------------------------------- Q3
    pause("Q3  how the model knows where the claim ends and the evidence begins")
    enc = tok(CLAIM, EVIDENCE)
    ids = enc["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    types = enc["token_type_ids"]
    print("  position  token                 segment   id")
    for i, (t, s, x) in enumerate(zip(toks, types, ids, strict=True)):
        if i < 4 or t in ("[CLS]", "[SEP]") or i >= len(toks) - 3:
            print(f"  {i:>8}  {t:<20}  {s:>7}   {x}")
        elif i == 4:
            print("       ...")
    print(
        "\n  [CLS] opens the sequence. Its final hidden vector is what the classifier head reads\n"
        "  -- the model is trained to pack the whole pair's meaning into that one position.\n"
        "  [SEP] closes each segment. token_type_ids says which segment each token is in."
    )

    # ---------------------------------------------------------------- Q4
    pause("Q4  real tokens vs padding at max_length=256")
    padded = tok(CLAIM, EVIDENCE, truncation=True, max_length=MAX_LENGTH, padding="max_length")
    mask = padded["attention_mask"]
    real = sum(mask)
    print(f"  sequence length:  {len(padded['input_ids'])}")
    print(f"  real tokens:      {real}")
    print(
        f"  padding tokens:   {len(mask) - real}   "
        f"({(len(mask) - real) / len(mask):.0%} of the row)"
    )
    print(f"  attention_mask:   {mask[: real + 3]} ... (all zeros to the end)")
    print(
        "\n  attention_mask=0 tells the model 'ignore this position'. The padding still costs\n"
        "  compute, though: the GPU multiplies every one of those zeros anyway."
    )

    # ---------------------------------------------------------------- Q5
    pause("Q5  the spelling mismatch")
    c = set(tok.tokenize(CLAIM))
    e = set(tok.tokenize(EVIDENCE))
    print(f"  claim pieces also in evidence: {sorted(c & e)}")
    print(f"  claim pieces NOT in evidence:  {sorted(c - e)}")
    print(
        "\n  BM25 compares whole words, so 'thalassemia' vs 'thalassaemia' is a total miss.\n"
        "  Word-pieces partly survive the spelling change. Whether the MODEL treats them as the\n"
        "  same thing is a separate question -- tokens only decide what it can see, not what\n"
        "  it understands."
    )


if __name__ == "__main__":
    main()
