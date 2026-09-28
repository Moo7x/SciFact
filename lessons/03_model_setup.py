"""Lesson 3 -- Model setup: what is inside the model, and one pair traced through it.

Method: traced worked example. The Lesson 1 pair goes in, and every stage prints its tensor
shape. Before each reveal the script asks you to predict the shape from what you already know.

Runs on CPU on purpose: inference on one pair is instant, and it spares a laptop on battery.

    .venv/Scripts/python.exe lessons/03_model_setup.py
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()
torch.set_grad_enabled(False)  # no training in this lesson: nothing needs gradients

OURS = "cross-encoder/ms-marco-MiniLM-L-6-v2"
NLI = "pritamdeka/PubMedBERT-MNLI-MedNLI"

CLAIM = (
    "A high microerythrocyte count protects against severe anemia in homozygous "
    "alpha (+)- thalassemia trait subjects."
)
EVIDENCE = (
    "CONCLUSIONS The increased erythrocyte count and microcytosis in children homozygous for "
    "alpha(+)-thalassaemia may contribute substantially to their protection against SMA."
)


def pause(prompt: str) -> None:
    input(f"\n  >>> PREDICT: {prompt}\n  --- press Enter to reveal ---")


def header(title: str) -> None:
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)


def params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> None:
    tok = AutoTokenizer.from_pretrained(OURS)
    model = AutoModelForSequenceClassification.from_pretrained(OURS).eval()

    # ======================================================================= PART 1
    header("PART 1 -- What is inside the model we trained")
    total = params(model)
    blocks = [
        ("embeddings (one vector per vocab entry)", model.bert.embeddings),
        *[(f"encoder layer {i}", layer) for i, layer in enumerate(model.bert.encoder.layer)],
        ("pooler", model.bert.pooler),
        ("classifier (the head)", model.classifier),
    ]
    pause("The model has 22.7M parameters. Which block holds the most of them?")
    for name, block in blocks:
        n = params(block)
        print(f"  {name:<42} {n:>11,}  {n / total:6.1%}  {'#' * round(40 * n / total)}")
    print(f"  {'total':<42} {total:>11,}")
    emb = model.bert.embeddings.word_embeddings.weight
    print(
        f"\n  The embedding table alone is {tuple(emb.shape)}: 30,522 vocabulary entries x 384\n"
        "  numbers each. That is Lesson 1's 'one vector per token', made literal: every\n"
        "  word-piece, including '##th' and '##ro', owns one row of this table, and it is\n"
        "  half the model. All six layers of actual computation share the other half."
    )

    # ======================================================================= PART 2
    header("PART 2 -- Trace the Lesson 1 pair through the network")
    enc = tok(CLAIM, EVIDENCE, return_tensors="pt")
    n_tok = enc["input_ids"].shape[1]
    print(f"  input_ids shape: {tuple(enc['input_ids'].shape)}   (batch of 1, {n_tok} tokens)")

    pause("Embeddings turn each token id into a vector. Shape after the embedding layer?")
    out = model.bert(**enc, output_hidden_states=True)
    states = out.hidden_states  # embeddings output, then one entry per encoder layer
    print(f"  after embeddings:  {tuple(states[0].shape)}   = (batch, tokens, 384)")

    pause("Each of the 6 layers mixes information between tokens. Shape after layer 6?")
    for i, s in enumerate(states[1:], start=1):
        print(f"  after layer {i}:     {tuple(s.shape)}")
    print(
        "\n  The shape never changes. Every layer takes 72 vectors and returns 72 vectors.\n"
        "  What changes is what they MEAN: by layer 6, each token's vector has absorbed\n"
        "  information from every other token through attention. That is what 'cross'\n"
        "  in cross-encoder means -- claim tokens and evidence tokens read each other."
    )

    pause("The head needs ONE vector for the whole pair, not 72. Which one does it take?")
    cls = states[-1][:, 0]  # position 0 is [CLS]
    print(f"  states[-1][:, 0] -> {tuple(cls.shape)}   position 0 = the [CLS] token")
    print(
        "  The other 71 vectors are discarded. This is why Lesson 1 said [CLS] matters:\n"
        "  training has to teach the network to pack its whole judgement into position 0."
    )

    # ======================================================================= PART 3
    header("PART 3 -- Reproduce the model's output by hand, from the [CLS] vector")
    pooled = torch.tanh(model.bert.pooler.dense(cls))
    by_hand = model.classifier(pooled)
    official = model(**enc).logits
    print(f"  pooler:      tanh(dense(cls))       -> {tuple(pooled.shape)}")
    print(f"  classifier:  linear(pooled)         -> {tuple(by_hand.shape)}")
    print(f"\n  computed by hand:  {by_hand.item():+.5f}")
    print(f"  model(**enc):      {official.item():+.5f}   (identical: nothing hidden)")

    pause(
        f"The head's weight matrix -- what shape is it, given it outputs {by_hand.shape[1]} number?"
    )
    w = model.classifier.weight
    print(
        f"  classifier.weight: {tuple(w.shape)}   classifier.bias: {tuple(model.classifier.bias.shape)}"
    )
    print(
        "\n  ONE output. This model was trained to answer a single question -- 'how relevant\n"
        "  is this passage to this search query?' -- as one number. It has no concept of\n"
        "  SUPPORT versus CONTRADICT. A passage can be highly relevant and still contradict."
    )

    # ======================================================================= PART 4
    header("PART 4 -- What happened to that head when we trained it (the MISMATCH warning)")
    swapped = AutoModelForSequenceClassification.from_pretrained(
        OURS, num_labels=3, ignore_mismatched_sizes=True
    ).eval()
    print(f"  original head: {tuple(model.classifier.weight.shape)}")
    print(f"  after num_labels=3: {tuple(swapped.classifier.weight.shape)}")
    same_body = torch.equal(
        model.bert.encoder.layer[5].output.dense.weight,
        swapped.bert.encoder.layer[5].output.dense.weight,
    )
    print(f"  encoder weights identical to the original? {same_body}")
    probs = torch.softmax(swapped(**enc).logits, dim=-1)[0]
    print(f"  untrained 3-way head on our pair: {[f'{p:.3f}' for p in probs.tolist()]}")

    pause(
        "Why are those three numbers all close to 1/3 -- and why did Stage 3 print "
        "'before training: acc 8.3%'?"
    )
    print(
        "  ignore_mismatched_sizes=True kept all 6 layers and THREW AWAY the 1-output head,\n"
        "  replacing it with a fresh (3, 384) matrix of small random numbers. Random weights\n"
        "  give near-equal logits, so softmax gives near-equal probabilities -- about 1/3 each.\n"
        "  That is the 'MISMATCH ... Reinit' warning you saw in every training run.\n\n"
        "  But 8.3% is not 'about chance' -- chance is 33%. The prediction is the ARGMAX, and\n"
        "  the argmax of three near-equal numbers is decided by tiny random biases in the new\n"
        "  weights. Those biases are the same for most inputs, so an untrained head tends to\n"
        "  predict the same class almost everywhere. If that class happens to be rare in the\n"
        "  evaluation set, accuracy lands far BELOW chance. Near-uniform probabilities and a\n"
        "  near-constant prediction are the same fact seen two ways.\n\n"
        "  So Stage 3 kept a body trained for RELEVANCE and grew a new head for ENTAILMENT\n"
        "  from 2,196 examples. ADR-0002 is about exactly that mismatch."
    )

    # ======================================================================= PART 5
    header("PART 5 -- The better-matched candidate, same pair, no training")
    nli_tok = AutoTokenizer.from_pretrained(NLI)
    nli = AutoModelForSequenceClassification.from_pretrained(NLI).eval()
    labels = [nli.config.id2label[i] for i in range(nli.config.num_labels)]
    print(f"  head: {tuple(nli.classifier.weight.shape)}   labels: {labels}")
    print(f"  size: {params(nli):,} parameters, hidden width 768 (ours: 384)")

    pause("NLI reads (premise, hypothesis). Which of our two texts is the premise?")
    nli_enc = nli_tok(EVIDENCE, CLAIM, return_tensors="pt")  # premise = evidence
    nli_probs = torch.softmax(nli(**nli_enc).logits, dim=-1)[0]
    for lab, p in zip(labels, nli_probs.tolist(), strict=True):
        print(f"  {lab:<14} {p:6.3f}  {'#' * round(40 * p)}")
    print(
        "\n  The premise is what you take as given -- the evidence. The hypothesis is what you\n"
        "  test against it -- the claim. Swap them and you ask a different question: whether\n"
        "  the claim implies the evidence. Stage 3 fed (claim, evidence); for a model trained\n"
        "  from scratch that order is arbitrary, for an NLI model it is not.\n\n"
        "  The gold label for this pair is SUPPORT = entailment. Remember it links 'severe\n"
        "  anemia' to 'SMA' with zero shared tokens. Compare with what our model could say:\n"
        "  one relevance number."
    )

    header("END OF LESSON 3 -- one question, answer it in chat")
    print(
        "  Our model is 22.7M parameters and half of them are the embedding table.\n"
        "  If you swapped in a tokenizer with a 128,000-word vocabulary (DeBERTa's) but kept\n"
        "  hidden size 384, roughly how many parameters would the embedding table alone be --\n"
        "  and what does that tell you about why nli-deberta-v3-xsmall is '71M' in the\n"
        "  ADR-0002 table despite being architecturally small?"
    )


if __name__ == "__main__":
    main()
