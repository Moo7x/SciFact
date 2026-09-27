# Lesson 1 — Tokenization: what the model actually receives

*2026-09-27. Run `lessons/01_tokenization.py`. Tokenizer: `cross-encoder/ms-marco-MiniLM-L-6-v2`,
the exact one used in training.*

The pair (train, SUPPORT):

> **Claim:** A high microerythrocyte count protects against severe anemia in homozygous
> alpha (+)- thalassemia trait subjects.
> **Evidence:** CONCLUSIONS The increased erythrocyte count and microcytosis in children
> homozygous for alpha(+)-thalassaemia may contribute substantially to their protection
> against SMA.

Mounir's predictions were made before running and are kept here, because the gap between the
prediction and the answer is the lesson.

---

## Q1 — 15 words became how many tokens?

**Predicted:** 19. Reasoning: punctuation becomes separate tokens, complex words split.
**Actual:** 30.

Right on both mechanisms; wrong only on how much splitting happens.

```
microerythrocyte   6  ['micro', '##ery', '##th', '##ro', '##cy', '##te']
homozygous         4  ['homo', '##zy', '##go', '##us']
(+)-               4  ['(', '+', ')', '-']
thalassemia        3  ['tha', '##lass', '##emia']
anemia             2  ['an', '##emia']
subjects.          2  ['subjects', '.']
nine other words   1  each
                  --
                  30
```

A second thing nobody predicted: **"A" became "a" and "CONCLUSIONS" became "conclusions".** This
tokenizer is *uncased* — it lowercases everything before it starts. Capital letters never reach
the model at all.

## Q2 — how does "microerythrocyte" split?

**Predicted:** `micro` + `erythrocyte`.
**Actual:** `micro ##ery ##th ##ro ##cy ##te` — six pieces.

The prediction splits the word the way a person does: on meaning (morphemes). The tokenizer knows
nothing about meaning. It splits on **frequency**.

**How it works (WordPiece).** The vocabulary is ~30,000 pieces, chosen before training by
repeatedly merging the character pairs that appear most often in the pretraining text. To
tokenize a word, it takes the **longest prefix that is in the vocabulary**, then the longest
continuation (`##` = "glued to the previous piece"), and so on until the word is used up.

`micro` is common, so it is one piece. `erythrocyte` is rare in the text this model was trained
on, so it breaks into fragments that are common for unrelated reasons (`##th`, `##ro`). None of
those fragments means anything on its own. The model has to rebuild "red blood cell" out of
them, and it does that worse than it would with a single token.

**Why this matters here.** This model was pretrained on MS MARCO — web search queries, not
biomedical papers. Its vocabulary reflects that. A tokenizer built from biomedical text would
very likely keep `erythrocyte` whole. That is a hypothesis about *our* data, logged as OQ-012,
and it is testable rather than assumed.

## Q3 — how does the model know where the claim ends?

**Predicted:** from the content, e.g. the full stop or the word "CONCLUSIONS".
**Actual:** it is told, explicitly, with special tokens.

```
position  token        segment
       0  [CLS]        0
       1  a            0
     ...
      31  [SEP]        0      <- end of claim
     ...               1      <- evidence tokens, segment 1
      71  [SEP]        1      <- end of evidence
```

`[CLS]` opens the sequence. `[SEP]` closes each segment. `token_type_ids` labels every token 0
(claim) or 1 (evidence).

**Why not your full-stop rule?** It is a reasonable guess, and it is the kind of rule that breaks
on real data. Structure should never be *inferred from* content that can contain the same
symbol. The general principle is **out-of-band signalling**: to mark a boundary, use a symbol the
content can never produce. `[SEP]` is a reserved vocabulary entry, so no text can ever
accidentally generate it.

**`[CLS]` matters more than it looks.** Its final hidden vector is the only thing the classifier
head reads. The model is trained to pack its judgement about the whole pair into that one
position. Lesson 3 follows it through the network.

## Q4 — how much of the 256 is padding?

**Predicted:** "the number of tokens, then the rest is padding." Right structure, no fraction.
**Actual:** 72 real tokens, **184 padding — 72% of the row.**

`attention_mask` marks each position 1 (real) or 0 (padding). Masked positions do not affect the
result. **They still cost compute**, though: the GPU multiplies every padded position anyway.
The linear layers do about 256/72 ≈ 3.6× the necessary work. Attention compares every position
with every other, so its cost grows with the *square* of the length.

Lesson 2 starts from this number.

## Q5 — do thalassemia and thalassaemia match?

**Predicted:** "probably not matching tokens but close."
**Actual:** exactly that.

```
thalassemia   -> tha ##lass ##emia
thalassaemia  -> tha ##lass ##ae ##mia
```

Two of the pieces are shared, then they diverge. BM25 compares whole words, so for it this is a
total miss. Word-pieces partly survive a spelling change.

**The more important thing this pair shows.** The claim's key idea is *"severe anemia"*. The
evidence expresses it as **SMA** — severe malarial anaemia:

```
anemia -> ['an', '##emia']
SMA    -> ['sm', '##a']
```

**Zero shared tokens.** The one piece of evidence that links "protection against SMA" to
"protects against severe anemia" is invisible at the token level. The model can only connect
them if it learned during pretraining that SMA means severe anaemia. A web-search model probably
did not.

And this pair is labelled **SUPPORT** — the class that collapsed in Stage 3. This does not prove
that acronyms and paraphrase *caused* the collapse. It does show a concrete mechanism by which
supporting evidence can look unrelated to the claim it supports, while a contradicting claim
(written by negating the original) keeps nearly all of its tokens.

**Token overlap is not meaning overlap, in either direction.** `micro` appears in both texts,
but it comes from *microerythrocyte* in one and *microcytosis* in the other — a false match.
`SMA` and `anemia` mean the same thing and share nothing — a false miss.
