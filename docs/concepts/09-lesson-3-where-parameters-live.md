# Lesson 3 — Where a model's parameters live, and why "71M" can be small

*2026-09-29. Counted directly from the two models, not estimated.*

## The embedding table is a spreadsheet

One row per vocabulary entry, one column per hidden dimension:

```
rows    = vocabulary size   (how many different tokens exist)
columns = hidden size       (384 numbers describe each token)
parameters = rows x columns
```

When a token comes in, the model looks up its row. That is all the table does.

## The arithmetic

| model | table | = | parameters |
|---|---|---|---|
| ms-marco-MiniLM (ours) | 30,522 x 384 | = | 11.7M |
| nli-deberta-v3-xsmall | 128,100 x 384 | = | **49.2M** |

Same row width (384). Four times the rows, so four times the table.

## Where each model's parameters go

| | word table | layers | total |
|---|---|---|---|
| ms-marco-MiniLM | 11.7M (52%) | 6 x 1,774,464 = 10.6M | 22.7M |
| nli-deberta-v3-xsmall | **49.2M (69%)** | 12 x 1,774,464 = 21.3M | 70.8M |

**Each layer is exactly the same size in both: 1,774,464 parameters.** DeBERTa-xsmall has twice
as many layers, and a table four times bigger. The "71M" is mostly the table.

## Why this matters: two kinds of parameters

**Lookup parameters (the table).** Each token reads one row. A 128,000-row table costs the same
per token as a 30,000-row table: you still fetch exactly one row. The table's size does not make
the model slower.

**Compute parameters (the layers).** Every token passes through every layer, so these set how
much work the model does per token.

Measured against the "3x bigger" the headline numbers suggest:

- **Speed:** DeBERTa-xsmall does about **2x** the per-token work of our model (12 layers instead
  of 6, same width). Its 71M does not mean 3x slower.
- **Training memory:** here the table counts in full. Every parameter needs a gradient and two
  Adam averages, 16 bytes in total, and the table is no exception. 49.2M x 16 bytes is about
  790 MB for the table alone before any text is processed.

So a parameter count says a lot about training memory and very little about speed or capability.

## Why anyone builds a table that big

Lesson 1: DeBERTa's tokenizer breaks SciFact text into 1.46 tokens per word against our 1.68,
and keeps `erythrocyte` and `homozygous` whole. **More rows means fewer fragments.** A bigger
vocabulary buys better word coverage and pays for it in parameters and training memory, not in
speed.
