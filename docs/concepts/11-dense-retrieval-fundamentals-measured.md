# Dense retrieval fundamentals, measured on SciFact (2026-10-14)

Train split only (n=505 claims with evidence, 5,183 abstracts). Dev untouched.
Reproduce: `scripts/compare_encoders.py`, `scripts/dense_fundamentals.py`, `scripts/survey_encoders.py`.

## Encoder comparison (each model with its own model-card conventions)

```
ENCODER COMPARISON -- train, n=505 claims with evidence, 5,183 abstracts, m=6 candidates
====================================================================================================
  model                                     hit@1  hit@10    MRR  dim  params  flags
  BM25 (reference)           74.9% [70.9%, 78.6%]   92.7%  0.813    -       -  
  all-MiniLM-L6-v2           68.7% [64.6%, 72.7%]   93.3%  0.786  384     23M  
  all-mpnet-base-v2          69.3% [65.3%, 73.3%]   93.5%  0.782  768    109M  
  bge-small-en-v1.5          80.8% [77.4%, 84.2%]   97.8%  0.873  384     33M  CARD REPORTS MTEB SciFact (= our dev)
  e5-base-v2                 78.0% [74.5%, 81.6%]   96.6%  0.848  768    109M  CARD REPORTS MTEB SciFact (= our dev)
  S-PubMedBert-MS-MARCO      74.7% [70.7%, 78.2%]   96.2%  0.829  768    109M  biomedical
  MedCPT                     78.0% [74.3%, 81.6%]   98.8%  0.857  768    219M  biomedical

  Selection inflation at m=6: roughly sigma*sqrt(2 ln m) on the winner.
```

## FAISS indexes and chunking

```
================================================================================================
PART A -- what each FAISS index stores (5,183 vectors x 384 dims, all-MiniLM-L6-v2)
================================================================================================
  raw vectors alone: 5,183 x 384 x 4 bytes = 7.96 MB
  Flat stores the vectors verbatim: reconstruct(0) == vector 0 ? True

  index                stored    build    query  exact top-10 kept  gold in top 10   what it stores
  Flat (exact)         7.96MB      0ms   15.4ms            100.0%           93.3%   every vector, uncompressed
  IVF, probe 1/64      8.10MB     73ms    2.7ms             48.1%           48.9%   vectors grouped under 64 k-means centroids
  IVF, probe 8/64      8.10MB     64ms    9.8ms             88.8%           83.8%   same, searches 8 nearest groups
  HNSW (M=32)          9.37MB    226ms    5.7ms             97.5%           92.1%   vectors + a navigable neighbour graph
  PQ (48 x 8 bit)      0.64MB   2639ms   50.3ms             76.8%           92.1%   48-byte compressed codes, NOT the vectors

================================================================================================
PART B -- truncation and chunk size
================================================================================================
  encoder max length: 256 tokens
  abstracts longer than that: 3,681 of 5,183 (71.0%)  median length 316, p95 567
  gold rationales that END past the cut (encoder never sees them whole): 516 of 957 (53.9%)

  granularity                   chunks                  hit@1  hit@10    MRR
  whole abstract                 5,183   68.7% [64.6%, 72.7%]   93.3%  0.786   (encode 0s)
  1 sentence (+title)           45,952   76.4% [72.7%, 80.0%]   95.0%  0.833   (encode 78s)
  3-sentence window             35,586   73.7% [69.5%, 77.4%]   95.2%  0.818   (encode 96s)

  Each sentence chunk carries the title, so it keeps the abstract's topic. A chunk
  that loses its context is a different, worse experiment, not 'sentence retrieval'.
```

## Findings

1. **Truncation hides more than half the evidence from all-MiniLM-L6-v2.** Its limit is 256
   tokens; the median abstract is 316 tokens, 71.0% of abstracts are longer, and **53.9% of gold
   rationales end past the cut.** This compounds the Stage 0 finding that evidence sits in the
   back half of abstracts: truncation removes exactly the part where evidence lives.
2. **The encoder comparison is confounded by input length.** MiniLM reads 256 tokens, mpnet 384,
   bge/e5/MedCPT 512. Part of the gap between them is how much of each abstract they see, not
   only model quality.
3. **Sentence chunks (with the title) beat whole abstracts:** hit@1 76.4% vs 68.7%, MRR 0.833 vs
   0.786. They avoid truncation entirely and match SciFact's unit of evidence (one sentence).
4. **Approximate indexes buy nothing at 5,183 vectors and cost recall.** IVF at probe 1 keeps
   only 48.1% of the exact top 10 (gold in top 10 drops 93.3% -> 48.9%). HNSW keeps 97.5%. PQ
   stores 12x less (0.64 MB vs 7.96 MB) but was slower here. Flat is the right index at this size.
5. **The two best hit@1 scores belong to models whose cards report MTEB SciFact (= our dev,
   OQ-014).** bge-small 80.8%, e5-base 78.0%. MedCPT matches e5 at hit@1 (78.0%) with no such
   statement and has the best hit@10 (98.8%). That number caps the reranker's shortlist.
