# Map — module, purpose, status

For cold sessions. Keep it current; a stale map is worse than no map.

| Module | Purpose | Status |
|---|---|---|
| `scripts/download_data.py` | Fetch SciFact into `data/` (gitignored). Reproducible and verified. | working |
| `scripts/peek.py` | Display claims with gold evidence marked, for reading by hand. Displays only — never summarises. | working |
| `src/scifact/data/` | Loading, schemas, typed access to claims and corpus | empty |
| `src/scifact/retrieval/` | BM25, dense retrieval, reranking | empty |
| `src/scifact/eval/` | Metrics and harness. **Metric bodies are Mounir's to write.** | empty |
| `src/scifact/cli.py` | Entry point. Every reported number comes from a command here. | scaffolded |
| `tests/` | pytest suite, CI gate | smoke only |

## Conventions, and the reason for each

- **`src/` layout, not flat.** Tests run against the *installed* package. An import that only
  works because you happened to be standing in the repo root will fail in CI instead of
  silently passing locally.
- **CLI, not notebooks, for anything reported.** Every number must come from a re-runnable
  command. Notebooks are fine for looking at data; they are a bad place for results you will
  later have to defend.
- **Dependencies are added when a stage needs them,** not upfront. A dependency you cannot
  point at a milestone for is a dependency you are carrying for decoration.
- **Data is never committed.** `data/` is gitignored; reproduce it with the download script.
