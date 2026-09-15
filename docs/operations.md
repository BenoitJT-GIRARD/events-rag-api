# Running and operating the service

## Starting it

```bash
cp .env.example .env                      # then set EVENTS_RAG_MISTRAL_API_KEY
uv sync
uv run python scripts/build_index.py      # ~1 min, 2 046 paid embedding calls
docker compose up                         # API on :8000, OpenAPI page at /docs
```

The index is written under `var/faiss/<index name>/` and is not committed: it is rebuilt from
the committed corpus, and rebuilding is cheaper than storing a few thousand vectors in git.
The container rebuilds it at image build time, so a fresh `docker compose up` needs the key
once.

Without a key the service still starts and `/health`, `/metadata` and the OpenAPI page all
answer. `/ask` returns 503 with a message naming the script that builds the index. That is
deliberate: a liveness probe that needed the model would report unhealthy for the minute the
index takes to load, and healthy afterwards even if the model behind it had gone.

## Watching it

| Signal | Where | What it means |
|---|---|---|
| `GET /health` | the container's `HEALTHCHECK` | the process is up and serving |
| `GET /metadata` | manual | which corpus and which retrieval settings this deployment carries |
| `events_rag` structured logs | stdout, JSON | one line per question: the question, the depth, the number of documents retrieved |

The logs are structured, so a question that retrieved nothing is a field, not a sentence to
grep. `EVENTS_RAG_LOG_LEVEL` sets the level; the default is `INFO`, which logs one line per
answer and nothing per chunk.

## Rebuilding the index

Two ways, and they do the same work:

```bash
uv run python scripts/build_index.py
curl -X POST localhost:8000/rebuild -d '{"token":"..."}' -H 'Content-Type: application/json'
```

The route is guarded by `EVENTS_RAG_REBUILD_TOKEN`. An empty token on the server refuses every
call, which is the right default for a service nobody is watching; a wrong token answers 403.
Rebuilding costs one embedding call per chunk, so the guard is a cost control as much as a
security one.

Rebuilding does **not** re-ingest. Nothing in the running service reaches the upstream portal:
the corpus it serves is the one committed beside it, and replacing that corpus is a separate,
documented act (`scripts/build_dataset.py`).

## Costs, in one place

| Operation | Calls | Rough time |
|---|---|---|
| Build the index | 2 046 embeddings | 1 min |
| Answer one question | 1 embedding + 1 generation | 2.5 s |
| Run the ablation | three index builds, then local search | 3 min |
| Run the RAGAS evaluation | 30 generations plus the judge's own calls | 10 min |

## When something is wrong

**`/ask` answers 503.** The index is missing. Run `scripts/build_index.py`, or check that
`EVENTS_RAG_INDEX_DIR` points where the index was written.

**`/ask` answers 400.** The question failed validation: under 3 characters, over 500, or a
`top_k` outside 1 to 10. The bounds are in the OpenAPI schema.

**Answers are correct but cite the wrong town.** Retrieval is working and the filter is not:
check `EVENTS_RAG_RETRIEVAL_K`, and read the city-filter row of the ablation table before
concluding anything from a single question.

**Everything answers, but nothing is found.** The index may have been built from a different
corpus. `var/faiss/<name>/manifest.json` records the chunking parameters, the embedding model
and the number of source documents it was built from; compare it with
`reports/corpus_profile.json`.

## Decommissioning

This deployment is archived. The API key that produced the published answers is revoked, and
the service runs locally or not at all. Nothing here depends on a hosted component: the
corpus, the question set, the results and the figure are all in the repository, and the only
external dependency is a model provider that a reader supplies themselves.
