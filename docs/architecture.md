# Architecture

Engineering decisions behind the pipeline, for a reader who has read the README and wants
the detail. Everything here is a choice that could reasonably have gone the other way.

## Ingestion

`ingestion/openagenda_client.py` pages the OpenDataSoft Explore v2.1 API a hundred records
at a time, filtered server-side by region and by a date window. Filtering server-side
rather than locally keeps the payload small and the window declarative — the query that
produced the committed corpus is written out in [`../data/raw/SOURCE.md`](../data/raw/SOURCE.md),
which is what makes the snapshot auditable.

`ingestion/preprocess.py` flattens each record into two things: a `text` block that
concatenates title, short and long descriptions, conditions, town, venue and dates; and a
`metadata` object carrying `uid`, `title`, `city`, `location_name`, `location_address`,
`date`, `conditions`, `keywords`, `canonicalurl`.

Two consequences of that split are worth naming, because both surfaced later:

- The `uid` travelling in metadata is what makes deterministic retrieval metrics possible
  at all. Without a stable identifier on every chunk there is no relevance label.
- Metadata is **not** embedded. Title and venue live outside the vector, so a query naming
  a venue has nothing to match. `dense-metadata-header` in the ablation tests putting them
  back into the text; it did not help on this corpus, but the reasoning was sound enough
  to measure.

## Chunking

`RecursiveCharacterTextSplitter`, 800 characters, 120 of overlap, splitting on paragraph,
then line, then sentence, then word. A thousand events produce about 2 000 chunks, so most
events survive as one or two.

The overlap matters more than it looks: an event's date and price often sit at the end of
the text, far from the title, and a split with no overlap strands them in a chunk that no
longer says what event it belongs to.

`dense-one-chunk-per-event` tested indexing each event whole. It scored slightly worse,
which is the expected direction — longer documents dilute the embedding — but the margin
is inside the noise at this sample size.

## Index

FAISS, saved to disk, loaded through an LRU cache. The reasoning is in the README: a few
thousand vectors do not justify a managed service.

The cache is sized for eight entries rather than one because the ablation holds several
indexes open at once; at one entry it evicted on every alternation and reloaded from disk.

`evaluation/indexes.py` builds one index per **chunking variant**, keyed by a fingerprint
of the chunking parameters, under `var/faiss/<fingerprint>/`. This is what keeps the
ablation cheap: only chunking changes the vectors. Filtering, rank fusion and reranking
are query-time operations on a shared index, so a new configuration costs no embedding
calls at all.

## Retrieval strategies

Each strategy in `evaluation/strategies.py` exposes `search(query, k) -> list[uid]` and
composes with the others:

| Strategy | What it does |
|---|---|
| `DenseSearch` | FAISS similarity over `mistral-embed` vectors |
| `BM25Search` | lexical scoring, the honest floor to compare against |
| `HybridSearch` | reciprocal rank fusion of the two |
| `CityFilter` | keeps events in the town the query names |
| `Rerank` | reorders candidates with an injected scoring function |

Two details that are easy to get wrong:

**Deduplicate by `uid` before ranking.** One event yields several chunks, so a top-k of
chunks collapses to fewer distinct events. Ranking chunks instead of events inflates
recall — the same event can occupy three of the first five positions. Every strategy
deduplicates, and there is a test pinning it.

**Over-fetch before filtering.** Because deduplication and filtering both shrink the
result list, asking the inner strategy for exactly `k` returns short. Strategies request
`k × 4` and trim afterwards.

`HybridSearch` uses reciprocal rank fusion rather than a weighted score sum, because RRF
needs no calibration between a cosine similarity and a BM25 score — there is no shared
scale to tune, and nothing to re-tune when either side changes. Its known quirk is that it
rewards *agreement* rather than *closeness to the top*: a document ranked first and third
beats one ranked second twice, since `1/61 + 1/63 > 2/62`.

`CityFilter` matches town names against those present in the corpus, longest first so
`Castelnaudary` wins over `Castelnau`, and passes results through untouched when it
recognises no town. It never removes what it cannot justify removing.

`Rerank` takes its scorer as an argument rather than constructing a model, so the unit
tests need no download and swapping the backend touches one call site.

## Evaluation

`evaluation/retrieval.py` holds the metrics — `recall@k` and `MRR@10` against the source
event's `uid`, roughly thirty lines, written out rather than imported so a reader can
check them. Only the twenty positive cases enter these metrics; the negative and ambiguous
cases have no target event and are scored on refusal instead.

`evaluation/difficulty.py` measures the lexical overlap between a question and the
document it should retrieve. It exists because "this benchmark is too easy" needed to be a
number before it could be acted on.

`evaluation/ablation.py` runs each configuration and **reports failures in the results
table with their error message** instead of dropping them. That rule came from watching
two RAGAS metrics return null silently for months.

## API

`api/main.py` exposes `/health`, `/metadata`, `/ask` and `/rebuild`. `/rebuild` is guarded
by a token from the environment: rebuilding the index is expensive and calls a paid API,
so it is not something an anonymous caller should be able to trigger.

The prompt in `rag/prompts.py` instructs the model to answer only from the supplied
context and to say when it cannot. That instruction is not decoration — five of the thirty
evaluation cases ask about events outside the corpus, and they exist to catch the day it
starts inventing them.

## What is deliberately absent

No vector database, no reranking in the shipped path, no query rewriting, no caching layer
in front of the API. Each was considered; none earned its place at this size. The ablation
in the README is the record of which ones were measured before being set aside.
