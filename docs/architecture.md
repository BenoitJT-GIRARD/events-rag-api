# Architecture

This page records the engineering decisions behind the pipeline, with the detail that the README
leaves out. Each of them could reasonably have gone the other way.

## Ingestion

`ingestion/openagenda_client.py` pages through the OpenDataSoft Explore v2.1 API a hundred
records at a time, with a filter on region and on a date window that the server applies. Running
the filter on the server keeps each response small, and the window is stated in the query itself.
The query that produced the committed corpus is written out in
[`../data/raw/SOURCE.md`](../data/raw/SOURCE.md), so a reader can check the snapshot against it.

`ingestion/preprocess.py` turns each record into two parts. The `text` block holds the title, the
short and long descriptions, the conditions, the town, the venue and its address, the start date
and the keywords. The `metadata` object carries `uid`, `title`, `city`, `location_name`,
`location_address`, `date`, `conditions`, `keywords` and `canonicalurl`.

Two consequences of that split appeared later:

- The `uid` carried in the metadata makes deterministic retrieval metrics possible. Every chunk
  keeps the identifier of its event, and that identifier serves as the relevance label.
- The metadata object is not embedded. The vector is built from the text block, which already
  contains the title, town, venue and start date as labelled lines. `dense-metadata-header`
  measures what repeating title, venue, town and date as a header at the top of each event would
  add, and on this corpus the sample supports no gain from it.

## Chunking

The splitter is `RecursiveCharacterTextSplitter`, with chunks of 800 characters and an overlap of
120, cutting at paragraph breaks first, then at line breaks, sentences, words and characters. The
thousand events give 2 046 chunks. The second notebook counts 487 split events, so 513 events
stay in a single chunk.

The start date sits after the long description, far from the title. When an event is split, its
last chunk carries the start date, and in none of the 487 split events does that chunk also carry
the title line. The overlap of 120 characters repeats the end of one chunk at the start of the
next, and the title line stays in the first chunk only.

`dense-one-chunk-per-event` tested indexing each event whole. It scored slightly lower, which fits
the idea that a longer document dilutes its embedding, but the margin is within the noise at this
sample size.

## Index

The index is a FAISS store saved to disk, and the README gives the reasons for this choice over a
managed service. The service loads it through `load_vectorstore` in `rag/retriever.py`, an LRU
cache keyed by directory and index name, so a process reads the index from disk once and keeps it
in memory until `/rebuild` empties the cache.

`evaluation/indexes.py` builds one index per chunking variant, keyed by a fingerprint of the
chunking parameters, under `var/faiss/<fingerprint>/`. It loads each index from disk with FAISS
directly, outside the cache of the service. Only chunking changes the vectors, so the ablation
needs three indexes. Filtering, rank fusion and reranking run at query time on a shared index, so
a new search configuration reuses an existing index and adds only the embedding of its questions.

## Retrieval strategies

Each strategy in `evaluation/strategies.py` exposes `search(query, k) -> list[uid]`, and the
strategies compose with one another:

| Strategy | What it does |
|---|---|
| `DenseSearch` | FAISS similarity over `mistral-embed` vectors |
| `BM25Search` | lexical scoring, the floor the other strategies are compared with |
| `HybridSearch` | reciprocal rank fusion of the two |
| `CityFilter` | keeps events in the town the query names |
| `Rerank` | reorders candidates with an injected scoring function |

Two details are easy to get wrong.

**Deduplicate by `uid` before ranking.** A reader asks about events, and the index returns chunks.
One event can take three of the first five positions, which leaves room for only three distinct
events in a top five. Dense and BM25 search deduplicate the uids they return, and the other
strategies work on those lists. A unit test checks the dense case.

**Over-fetch before filtering.** Deduplication and filtering both shrink the result list, so
asking the inner search for exactly `k` results returns fewer than `k`. The strategies request
`k × 4` and trim afterwards.

`HybridSearch` combines the two rankings with reciprocal rank fusion (RRF), which adds
`1 / (60 + rank)` over the lists that found a document. RRF uses ranks only, so a dense similarity
and a BM25 score need no common scale, and either side can change without retuning. It
favours agreement between the lists. An event ranked tenth by both scores 2/70, about 0.029, and
outranks an event that only one list found, even at rank one, which scores 1/61, about 0.016.

`CityFilter` matches town names against those present in the corpus, longest name first, so that
`Castelnaudary` is matched before `Castelnau`. When it recognises no town in the query, it passes
the results through unchanged.

`Rerank` receives its scoring function as an argument. The unit tests pass a plain function, and
the ablation passes a FlashRank cross-encoder built in `_flashrank_scorer`, the one place to
change for another backend.

## Evaluation

`evaluation/retrieval.py` holds the metrics, `recall@k` and `MRR@10` against the `uid` of the
source event, as short functions written out in full so that a reader can check them. It also
holds the rank of the target per question and McNemar's exact test, which the paired comparison
uses. Only the twenty positive cases enter these metrics, since the negative and ambiguous cases
have no target event. In the answer grading of `evaluation/evaluate.py`, the five negative cases
are scored on whether the answer declines. The five ambiguous cases carry no keywords and no town,
so every answer to them is labelled partially correct.

`evaluation/difficulty.py` measures the lexical overlap between a question and the event it should
retrieve. It turned the suspicion that the benchmark was too easy into a number that could be
acted on.

`evaluation/ablation.py` runs each configuration and keeps a failed configuration in the results,
with its error message in place of metrics. That rule came from the RAGAS summary, which had shown
two metrics as null over a long stretch of the commit history, one of them only because the
summary read the wrong column.

## API

`api/main.py` exposes `/health`, `/metadata`, `/ask` and `/rebuild`. The token that guards
`/rebuild` is first a cost control. The route embeds the 2 046 chunks of this corpus through the
paid API, and the docstring of the module gives that as the reason for the token.

The prompt in `rag/prompts.py` instructs the model to answer only from the supplied context and
to say when the context does not hold the answer. Five of the thirty evaluation cases test that
instruction with questions about events outside the corpus, which would show a model that starts
inventing events.

## Alternatives the service leaves out

The service uses no vector database, no reranking, no query rewriting and no cache in front of the
API. Reranking is the only one of the four that the ablation measured, and it ranked below the
configuration that ships. The README gives the reason for leaving out a vector database, and notes
that query rewriting, which was not tested, would add one model call per question. No page gives a
reason for leaving out the cache.
