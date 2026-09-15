# How this repository is evaluated

Two benchmarks, and they are not the same exercise. Retrieval is scored against a hard label
and costs nothing to re-run. Generation is scored by a model judging a model, costs money, and
is the weaker half. This page says how each one is built, what it costs, and where its
numbers land.

## The question set

`data/questions/reference_qa.json` holds 30 cases: 20 positive, and 10 that are negative or
ambiguous. A positive case records the `uid` of the event it was written from, which is what
makes retrieval scorable without a judge — the retrieved list either contains that event or
it does not.

The events behind the positive cases were sampled under a fixed seed
(`EVENTS_RAG_EVAL_SEED`), category by category, by
`src/events_rag/evaluation/generate_dataset.py`. Sampling rather than taking the head of the
file keeps the set from concentrating on one slice of the corpus; fixing the seed is what
lets two ablation runs be compared at all.

### Why the first set was thrown away

A model wrote the first set from the events themselves, titles included. Every question then
retrieved its source at rank one, BM25 alone reached 0.90 at `recall@5`, and a benchmark at
the ceiling ranks nothing.

`scripts/measure_difficulty.py` puts a number on that. It measures, for each question, the
share of its content words that appear verbatim in the event it should retrieve. The generated
set scored 0.61 on the run of 2026-09-02. The rewrite drew from the same seeded sample, so
no question was picked for being easy, and the set now scores 0.5, recorded in
`reports/difficulty.json`. The
residue is town names, which someone asking about an outing genuinely says.

That earlier 0.61, and the scores the generated set produced, cannot be re-measured here: the
generated set was replaced rather than kept. It is the one pair of numbers in this repository
that no command reproduces, and the README says so where it uses them.

## Retrieval: the ablation

```bash
EVENTS_RAG_MISTRAL_API_KEY=... uv run python scripts/build_index.py    # ~1 min, paid
EVENTS_RAG_MISTRAL_API_KEY=... uv run python scripts/run_ablation.py   # ~2 min, paid
uv run python scripts/plot_ablation.py                                 # free, offline
```

Seven configurations, the same twenty questions, the same corpus. Each configuration is
scored with `recall@k` and `MRR@10`, defined in [`metrics.yaml`](../metrics.yaml). The run
writes `reports/ablation_results.json` and `reports/ablation_table.md`; the plot script
redraws `reports/figures/ablation.svg` from the first of those, and records the image in
`reports/figures/MANIFEST.json`.

**Cost.** Embedding the corpus is 2 046 calls. The ablation builds three indexes, one per
chunking variant, and reuses them across configurations, so the bill is dominated by the
first build.

**What the numbers can carry.** n = 20. The standard error of a proportion at that size is
about 6.7 points at p = 0.9, so two configurations within roughly 13 points are not
distinguishable. The README states this next to the table, and no conclusion is drawn across
a gap the sample cannot support.

## Generation: the RAGAS run

```bash
EVENTS_RAG_MISTRAL_API_KEY=... uv run python scripts/evaluate_rag.py     # heuristic grading
EVENTS_RAG_MISTRAL_API_KEY=... uv run python scripts/evaluate_ragas.py   # model grading
```

The first writes `reports/evaluation_results.json`: keyword coverage, town match, and a
refusal test for the negative cases, all deterministic. The second writes
`reports/ragas_results.json`: faithfulness, context precision and context recall, each
computed by a model. Neither file is committed, because neither has been produced since the
account ran out of chat quota.

**One of the five metrics returns null on this setup.** `answer_relevancy` is reported as
not measured, and `NaNSafeEncoder` is what keeps it a `null` through serialisation instead of
the token `NaN`, which no JSON reader accepts. The mapping from a published metric to the
column RAGAS writes it into lives in one place, `RAGAS_COLUMN`; it was written twice before,
and the two spellings of context relevancy disagreed.

**The reference answers come from the events themselves**, so these numbers are optimistic by
construction. Writing them blind is a different exercise, and the README lists it among the
things this repository does not prove.

## What a re-run must not do

`scripts/build_dataset.py` re-ingests from the live portal. Upstream is a rolling window, so
running it produces a different corpus and every number on this page stops applying.
`scripts/generate_eval_dataset.py` does the same to the question set. Neither belongs in a
reproduction: the corpus and the questions are committed so that everything else can be
re-run against them.

## Withdrawn results

`reports/errata.json` carries every row that stopped being a measurement, with its reason,
its fix and the command that measures it again; `events_rag.evaluation.errata` is what reads
it, so the figure and the README agree on which rows count. One entry stands, resolved: the
reranking configuration, whose passages were assembled wrongly before scoring.

## The archived run of 2026-09-02

Two files carry the state of the grading chain on the day the methodological audit read it.
Neither the questions nor the corpus of that day survive: the question set was replaced by
the hand-written one, and the corpus covered Montpellier alone before the region-wide
extraction took over. Two of the 139 events their answers cite are still in the corpus.

- `reports/evaluation_2026-09-02.json` — the thirty answers, the summary the defective
  scorer published, and the summary the current code computes from the same text.
  `scripts/rescore_archive.py` regenerates the second, deterministically and without a model
  call, which is how the correction of the refusal detector is verified today.
- `reports/ragas_2026-09-02.json` — the model grading of the same thirty answers, kept for
  `avg_context_precision` = 0.575, the number that made the retrieval side worth a full
  ablation. Its means are recomputed from its own rows, each with the count it was taken
  over.

Both are dated in their names because they measure a setup this repository no longer carries.
Nothing in the README is derived from them apart from the before-and-after of the scorer, and
the one metric that comes back null.
