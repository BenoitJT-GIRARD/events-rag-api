# How this repository is evaluated

The repository evaluates retrieval and generation separately. Retrieval is scored against a fixed
label, and running it again costs embedding calls only. Generation is graded by keyword and
refusal checks and by RAGAS, where a model judges the answers of a model. Both gradings need chat
calls to produce the answers, and the generation side is the weaker of the two evaluations. This
page describes how each one is built, what it costs, and where its numbers are written.

## The question set

`data/questions/reference_qa.json` holds 30 cases: 20 positive, 5 negative and 5 ambiguous. Each
positive case stores the `uid` of the event it targets, so retrieval can be scored without a
judge, by checking whether that event is in the retrieved list.

`src/events_rag/evaluation/generate_dataset.py` sampled the events behind the positive cases under
a fixed seed (`EVENTS_RAG_EVAL_SEED`), two per category. Sampling, as opposed to taking the first
matching events of the file, keeps the set from concentrating on one slice of the corpus. The
seed fixes which events are drawn, so a regenerated set targets the same twenty events.

### Why the generated question set was replaced

Mistral's chat model wrote the earlier set from the events themselves, and in most of its
questions the event title, or a name taken from it, appeared word for word. On that set, the dense
baseline retrieved every source event at rank one and BM25 alone reached 0.90 at `recall@5`. With
the shipped configuration at 1.00, no other configuration could show an improvement.

`scripts/measure_difficulty.py` measures that effect. For each question, it takes the words of
four characters or more, folds their accents, and computes the fraction found word for word in the
target event. The generated set scores 0.61. A language model other than Mistral then rewrote the
twenty positive questions for the same events of the seeded sample, so no question was chosen for
being easy, and the set now scores 0.5, recorded in `reports/difficulty.json`. Words from town
names make up about a quarter of the remaining shared words, 22 of 97. The commit that brought the
rewritten questions also swapped all ten negative and ambiguous questions for new ones, and the
ambiguous ones lost their expected keywords.

The generated set was removed from the working tree. The git history still holds it as
`data/eval/reference_qa.json`, with the ablation results it produced in
`data/eval/ablation_results.json`. The current scripts read the committed set only, and the README
says which of its figures come from that set.

## Retrieval: the ablation

```bash
EVENTS_RAG_MISTRAL_API_KEY=... uv run python scripts/build_index.py    # ~1 min, paid
EVENTS_RAG_MISTRAL_API_KEY=... uv run python scripts/run_ablation.py   # ~3 min, paid
uv run python scripts/plot_ablation.py                                 # free, offline
```

The first command builds the index that the API serves. The reranking configuration needs the
optional `rerank` extra, installed with `uv sync --extra rerank`. Without it, the run records that
configuration as failed. `run_ablation.py` builds or reuses its own indexes under `var/faiss/`,
then runs seven configurations on the same twenty questions and the same corpus. It scores each
configuration with `recall@k` and `MRR@10`, which [`metrics.yaml`](../metrics.yaml) defines, and
writes `reports/ablation_results.json` and `reports/ablation_table.md`. The plot script redraws
`reports/figures/ablation.svg` from the results file and records the image in
`reports/figures/MANIFEST.json`.

Embedding the corpus covers its 2 046 chunks, grouped by the Mistral client into requests of at
most 16 000 tokens each. The ablation builds three indexes, one per chunking variant, and reuses
each across the configurations that share it. It then embeds each question once per
configuration that includes dense search. The three index builds account for most of the cost.

The sample is n = 20. The standard error of a proportion at that size is about 6.7 points at
p = 0.9, so a gap of less than roughly 13 points between two configurations cannot be told apart
from noise by that measure. All configurations answer the same questions, so
`scripts/compare_paired.py` also runs McNemar's exact test on the paired results and writes
`reports/paired_comparison.json`, which the README presents beside the ablation table.

## Generation: answer grading and RAGAS

```bash
EVENTS_RAG_MISTRAL_API_KEY=... uv run python scripts/evaluate_rag.py     # heuristic grading
EVENTS_RAG_MISTRAL_API_KEY=... uv run python scripts/evaluate_ragas.py   # model grading
```

Both scripts generate an answer to each of the thirty questions with the service code. The first
writes `reports/evaluation_results.json`, which grades each answer by keyword coverage, by town
match and, for the negative cases, by a refusal test, with no model involved in the grading. The
second writes `reports/ragas_results.json`, with five metrics that a judge model computes:
faithfulness, answer relevancy, context precision, context recall and context relevancy. Neither
file is committed, because neither has been produced since the account ran out of chat quota.

**One of the five metrics returned null on every answer.** `answer_relevancy` is reported as not
measured. `NaNSafeEncoder` writes a missing value as `null`, where the default encoder would write
the token `NaN`, which is not valid JSON and which strict parsers reject. The mapping from each
published metric to the column RAGAS writes it into is defined once, in `RAGAS_COLUMN`. It used to
be written in two places, and the two spellings of context relevancy disagreed.

**The reference answers come from the events themselves.** Of the five metrics, context precision
and context recall compare the retrieved text with these references, so their scores are
optimistic by construction. Writing the references blind would be a separate piece of work, and
the README lists it among the things this repository does not prove.

## Scripts that replace a frozen input

`scripts/build_dataset.py` re-ingests from the live portal. The portal serves a rolling window,
so running the script produces a different corpus, and every number on this page stops applying.
`scripts/generate_eval_dataset.py` has the same effect on the question set. Neither script is
part of a reproduction. The corpus and the questions are committed so that every other script
can run against the same inputs.

## Withdrawn results

`reports/errata.json` lists every published row that was found not to measure what it claimed,
with the reason, the fix and the command that measures it again. `events_rag.evaluation.errata`
reads that file for both the figure script and the test that checks the README, so the two agree
on which rows count. The file holds one entry, now resolved: the reranking configuration, whose
passages were assembled wrongly before scoring.

## The archived evaluation run

Two files keep the state of the grading chain at the time its defects were found. The thirty
questions of that run are stored in them, but the set was replaced, first by a set regenerated on
the Occitanie corpus and then by the rewritten one. The corpus of that run covered Montpellier
alone and was never committed, and two of the 83 distinct events its answers cite are in the
committed corpus.

- `reports/evaluation_2026-09-02.json` holds the thirty answers, the summary the defective scorer
  published, and the summary the current code computes from the same text.
  `scripts/rescore_archive.py` regenerates the second summary deterministically and without a
  model call, which allows the correction of the refusal detector to be checked today.
- `reports/ragas_2026-09-02.json` holds the model grading of answers to the same thirty
  questions, generated in a separate pass, with 25 of the 30 answers identical to those of the
  first file. It is kept as the record of that grading, including its `avg_context_precision`
  of 0.575. Its means are recomputed from its own rows, each with the count of rows it was taken
  over.

Both files are named apart from the `evaluation_results.json` and `ragas_results.json` that a new
run writes, because they measure a setup that this repository no longer contains. The README cites
them for the corrections to the scorer and to the RAGAS summary, and for examples of refusals, and
presents none of their numbers as a measurement of the current service.
