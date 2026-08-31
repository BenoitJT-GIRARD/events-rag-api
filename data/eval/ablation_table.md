| Configuration | Chunking | recall@1 | recall@5 | MRR@10 | Median latency | Notes |
|---|---|---|---|---|---|---|
| `bm25-only` | baseline | 0.7 | 0.9 | 0.775 | 3.3 ms | Trivial floor: no embeddings at all. |
| `dense-baseline` | baseline | 1.0 | **1.0** | 1.0 | 211.9 ms | The shipped configuration. |
| `dense-one-chunk-per-event` | one-chunk-per-event | 0.9 | 0.95 | 0.925 | 178.9 ms | Index each event whole instead of splitting it. |
| `dense-metadata-header` | metadata-header | 0.9 | 0.95 | 0.925 | 177.0 ms | Embed title, venue, city and date with the text so proper nouns match. |
| `dense+city-filter` | baseline | 0.95 | 0.95 | 0.95 | 163.1 ms | Narrow to the town named in the question. Part of any gain may be an artefact: questions were generated from an event whose town the prompt showed the model. |
| `hybrid-rrf` | baseline | 0.85 | 0.9 | 0.881 | 168.0 ms | Reciprocal rank fusion of dense and lexical search. |
| `hybrid-rrf+rerank` | baseline | 0.4 | 0.7 | 0.527 | 213.4 ms | Cross-encoder reranking on top of hybrid search. Optional dependency, 41 MB. |
