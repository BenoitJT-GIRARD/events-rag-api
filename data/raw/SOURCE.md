# Dataset

- **Name:** Public events - OpenAgenda (`evenements-publics-openagenda`)
- **Publisher:** OpenAgenda
- **Distributed by:** OpenDataSoft — https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda/
- **Licence:** Licence Ouverte / Open Licence v1.0 — https://www.etalab.gouv.fr/wp-content/uploads/2014/05/Licence_Ouverte.pdf
- **Extracted on:** 2026-09-02
- **Records in this snapshot:** 1000 (1.48 MB)

The Open Licence allows redistribution, including for commercial purposes, and requires
attribution to the source and its last update date.

## Extraction parameters

```
location_region = 'Occitanie'
firstdate_begin  >= 2025-09-02
firstdate_begin  <= 2026-09-02
lang             = fr
timezone         = Europe/Paris
limit            = 1000 records, fetched 100 at a time
```

## Why this snapshot is committed

Upstream is a rolling date window over a live dataset. Re-ingesting produces a different
corpus, so no measurement in this repository would be reproducible — neither by a reader
nor by its author six months later. Freezing the corpus is what makes the ablation
results mean anything.

## Shape

Each record carries `text` and a `metadata` object with `uid`, `title`, `city`,
`location_name`, `location_address`, `date`, `conditions`, `keywords`, `canonicalurl`.

<!-- source: reports/corpus_profile.json -->
The snapshot spans **414 towns across Occitanie**, led by Toulouse (161 events), Alès (63),
Nîmes (22), Perpignan (19) and Montpellier (18); one record carries no town at all. It is a
regional corpus, and `scripts/profile_corpus.py` recounts every figure in this paragraph.
