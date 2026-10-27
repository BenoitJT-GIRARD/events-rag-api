# Where the corpus comes from, and what may be done with it

## The dataset

The corpus comes from
[Événements publics OpenAgenda](https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda/),
a dataset published by [OpenAgenda](https://openagenda.com/) and distributed by OpenDataSoft on
its public data portal. It gathers the event listings that thousands of French organisers publish
through OpenAgenda, such as concerts, exhibitions, workshops, guided tours and markets.

## What was taken

| | |
|---|---|
| Endpoint | `https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records` |
| Filter | `location_region = 'Occitanie'`, with `lang=fr` passed to the API; the ingestion reads the French fields of each record |
| Window | events starting in the 365 days before the extraction |
| Records | 1 000, the ceiling set by `EVENTS_RAG_INGESTION_MAX_RECORDS` |
| Extracted on | 2026-09-02 |
| Written to | `data/raw/events.json` (1.5 MB, committed) |

`src/events_rag/ingestion/preprocess.py` turns each record into a text block, which is the part
that gets embedded, and a metadata block. [`architecture.md`](architecture.md) lists the fields
of both. The rest of the pipeline relies on the `uid` of each record, kept in the metadata. The
evaluation uses it as the relevance label, and the API returns it with every source, so an answer
can be traced back to a record on the portal.

## Why the corpus is committed

The portal serves a rolling window. The same query run on another day returns a different set of
events, because past events leave the window and new ones enter it. A repository that fetched its
corpus at run time would publish numbers that no later run could reproduce, its author's
included. The 1 000 records are therefore committed. `scripts/build_dataset.py`, the script that
would replace them, is documented as a command that invalidates every published figure, and so
is `scripts/generate_eval_dataset.py` for the question set.

## Licence

The data is published under the
[Licence Ouverte / Open Licence v1.0](https://www.etalab.gouv.fr/wp-content/uploads/2014/05/Licence_Ouverte.pdf),
which permits reproduction, redistribution and reuse, including for commercial purposes. Its one
condition is to state the source and the date of its last update. This page and
[`data/raw/SOURCE.md`](../data/raw/SOURCE.md) name the source and give the date of the extraction,
2026-09-02.

Only the text and metadata described above are taken from the portal. The images that some
records link to are not downloaded.

## Personal data

The records describe public events. Descriptions name organisers, performers and workshop leaders,
some of them private individuals, as the portal publishes them. The ingestion keeps the fields
listed above and drops the others, so no contact, ticketing or attendee field of the portal enters
the corpus. The free-text descriptions still contain what organisers wrote in them, and some of
them include a phone number or an email address.
