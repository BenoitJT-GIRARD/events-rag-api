# Where the corpus comes from, and what may be done with it

## The dataset

[Événements publics OpenAgenda](https://public.opendatasoft.com/explore/dataset/evenements-publics-openagenda/),
published by [OpenAgenda](https://openagenda.com/) and distributed by OpenDataSoft on their
public data portal. It aggregates the event listings that thousands of French organisers
publish through OpenAgenda: concerts, exhibitions, workshops, guided tours, markets.

## What was taken

| | |
|---|---|
| Endpoint | `https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/evenements-publics-openagenda/records` |
| Filter | `location_region = "Occitanie"`, French-language records |
| Window | the 365 days preceding the extraction |
| Records | 1 000, the ceiling set by `EVENTS_RAG_INGESTION_MAX_RECORDS` |
| Extracted on | 2026-09-02 |
| Written to | `data/raw/events.json` (1.5 MB, committed) |

Each record is flattened to a text block and a metadata block by
`src/events_rag/ingestion/preprocess.py`. The text holds the title, description, venue, town
and dates; the metadata holds the `uid`, the town and the start date. The `uid` is the key
everything else hangs off: the evaluation uses it as the relevance label, and the API returns
it with every source so an answer can be traced back to a record on the portal.

## Why it is committed rather than fetched

Upstream is a rolling window: the same query run tomorrow returns a different set of events,
because past events fall out of it and new ones appear. A repository that fetched its corpus
at run time would publish numbers that nobody could reproduce, including its author. So the
1 000 records are committed, and `scripts/build_dataset.py`, the script that would replace
them, is documented as the one command that invalidates every published figure.

## Licence

The data is published under the
[Licence Ouverte / Open Licence v1.0](https://www.etalab.gouv.fr/wp-content/uploads/2014/05/Licence_Ouverte.pdf),
which permits reproduction, redistribution and reuse, including commercially, on one
condition: the source and the date of the last update must be stated. That is what this page
and [`data/raw/SOURCE.md`](../data/raw/SOURCE.md) are for.

Nothing else from the portal is redistributed here. The images that some records link to are
not downloaded; only the text and the metadata above are.

## Personal data

The records describe public events, not people. Organiser names appear where the organiser is
an association or a venue, which is public information on the portal itself. No contact
details, no ticketing data and no attendee information enter the corpus: the ingestion keeps
the fields listed above and drops the rest.
