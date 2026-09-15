# The API surface

Four routes. The authority is the OpenAPI schema the application generates from its Pydantic
models, served at `/docs` and `/openapi.json` when the service is running; this page is the
map, and says what the schema cannot: what each route costs and how it fails.

## `GET /health`

Liveness. Answers `{"status": "ok"}` as soon as the process is up, without touching the index
or the model — which is what makes it usable as a container probe. A service whose health
check loaded the index would report unhealthy for the minute it takes to load, and would
report healthy afterwards even if the model behind it had gone.

## `GET /metadata`

What this deployment is serving: the region and field the corpus was filtered on, its
language, the date window, and the number of documents retrieval asks for. Derived from the
settings object, so it cannot drift from the running configuration.

Use it to tell two deployments apart before comparing their answers.

## `POST /ask`

The product. Takes a question and an optional depth:

```json
{ "question": "Un atelier vélo à Toulouse au printemps ?", "top_k": 3 }
```

`question` is 3 to 500 characters and `top_k` is 1 to 10; both bounds are in the schema, so a
malformed call is refused before it reaches the model. The answer carries the events it was
built from, each with its `uid`, so any sentence can be traced to a record in the corpus.

**Cost.** One embedding call and one generation call per question, both paid. Typical end to
end: about 2.5 s.

**Failures.** `400` for a question the retriever cannot use, `503` when the index is absent —
the message names the script that builds it — and `500` for anything else, with the error in
the detail rather than swallowed.

## `POST /rebuild`

Re-embeds the committed corpus and replaces the index on disk. Takes the token configured in
`EVENTS_RAG_REBUILD_TOKEN`; without a token on the server the route answers `500` rather than
running unguarded, and with a wrong one it answers `403`.

**Cost.** About a minute, and one embedding call per chunk: 2 046 of them on the committed
corpus. This is the expensive route, which is why it is the only one behind a secret.

It rebuilds from `data/raw/events.json`. It does not re-ingest: nothing here reaches the
upstream portal, and the corpus a deployment serves is the one committed beside it.
