FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv==0.11.6

# uid 1000, named `events`: the API reads a corpus and answers. It writes nothing outside
# the index it rebuilds, so the privilege it would hold as root has no use here, and a
# mounted volume would inherit it.
RUN useradd --create-home --uid 1000 events

COPY --chown=events:events pyproject.toml uv.lock README.md /app/
COPY --chown=events:events src /app/src
COPY --chown=events:events scripts /app/scripts
COPY --chown=events:events data /app/data
COPY --chown=events:events .env.example /app/.env.example

RUN uv sync --frozen --no-dev

EXPOSE 8000

# The probe calls /health, which answers without loading the index: the route is up a
# minute before retrieval is, and the container says so instead of being restarted.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

USER events

CMD ["uv", "run", "uvicorn", "events_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
