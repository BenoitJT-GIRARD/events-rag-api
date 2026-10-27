FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv==0.11.6

# uid 1000, named `events`: the API reads a corpus and answers. It writes only under
# /app/var, where /rebuild replaces the index, so the privilege it would hold as root has no
# use here. The image carries no index: docker-compose.yml mounts the host's var/ directory
# at /app/var, where scripts/build_index.py writes it. The directory is created here and
# given to that user, so the route can write to it when nothing is mounted.
RUN useradd --create-home --uid 1000 events \
    && mkdir -p /app/var \
    && chown events:events /app/var

COPY --chown=events:events pyproject.toml uv.lock README.md /app/
COPY --chown=events:events src /app/src
COPY --chown=events:events scripts /app/scripts
COPY --chown=events:events data /app/data
COPY --chown=events:events .env.example /app/.env.example

RUN uv sync --frozen --no-dev

EXPOSE 8000

# The probe calls /health, which answers without loading the index, so a container whose
# index is missing is reported healthy and left running; /ask reports the missing index
# with a 503 that names the script that builds it.
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

USER events

# `--no-sync` runs the environment built above as it is. Without it, `uv run` would try to add
# the default `dev` group at startup, into a directory that belongs to root, and the
# container would exit before serving anything.
CMD ["uv", "run", "--no-sync", "uvicorn", "events_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
