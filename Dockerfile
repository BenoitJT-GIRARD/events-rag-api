FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock requirements.txt README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY data /app/data
COPY .env.example /app/.env.example

RUN uv sync --no-dev

EXPOSE 8000

CMD ["uv", "run", "uvicorn", "events_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
