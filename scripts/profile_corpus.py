"""Describe the committed corpus in numbers, so the pages that describe it can cite them.

Writes reports/corpus_profile.json. No model, no index: it reads data/raw/events.json and
counts.
"""

import json
from pathlib import Path

from events_rag.config import get_settings
from events_rag.ingestion.profile import profile_corpus


def main() -> int:
    settings = get_settings()
    events = json.loads((Path(settings.raw_data_dir) / "events.json").read_text(encoding="utf-8"))
    payload = {
        "corpus": "data/raw/events.json",
        **profile_corpus(events),
        "written_by": "scripts/profile_corpus.py",
    }
    output = Path(settings.reports_dir) / "corpus_profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline=""
    )
    print(f"wrote {output} — {payload['events']} events, {payload['distinct_towns']} towns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
