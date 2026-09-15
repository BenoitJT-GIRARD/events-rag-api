"""Measure how much the published question set borrows from the events it should retrieve.

Writes reports/difficulty.json. No model and no index: the measure is a word overlap between
a question and the event it was written from, so it runs offline and on any checkout.
"""

import json
from pathlib import Path

from events_rag.config import get_settings
from events_rag.evaluation.difficulty import measure_question_set


def main() -> int:
    settings = get_settings()
    cases = json.loads(
        (Path(settings.questions_dir) / "reference_qa.json").read_text(encoding="utf-8")
    )
    events = {
        event["metadata"]["uid"]: event["text"]
        for event in json.loads(
            (Path(settings.raw_data_dir) / "events.json").read_text(encoding="utf-8")
        )
        if (event.get("metadata") or {}).get("uid")
    }

    payload = {
        "question_set": "data/questions/reference_qa.json",
        **measure_question_set(cases, events),
        "written_by": "scripts/measure_difficulty.py",
    }
    output = Path(settings.reports_dir) / "difficulty.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline=""
    )
    print(f"wrote {output} — mean overlap {payload['mean_lexical_overlap']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
