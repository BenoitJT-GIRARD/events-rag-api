"""Score retrieval on the hand-written question set. Deterministic: no judge, no LLM call."""

import json

from events_rag.evaluation.evaluate import run_evaluation

if __name__ == "__main__":
    payload = run_evaluation()
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
