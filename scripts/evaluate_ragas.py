import json

from events_rag.evaluation.ragas_eval import run_ragas_evaluation

if __name__ == "__main__":
    payload = run_ragas_evaluation()
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
