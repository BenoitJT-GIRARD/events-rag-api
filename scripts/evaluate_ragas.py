"""Score the generation side with RAGAS.

Slower and weaker than `evaluate_rag.py`: it calls a model to grade a model, and the
reference answers are derived from the events themselves. Read the caveats in the README
before reading the numbers.
"""

import json

from events_rag.evaluation.ragas_eval import run_ragas_evaluation

if __name__ == "__main__":
    payload = run_ragas_evaluation()
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
