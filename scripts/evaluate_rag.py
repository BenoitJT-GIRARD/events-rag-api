"""Ask the thirty questions of data/questions/ through the service, and score the answers.

Every question is one embedding request plus one chat request to Mistral, both paid. The scoring
rules call no judge model. Writes `evaluation_results.json` under the reports directory.
"""

import json

from events_rag.evaluation.evaluate import run_evaluation

if __name__ == "__main__":
    payload = run_evaluation()
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
