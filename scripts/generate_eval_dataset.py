"""Regenerate the question set from scratch.

Outside the reproduction path, for the same reason as `build_dataset.py`. It is also what
produced the first, broken benchmark: questions generated from the event text quoted the
titles verbatim, and the set scored a perfect `recall@1` on exact string matching. The
twenty positive questions in use were rewritten by hand.
"""

import json

from events_rag.evaluation.generate_dataset import generate_eval_dataset, summarize_cases

if __name__ == "__main__":
    cases = generate_eval_dataset()
    print(json.dumps(summarize_cases(cases), ensure_ascii=False, indent=2))
