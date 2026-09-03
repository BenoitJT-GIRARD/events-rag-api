"""Run the seven retrieval configurations over the question set and write the table.

This is what reproduces the README's ablation exactly, from the committed corpus.
"""

from events_rag.evaluation.report import run_ablation

if __name__ == "__main__":
    payload = run_ablation()
    for result in payload["results"]:
        print(f"{result['name']:32s} {result['error'] or result['metrics']}")
