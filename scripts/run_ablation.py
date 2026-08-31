from events_rag.evaluation.report import run_ablation

if __name__ == "__main__":
    payload = run_ablation()
    for result in payload["results"]:
        print(f"{result['name']:32s} {result['error'] or result['metrics']}")
