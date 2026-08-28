import json

from events_rag.evaluation.generate_dataset import generate_eval_dataset, summarize_cases

if __name__ == "__main__":
    cases = generate_eval_dataset()
    print(json.dumps(summarize_cases(cases), ensure_ascii=False, indent=2))
