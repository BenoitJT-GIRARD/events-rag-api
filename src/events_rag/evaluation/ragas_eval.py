import json
from pathlib import Path

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from events_rag.config import get_settings
from events_rag.rag.retriever import retrieve_documents
from events_rag.rag.service import answer_question, build_chat_model
from events_rag.rag.retriever import build_embeddings


def load_reference_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    return payload


def build_ragas_dataset(cases: list[dict]) -> Dataset:
    rows = []

    for case in cases:
        question = case["question"]
        reference = case["reference_answer"]

        retrieved_docs = retrieve_documents(question=question, top_k=5)
        contexts = [doc.page_content for doc in retrieved_docs]

        result = answer_question(question=question, top_k=5)
        answer = result["answer"]

        rows.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "reference": reference,
            }
        )

    return Dataset.from_list(rows)


def run_ragas_evaluation() -> dict:
    settings = get_settings()
    input_path = settings.eval_data_dir / "reference_qa.json"
    output_path = settings.eval_data_dir / "ragas_results.json"

    cases = load_reference_dataset(input_path)
    dataset = build_ragas_dataset(cases)

    evaluator_llm = build_chat_model()
    evaluator_embeddings = build_embeddings()

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    payload = result.to_pandas().to_dict(orient="records")

    summary = {
        "count": len(payload),
        "avg_faithfulness": round(
            sum(item.get("faithfulness", 0) for item in payload) / len(payload), 3
        )
        if payload
        else 0.0,
        "avg_answer_relevancy": round(
            sum(item.get("answer_relevancy", 0) for item in payload) / len(payload), 3
        )
        if payload
        else 0.0,
        "avg_context_precision": round(
            sum(item.get("context_precision", 0) for item in payload) / len(payload), 3
        )
        if payload
        else 0.0,
    }

    final_payload = {
        "summary": summary,
        "results": payload,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(final_payload, file, ensure_ascii=False, indent=2)

    return final_payload