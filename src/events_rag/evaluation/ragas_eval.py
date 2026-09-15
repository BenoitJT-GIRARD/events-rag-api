"""The generation-side metrics, which are the weaker half and are reported as such.

Two of the five RAGAS metrics return null on this setup; they are reported as not measured
rather than shown as zero, because a null that looks like a score is worse than an admitted
gap. `NaNSafeEncoder` exists so that a missing metric survives serialisation as `null`
instead of as the string `NaN`, which no JSON reader accepts.

The reference answers are derived from the events themselves, so these numbers are
optimistic by construction. The README says so.
"""

import json
import math
import time
from pathlib import Path

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper

# These names are underscore-prefixed, and that is not an oversight to clean up: as of
# ragas 0.4.3 there is no public path to the concrete metrics at all. `ragas.metrics.__all__`
# is empty of them and every metric module is itself underscore-prefixed. The classes are
# re-exported at package level, which is the closest thing to a supported entry point.
#
# The protection is therefore the version cap in pyproject.toml (`ragas>=0.4.3,<0.5`), not
# the import style. Without it an unrelated `uv sync` would pull a release where this
# surface has moved and break the evaluation with no warning.
from ragas.metrics import (  # noqa: PLC2701
    _AnswerRelevancy as AnswerRelevancy,
)
from ragas.metrics import (  # noqa: PLC2701
    _ContextPrecision as ContextPrecision,
)
from ragas.metrics import (  # noqa: PLC2701
    _ContextRecall as ContextRecall,
)
from ragas.metrics import (  # noqa: PLC2701
    _ContextRelevance as ContextRelevancy,
)
from ragas.metrics import (  # noqa: PLC2701
    _Faithfulness as Faithfulness,
)
from ragas.run_config import RunConfig

from events_rag.config import get_settings
from events_rag.rag.prompts import SYSTEM_PROMPT, build_user_prompt
from events_rag.rag.retriever import build_embeddings, retrieve_documents
from events_rag.rag.service import build_chat_model, format_context


def safe_mean(values: list) -> float | None:
    valid = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not valid:
        return None
    return round(sum(valid) / len(valid), 3)


class NaNSafeEncoder(json.JSONEncoder):
    def iterencode(self, o, _one_shot=False):
        return super().iterencode(self._sanitize(o), _one_shot)

    def _sanitize(self, obj):
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize(v) for v in obj]
        return obj


def build_case_type_summary(cases: list[dict]) -> dict:
    type_counts: dict[str, int] = {}
    for case in cases:
        ct = case.get("case_type", "positive")
        type_counts[ct] = type_counts.get(ct, 0) + 1
    return dict(type_counts)


def load_reference_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Evaluation dataset must be a JSON list.")

    return payload


def build_ragas_dataset(cases: list[dict]) -> EvaluationDataset:
    samples = []

    for case in cases:
        question = case["question"]
        reference = case.get("ground_truth") or case.get("reference_answer", "")

        documents = retrieve_documents(question=question, top_k=5)
        contexts = [doc.page_content for doc in documents]

        context_text = format_context(documents)
        user_prompt = build_user_prompt(question=question, context=context_text)
        chat_model = build_chat_model()
        for attempt in range(3):
            try:
                response = chat_model.invoke([("system", SYSTEM_PROMPT), ("human", user_prompt)])
                break
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 10 * (attempt + 1)
                print(f"  LLM call failed ({exc}), retrying in {wait}s...")
                time.sleep(wait)
        answer = response.content if isinstance(response.content, str) else str(response.content)

        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=reference,
            )
        )

    return EvaluationDataset(samples=samples)


def run_ragas_evaluation() -> dict:
    settings = get_settings()
    input_path = settings.questions_dir / "reference_qa.json"
    output_path = settings.reports_dir / "ragas_results.json"

    cases = load_reference_dataset(input_path)
    dataset = build_ragas_dataset(cases)

    wrapped_llm = LangchainLLMWrapper(build_chat_model())
    wrapped_embeddings = LangchainEmbeddingsWrapper(build_embeddings())

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
        ContextRelevancy(),
    ]
    for metric in metrics:
        metric.llm = wrapped_llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = wrapped_embeddings

    run_config = RunConfig(max_workers=1, max_retries=5, max_wait=60, timeout=120)
    result = evaluate(dataset=dataset, metrics=metrics, run_config=run_config)

    payload = result.to_pandas().to_dict(orient="records")

    summary = {
        "count": len(payload),
        "avg_faithfulness": safe_mean([r.get("faithfulness") for r in payload]),
        "avg_answer_relevancy": safe_mean([r.get("answer_relevancy") for r in payload]),
        "avg_context_precision": safe_mean([r.get("context_precision") for r in payload]),
        "avg_context_recall": safe_mean([r.get("context_recall") for r in payload]),
        "avg_context_relevancy": safe_mean([r.get("context_relevancy") for r in payload]),
        "by_case_type": build_case_type_summary(cases),
    }

    final_payload = {
        "summary": summary,
        "results": payload,
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(final_payload, file, cls=NaNSafeEncoder, ensure_ascii=False, indent=2)

    return final_payload
