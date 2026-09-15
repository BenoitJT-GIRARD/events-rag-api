"""Compare each retrieval configuration with the shipped one, question by question.

The seven configurations answer the same twenty questions, so the comparison is paired, and
the interval of a single proportion is the wrong yardstick: it ignores that the two sides
agree on most questions. What separates two configurations is the discordant pairs, the
questions one gets right and the other gets wrong, and McNemar's exact test reads exactly
those.

Reads `reports/ablation_results.json`, which carries the rank of the target event for every
question and every configuration. No model call, no index: this is arithmetic on a committed
artefact.

    uv run python scripts/compare_paired.py
"""

from __future__ import annotations

import json

from events_rag.evaluation.retrieval import mcnemar_exact
from events_rag.utils.paths import REPORTS_DIR

ABLATION = REPORTS_DIR / "ablation_results.json"
OUTPUT = REPORTS_DIR / "paired_comparison.json"

#: The configuration everything is compared with: the one the service actually runs.
REFERENCE = "dense-baseline"

#: The depths the README publishes. `recall@1` is what a reader sees first, `recall@5` is
#: what the answer is built from.
DEPTHS = (1, 5)


def hits(per_question: list[dict], k: int) -> list[bool]:
    """Whether the target event was retrieved within the first k results, question by question."""
    return [entry["rank"] is not None and entry["rank"] <= k for entry in per_question]


def compare(results: list[dict]) -> dict:
    by_name = {r["name"]: r for r in results if r.get("per_question")}
    if REFERENCE not in by_name:
        raise SystemExit(
            f"{ABLATION.name} carries no per-question ranks for {REFERENCE}: rerun "
            "scripts/run_ablation.py, which records them"
        )

    reference = by_name[REFERENCE]
    comparisons = []
    for name, result in by_name.items():
        if name == REFERENCE:
            continue
        for k in DEPTHS:
            test = mcnemar_exact(
                hits(result["per_question"], k), hits(reference["per_question"], k)
            )
            comparisons.append(
                {
                    "configuration": name,
                    "against": REFERENCE,
                    "metric": f"recall@{k}",
                    "won": test["only_first"],
                    "lost": test["only_second"],
                    "discordant": test["discordant"],
                    "p_value": test["p_value"],
                    "decided_at_5_percent": test["p_value"] < 0.05,
                }
            )
    return {
        "n_questions": len(reference["per_question"]),
        "reference": REFERENCE,
        "test": "McNemar, exact binomial on the discordant pairs, two-sided",
        "comparisons": comparisons,
    }


def main() -> int:
    payload = json.loads(ABLATION.read_text(encoding="utf-8"))
    comparison = compare(payload["results"])
    OUTPUT.write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline=""
    )
    decided = sum(1 for c in comparison["comparisons"] if c["decided_at_5_percent"])
    print(
        f"{OUTPUT.name}: {len(comparison['comparisons'])} comparison(s) against "
        f"{REFERENCE}, {decided} decided at 5 %"
    )
    for c in comparison["comparisons"]:
        print(
            f"  {c['configuration']:28} {c['metric']:9} "
            f"won {c['won']}, lost {c['lost']}, p = {c['p_value']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
