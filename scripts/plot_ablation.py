"""Redraw reports/figures/ablation.svg from the published ablation results.

Reads the run's own file and the errata beside it, so a row a later analysis withdrew is
drawn as withdrawn instead of disappearing. Writes through the shared figure writer, which
records the image, its sample size and its axes in reports/figures/MANIFEST.json.
"""

import json
from pathlib import Path

from events_rag.config import get_settings
from events_rag.evaluation.chart import render_ablation_svg
from events_rag.figure_style import save_svg


def withdrawn_rows(reports_dir: Path, artefact: str) -> set[str]:
    errata = reports_dir / "errata.json"
    if not errata.is_file():
        return set()
    entries = json.loads(errata.read_text(encoding="utf-8")).get("entries", [])
    return {e["row"] for e in entries if e.get("artefact") == artefact and e.get("row")}


def main() -> int:
    settings = get_settings()
    reports = Path(settings.reports_dir)
    payload = json.loads((reports / "ablation_results.json").read_text(encoding="utf-8"))
    results = payload["results"]
    withdrawn = withdrawn_rows(reports, "reports/ablation_results.json")

    sample = next((r["metrics"]["n"] for r in results if r["metrics"] and "n" in r["metrics"]), 0)
    output = save_svg(
        render_ablation_svg(results, withdrawn=withdrawn),
        reports / "figures" / "ablation.svg",
        n=sample,
        axes={
            "x": "Proportion of questions (0 to 1)",
            "y": "Retrieval configuration",
            "title": "Retrieval ablation — recall@1 and MRR@10",
        },
        dispersion="±1 standard error of a proportion at this sample size",
        source="scripts/plot_ablation.py",
    )
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
