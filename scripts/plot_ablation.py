import json
from pathlib import Path

from events_rag.config import get_settings
from events_rag.evaluation.chart import render_ablation_svg

if __name__ == "__main__":
    settings = get_settings()
    payload = json.loads(
        (Path(settings.eval_data_dir) / "ablation_results.json").read_text(encoding="utf-8")
    )
    output = Path("docs/images/ablation.svg")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_ablation_svg(payload["results"]), encoding="utf-8")
    print(f"wrote {output}")
