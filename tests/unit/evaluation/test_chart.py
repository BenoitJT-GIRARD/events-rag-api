"""The ablation figure is drawn from the results file and hides nothing.

Well-formed SVG, one bar per configuration, a longer bar for a higher score, and a failed
configuration drawn as failed rather than omitted — the same guarantee the table gives,
because a reader looks at the figure first.
"""

import re

from events_rag.evaluation.chart import render_ablation_svg


def _result(name: str, recall1: float | None = 0.9, error: str | None = None) -> dict:
    metrics = None if error else {"recall@1": recall1, "recall@5": 1.0, "mrr@10": 0.95}
    return {"name": name, "metrics": metrics, "error": error}


def test_svg_is_well_formed_and_names_every_configuration() -> None:
    svg = render_ablation_svg([_result("bm25-only", 0.5), _result("dense-baseline", 0.9)])

    assert svg.startswith("<svg")
    assert svg.rstrip().endswith("</svg>")
    assert "bm25-only" in svg
    assert "dense-baseline" in svg


def test_a_higher_score_draws_a_longer_bar() -> None:
    svg = render_ablation_svg([_result("low", 0.25), _result("high", 1.0)])

    # Matched on the class alone, not on attribute order. Legend swatches carry their
    # own class so they never land in this list.
    widths = [float(w) for w in re.findall(r'class="bar-recall"[^>]*?width="([\d.]+)"', svg)]

    assert len(widths) == 2
    assert widths[1] > widths[0] * 3


def test_a_failed_configuration_is_drawn_as_such_instead_of_vanishing() -> None:
    svg = render_ablation_svg([_result("broken", error="RuntimeError: boom")])

    assert "broken" in svg
    assert "not measured" in svg


def test_height_grows_with_the_number_of_configurations() -> None:
    short = render_ablation_svg([_result("a")])
    tall = render_ablation_svg([_result("a"), _result("b"), _result("c")])

    assert len(tall) > len(short)
    assert 'height="' in tall
