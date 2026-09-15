"""Draw the ablation results as an SVG bar chart.

Hand-rolled SVG and no plotting library: one chart does not justify a dependency in a project
whose point is to stay installable, and the output has to survive being read on a repository
page with no runtime behind it.

The colours come from :mod:`events_rag.figure_style`, the palette shared by every figure of
the same module the figures use, so this chart and any future matplotlib figure agree on
what a series looks
like. Nothing here writes a colour of its own.

The chart shows what the numbers can carry: a graduated axis so a bar can be read against a
scale, the sample size in the subtitle, and on every proportion the standard error it has at
that sample size. At n = 20 that error is about 11 points at p = 0.5, which is the point of
drawing it — two bars whose whiskers overlap are not two different results.
"""

from math import sqrt

from events_rag.figure_style import PALETTE

ROW_HEIGHT = 46
TOP = 78
LEFT = 210
BAR_WIDTH = 380
WIDTH = LEFT + BAR_WIDTH + 96

#: Where the x-axis is graduated. A bar with no scale behind it is a length, not a measure.
TICKS = (0.0, 0.25, 0.5, 0.75, 1.0)

STYLE = f"""
  .title {{ font: 600 15px system-ui, sans-serif; fill: {PALETTE["ink"]} }}
  .label {{ font: 13px system-ui, sans-serif; fill: {PALETTE["ink"]} }}
  .value {{ font: 12px system-ui, sans-serif; fill: {PALETTE["muted"]} }}
  .muted {{ font: italic 12px system-ui, sans-serif; fill: {PALETTE["muted"]} }}
  .axis  {{ stroke: {PALETTE["grid"]}; stroke-width: 1 }}
  .tick  {{ font: 11px system-ui, sans-serif; fill: {PALETTE["muted"]} }}
  .error {{ stroke: {PALETTE["ink"]}; stroke-width: 1.2; opacity: 0.55 }}
  .bar-recall, .swatch-recall {{ fill: {PALETTE["primary"]} }}
  .bar-mrr, .swatch-mrr {{ fill: {PALETTE["tertiary"]} }}
"""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def standard_error(proportion: float, sample: int) -> float:
    """The standard error of a proportion measured on `sample` questions.

    Binomial, because every question either retrieves its event or does not. It is drawn
    rather than written in the text so that the reader sees the width of what they compare.
    """
    if sample <= 0:
        return 0.0
    proportion = min(max(proportion, 0.0), 1.0)
    return sqrt(proportion * (1 - proportion) / sample)


def _whisker(x_centre: float, y: float, half_width: float) -> str:
    left, right = x_centre - half_width, x_centre + half_width
    return (
        f'<line class="error" x1="{left:.1f}" y1="{y:.1f}" x2="{right:.1f}" y2="{y:.1f}"/>'
        f'<line class="error" x1="{left:.1f}" y1="{y - 3:.1f}" x2="{left:.1f}" y2="{y + 3:.1f}"/>'
        f'<line class="error" x1="{right:.1f}" y1="{y - 3:.1f}" x2="{right:.1f}" y2="{y + 3:.1f}"/>'
    )


def render_ablation_svg(results: list[dict], withdrawn: set[str] | None = None) -> str:
    """The chart, from the results file and the list of rows an erratum has withdrawn.

    A withdrawn row keeps its line and loses its bar: removing it silently would leave the
    reader counting six configurations where seven were run.
    """
    withdrawn = withdrawn or set()
    height = TOP + ROW_HEIGHT * len(results) + 38
    # The sample size is read off the data. A subtitle that keeps claiming twenty questions
    # after the set has changed says something the figure no longer shows.
    sample = next((r["metrics"]["n"] for r in results if r["metrics"] and "n" in r["metrics"]), 0)
    baseline = height - 30

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" '
        f'aria-label="Retrieval ablation: recall@1 and MRR@10 for every configuration, '
        f'with standard error at n = {sample}">',
        f"<style>{STYLE}</style>",
        f'<rect width="{WIDTH}" height="{height}" fill="{PALETTE["paper"]}"/>',
        '<text x="16" y="26" class="title">Retrieval ablation — recall@1 and MRR@10</text>',
        '<text x="16" y="45" class="value">Proportion of questions whose source event is '
        "retrieved. Higher is better.</text>",
        f'<text x="16" y="62" class="value">n = {sample} questions with a known target event; '
        f"whiskers are ±1 standard error.</text>",
    ]

    for tick in TICKS:
        x = LEFT + tick * BAR_WIDTH
        parts.append(
            f'<line class="axis" x1="{x:.1f}" y1="{TOP - 8}" x2="{x:.1f}" y2="{baseline}"/>'
        )
        parts.append(
            f'<text class="tick" x="{x:.1f}" y="{baseline + 14}" text-anchor="middle">'
            f"{tick:.2f}</text>"
        )

    for row, result in enumerate(results):
        y = TOP + row * ROW_HEIGHT
        parts.append(f'<text x="16" y="{y + 14}" class="label">{_escape(result["name"])}</text>')

        if result["name"] in withdrawn:
            parts.append(
                f'<text x="{LEFT + 8}" y="{y + 16}" class="muted">withdrawn — see '
                "reports/errata.json</text>"
            )
            continue
        if not result["metrics"]:
            parts.append(f'<text x="{LEFT + 8}" y="{y + 16}" class="muted">not measured</text>')
            continue

        recall = result["metrics"]["recall@1"] or 0.0
        mrr = result["metrics"]["mrr@10"] or 0.0
        error = standard_error(recall, sample) * BAR_WIDTH
        parts.append(
            f'<rect class="bar-recall" x="{LEFT}" y="{y}" '
            f'width="{recall * BAR_WIDTH:.1f}" height="13" rx="2"/>'
        )
        parts.append(_whisker(LEFT + recall * BAR_WIDTH, y + 6.5, error))
        parts.append(
            f'<rect class="bar-mrr" x="{LEFT}" y="{y + 17}" '
            f'width="{mrr * BAR_WIDTH:.1f}" height="13" rx="2"/>'
        )
        parts.append(
            f'<text x="{LEFT + BAR_WIDTH + 10}" y="{y + 21}" class="value">'
            f"{recall:.2f} / {mrr:.2f}</text>"
        )

    parts.append(
        f'<rect class="swatch-recall" x="16" y="{height - 16}" width="12" height="11" rx="2"/>'
        f'<text x="34" y="{height - 6}" class="value">recall@1</text>'
        f'<rect class="swatch-mrr" x="110" y="{height - 16}" width="12" height="11" rx="2"/>'
        f'<text x="128" y="{height - 6}" class="value">MRR@10</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
