"""Draw the ablation results as an SVG bar chart.

Hand-rolled SVG rather than a plotting library: one chart does not justify pulling a
dependency into a project whose point is to stay installable, and the output has to
survive being read on a repository page with no runtime.
"""

ROW_HEIGHT = 44
TOP = 56
LEFT = 210
BAR_WIDTH = 380
WIDTH = LEFT + BAR_WIDTH + 90

STYLE = """
  .title { font: 600 15px system-ui, sans-serif; fill: #1f2328 }
  .label { font: 13px system-ui, sans-serif; fill: #1f2328 }
  .value { font: 12px system-ui, sans-serif; fill: #57606a }
  .muted { font: italic 12px system-ui, sans-serif; fill: #8c959f }
  .axis  { stroke: #d0d7de; stroke-width: 1 }
  .bar-recall, .swatch-recall { fill: #2c6e9b }
  .bar-mrr, .swatch-mrr { fill: #9dc3d9 }
"""


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_ablation_svg(results: list[dict]) -> str:
    height = TOP + ROW_HEIGHT * len(results) + 34
    # Read the sample size off the data rather than writing it in: a subtitle that keeps
    # claiming twenty questions after the set changes is worse than no subtitle.
    sample = next((r["metrics"]["n"] for r in results if r["metrics"] and "n" in r["metrics"]), 0)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" '
        f'aria-label="Retrieval ablation results">',
        f"<style>{STYLE}</style>",
        f'<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>',
        '<text x="16" y="26" class="title">Retrieval ablation — recall@1 and MRR@10</text>',
        f'<text x="16" y="44" class="value">Higher is better. {sample} questions with a '
        f"known target event.</text>",
        f'<line x1="{LEFT}" y1="{TOP - 6}" x2="{LEFT}" y2="{height - 26}" class="axis"/>',
    ]

    for row, result in enumerate(results):
        y = TOP + row * ROW_HEIGHT
        parts.append(f'<text x="16" y="{y + 14}" class="label">{_escape(result["name"])}</text>')

        if not result["metrics"]:
            parts.append(f'<text x="{LEFT + 8}" y="{y + 14}" class="muted">not measured</text>')
            continue

        recall = result["metrics"]["recall@1"] or 0.0
        mrr = result["metrics"]["mrr@10"] or 0.0
        parts.append(
            f'<rect class="bar-recall" x="{LEFT}" y="{y}" '
            f'width="{recall * BAR_WIDTH:.1f}" height="13" rx="2"/>'
        )
        parts.append(
            f'<rect class="bar-mrr" x="{LEFT}" y="{y + 16}" '
            f'width="{mrr * BAR_WIDTH:.1f}" height="13" rx="2"/>'
        )
        parts.append(
            f'<text x="{LEFT + BAR_WIDTH + 10}" y="{y + 20}" class="value">'
            f"{recall:.2f} / {mrr:.2f}</text>"
        )

    parts.append(
        f'<rect class="swatch-recall" x="16" y="{height - 22}" width="12" height="11" rx="2"/>'
        f'<text x="34" y="{height - 12}" class="value">recall@1</text>'
        f'<rect class="swatch-mrr" x="110" y="{height - 22}" width="12" height="11" rx="2"/>'
        f'<text x="128" y="{height - 12}" class="value">MRR@10</text>'
    )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"
