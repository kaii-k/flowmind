from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def write_html_report(scored: pd.DataFrame, summary: dict[str, object], output_path: str | Path) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    top_hits = (
        scored.sort_values("anomaly_score", ascending=False)
        .head(25)
        .copy()
    )

    protocol_counts = scored["protocol"].value_counts().to_dict()
    flagged = scored.loc[scored["is_anomaly"]].copy() if "is_anomaly" in scored.columns else scored.copy()
    risk_counts = flagged["risk_level"].value_counts().to_dict()

    rows_html = "\n".join(
        _table_row(
            [
                str(row.timestamp),
                getattr(row, "dataset_source", "local"),
                row.src_ip,
                row.dst_ip,
                f"{row.protocol}/{int(row.dst_port)}",
                f"{row.anomaly_score:.3f}",
                row.risk_level,
                row.explanation,
            ]
        )
        for row in top_hits.itertuples()
    )

    summary_cards = "\n".join(
        f"""
        <div class="card">
          <div class="label">{escape(str(key).replace('_', ' ').title())}</div>
          <div class="value">{escape(str(value))}</div>
        </div>
        """
        for key, value in summary.items()
    )

    protocol_list = "".join(
        f"<li><strong>{escape(str(name))}</strong>: {count}</li>"
        for name, count in protocol_counts.items()
    )
    risk_list = "".join(
        f"<li><strong>{escape(str(name))}</strong>: {count}</li>"
        for name, count in risk_counts.items()
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FlowMind AI Report</title>
  <style>
    :root {{
      --bg: #08111f;
      --panel: #0f1b31;
      --panel-2: #13233f;
      --line: #27436d;
      --text: #ecf3ff;
      --muted: #9fb4d3;
      --accent: #4de0a8;
      --alert: #ff6b6b;
      --warn: #ffbd59;
    }}
    * {{
      box-sizing: border-box;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background:
        radial-gradient(circle at top right, rgba(77, 224, 168, 0.12), transparent 28%),
        linear-gradient(180deg, #091321, #07101b 60%);
      color: var(--text);
      padding: 32px;
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 28px;
      max-width: 760px;
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 16px;
      margin-bottom: 28px;
    }}
    .card, .panel {{
      background: linear-gradient(180deg, rgba(19, 35, 63, 0.95), rgba(12, 25, 44, 0.95));
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
    }}
    .card {{
      padding: 18px;
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .layout {{
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .panel {{
      padding: 20px;
    }}
    ul {{
      margin: 8px 0 0;
      padding-left: 18px;
      color: var(--muted);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 12px 10px;
      border-bottom: 1px solid rgba(159, 180, 211, 0.14);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .tag {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 10px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .critical, .high {{
      background: rgba(255, 107, 107, 0.14);
      color: var(--alert);
    }}
    .medium {{
      background: rgba(255, 189, 89, 0.14);
      color: var(--warn);
    }}
    .low {{
      background: rgba(77, 224, 168, 0.14);
      color: var(--accent);
    }}
    @media (max-width: 900px) {{
      body {{ padding: 18px; }}
      .layout {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <h1>FlowMind AI Threat Analysis Report</h1>
  <div class="sub">
    Offline behavioral anomaly detection over network flows. This report highlights the
    most suspicious sessions based on context-aware embeddings, baseline deviation, and
    contextual rarity.
  </div>

  <div class="grid">{summary_cards}</div>

  <div class="layout">
    <div class="panel">
      <h2>Traffic Snapshot</h2>
      <ul>{protocol_list}</ul>
    </div>
    <div class="panel">
      <h2>Risk Distribution</h2>
      <ul>{risk_list}</ul>
    </div>
  </div>

  <div class="panel">
    <h2>Top Suspicious Flows</h2>
    <table>
      <thead>
        <tr>
          <th>Timestamp</th>
          <th>Dataset</th>
          <th>Source</th>
          <th>Destination</th>
          <th>Service</th>
          <th>Score</th>
          <th>Risk</th>
          <th>Explanation</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>
"""

    destination.write_text(html, encoding="utf-8")
    return destination


def _table_row(values: list[str]) -> str:
    risk = escape(values[6])
    return (
        "<tr>"
        + f"<td>{escape(values[0])}</td>"
        + f"<td>{escape(values[1])}</td>"
        + f"<td>{escape(values[2])}</td>"
        + f"<td>{escape(values[3])}</td>"
        + f"<td>{escape(values[4])}</td>"
        + f"<td>{escape(values[5])}</td>"
        + f"<td><span class=\"tag {risk}\">{risk}</span></td>"
        + f"<td>{escape(values[7])}</td>"
        + "</tr>"
    )
