from __future__ import annotations

from html import escape
from pathlib import Path
import unittest.result as result

import pandas as pd


# ── Traffic class colour palette ──────────────────────────────────────────────
_CLASS_STYLE: dict[str, tuple[str, str]] = {
    "BENIGN":          ("#00d4aa", "rgba(0,212,170,0.12)"),
    "DoS/DDoS":        ("#ff4757", "rgba(255,71,87,0.12)"),
    "Port Scan":       ("#ffa502", "rgba(255,165,2,0.12)"),
    "Brute Force":     ("#ff6b81", "rgba(255,107,129,0.12)"),
    "Web Attack":      ("#eccc68", "rgba(236,204,104,0.12)"),
    "Botnet":          ("#a29bfe", "rgba(162,155,254,0.12)"),
    "Infiltration":    ("#fd79a8", "rgba(253,121,168,0.12)"),
    "Unknown Anomaly": ("#636e72", "rgba(99,110,114,0.12)"),
}


def write_html_report(
    scored: pd.DataFrame,
    summary: dict[str, object],
    output_path: str | Path,
) -> Path:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # ── Deduplicate top hits: keep the highest-scoring unique flow signature ──
    dedup_cols = ["src_ip", "dst_ip", "dst_port", "protocol"]
    available_dedup = [c for c in dedup_cols if c in scored.columns]
    dedup_cols = [c for c in ["src_ip", "dst_ip", "dst_port", "protocol"] if c in scored.columns]
    top_hits = (
    scored
    .sort_values("anomaly_score", ascending=False)
    .drop_duplicates(subset=dedup_cols if dedup_cols else None)
    .head(5)
)

    protocol_counts = scored["protocol"].value_counts().to_dict()
    flagged = scored.loc[scored["is_anomaly"]].copy() if "is_anomaly" in scored.columns else scored.copy()
    risk_counts = flagged["risk_level"].value_counts().to_dict()

    has_class = "traffic_class" in scored.columns

    # ── Summary cards (skip nested dicts like class_breakdown) ───────────────
    summary_cards = "\n".join(
        f"""
        <div class="card">
          <div class="label">{escape(str(key).replace('_', ' ').title())}</div>
          <div class="value">{escape(str(value))}</div>
        </div>
        """
        for key, value in summary.items()
        if not isinstance(value, dict)           # skip class_breakdown dict
    )

    # ── Table rows ────────────────────────────────────────────────────────────
    rows_html = "\n".join(
        _table_row(row, has_class)
        for row in top_hits.itertuples()
    )

    # ── Protocol / risk panel lists ───────────────────────────────────────────
    protocol_list = "".join(
        f"<li><strong>{escape(str(name))}</strong>: {count:,}</li>"
        for name, count in protocol_counts.items()
    )
    risk_list = "".join(
        f"<li><strong>{escape(str(name))}</strong>: {count:,}</li>"
        for name, count in risk_counts.items()
    )

    # ── Class breakdown panel (only when classification was run) ──────────────
    class_breakdown_html = ""
    if has_class:
        class_counts = scored["traffic_class"].value_counts()
        total = len(scored)
        rows_cls = ""
        for cls, cnt in class_counts.items():
            pct = cnt / total * 100 if total else 0
            color, bg = _CLASS_STYLE.get(str(cls), ("#ecf3ff", "rgba(255,255,255,0.08)"))
            bar_w = max(4, int(pct * 2.8))   # max bar ~280 px (100 %)
            rows_cls += f"""
            <tr>
              <td><span class="cls-tag" style="color:{color};background:{bg}">{escape(str(cls))}</span></td>
              <td style="text-align:right;font-variant-numeric:tabular-nums">{cnt:,}</td>
              <td style="width:300px">
                <div style="background:{color};height:10px;border-radius:5px;width:{bar_w}px;opacity:0.85"></div>
              </td>
              <td style="color:var(--muted)">{pct:.1f}%</td>
            </tr>"""
        class_breakdown_html = f"""
  <div class="panel" style="margin-bottom:24px">
    <h2>Traffic Class Breakdown
      <span style="font-size:13px;font-weight:400;color:var(--muted);margin-left:10px">
        Context-Aware Flow Embeddings → Multi-Class Classification
      </span>
    </h2>
    <table style="font-size:14px">
      <thead>
        <tr>
          <th>Traffic Class</th>
          <th style="text-align:right">Flows</th>
          <th>Distribution</th>
          <th>%</th>
        </tr>
      </thead>
      <tbody>{rows_cls}</tbody>
    </table>
  </div>"""

    # ── Table header (conditionally add Class column) ─────────────────────────
    class_th = "<th>Class</th>" if has_class else ""

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
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Arial, sans-serif;
      background:
        radial-gradient(circle at top right, rgba(77, 224, 168, 0.12), transparent 28%),
        linear-gradient(180deg, #091321, #07101b 60%);
      color: var(--text);
      padding: 32px;
    }}
    h1, h2 {{ margin: 0 0 12px; }}
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
    .card {{ padding: 18px; }}
    .label {{
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .value {{ font-size: 28px; font-weight: 700; }}
    .layout {{
      display: grid;
      grid-template-columns: 1.5fr 1fr;
      gap: 20px;
      margin-bottom: 24px;
    }}
    .panel {{ padding: 20px; }}
    ul {{ margin: 8px 0 0; padding-left: 18px; color: var(--muted); }}
    table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
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
    .cls-tag {{
      display: inline-block;
      border-radius: 6px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 700;
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
  <h1>FlowMind AI — Traffic Classification &amp; Threat Analysis</h1>
  <div class="sub">
    Context-aware flow embeddings → adaptive anomaly detection → multi-class traffic
    classification. Each flagged flow is assigned a named attack class based on its
    20-dimensional embedding vector and rule-based signature matching.
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

  {class_breakdown_html}

  <div class="panel">
    <h2>Top Suspicious Flows
      <span style="font-size:13px;font-weight:400;color:var(--muted);margin-left:10px">
        deduplicated by (src → dst:port/protocol)
      </span>
    </h2>
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
          {class_th}
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


def _table_row(row: object, has_class: bool) -> str:
    risk = escape(str(getattr(row, "risk_level", "")))
    cls  = str(getattr(row, "traffic_class", "")) if has_class else ""

    color, bg = _CLASS_STYLE.get(cls, ("#ecf3ff", "rgba(255,255,255,0.08)"))
    cls_td = (
        f'<td><span class="cls-tag" style="color:{color};background:{bg}">'
        f"{escape(cls)}</span></td>"
        if has_class
        else ""
    )

    return (
        "<tr>"
        + f"<td>{escape(str(getattr(row, 'timestamp', '')))}</td>"
        + f"<td>{escape(str(getattr(row, 'dataset_source', 'local')))}</td>"
        + f"<td>{escape(str(getattr(row, 'src_ip', '')))}</td>"
        + f"<td>{escape(str(getattr(row, 'dst_ip', '')))}</td>"
        + f"<td>{escape(str(getattr(row, 'protocol', '')))}/{escape(str(int(getattr(row, 'dst_port', 0))))}</td>"
        + "<td>" + escape(f"{float(getattr(row, 'anomaly_score', 0)):.3f}") + "</td>"
        + f'<td><span class="tag {risk}">{risk}</span></td>'
        + cls_td
        + f"<td>{escape(str(getattr(row, 'explanation', '')))}</td>"
        + "</tr>"
    )