#!/usr/bin/env python3
"""
create_dashboard.py

Reads:
- SYMBOL_TF_selloffs_detail.csv
- SYMBOL_TF_buyups_detail.csv
- SYMBOL_TF_chop_days_detail.csv
- SYMBOL_TF_other_days_detail.csv
- SYMBOL_TF_summary_stats.json
- SYMBOL_TF_charts.json

Writes:
- SYMBOL_TF_analysis_dashboard.html

Usage:
  python create_dashboard.py --symbol SPY --tf 1h
"""

import argparse
import json
import os


def load_json(path, default=None):
    if not os.path.exists(path):
        return default if default is not None else {}
    with open(path, "r") as f:
        return json.load(f)


def read_csv(path):
    # simple csv reader to avoid pandas dependency here
    if not os.path.exists(path):
        return []
    import csv
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def fmt(x, nd=2):
    if x is None:
        return ""
    try:
        v = float(x)
        if v != v:  # NaN
            return ""
        return f"{v:.{nd}f}"
    except Exception:
        return str(x)


def pct(x, nd=1):
    if x is None:
        return ""
    try:
        v = float(x)
        if v != v:
            return ""
        return f"{v:.{nd}f}%"
    except Exception:
        return str(x)


def make_table(rows, cols):
    # cols = [(key, label, formatter_fn)]
    head = "".join([f"<th>{label}</th>" for _, label, _ in cols])
    body = []
    for r in rows:
        tds = []
        for key, _, fn in cols:
            val = r.get(key, "")
            tds.append(f"<td>{fn(val) if fn else val}</td>")
        body.append("<tr>" + "".join(tds) + "</tr>")
    return f"""
    <table class="tbl">
      <thead><tr>{head}</tr></thead>
      <tbody>
        {''.join(body) if body else '<tr><td colspan="'+str(len(cols))+'">No rows</td></tr>'}
      </tbody>
    </table>
    """


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--tf", default="1h")
    args = ap.parse_args()

    symbol = args.symbol.upper()
    tf = args.tf

    sell_csv = f"{symbol}_{tf}_selloffs_detail.csv"
    buy_csv = f"{symbol}_{tf}_buyups_detail.csv"
    chop_csv = f"{symbol}_{tf}_chop_days_detail.csv"
    other_csv = f"{symbol}_{tf}_other_days_detail.csv"
    sum_json = f"{symbol}_{tf}_summary_stats.json"
    charts_json = f"{symbol}_{tf}_charts.json"
    out_html = f"{symbol}_{tf}_analysis_dashboard.html"

    sell_rows = read_csv(sell_csv)
    buy_rows = read_csv(buy_csv)
    chop_rows = read_csv(chop_csv)
    other_rows = read_csv(other_csv)
    summary = load_json(sum_json, default={})
    charts = load_json(charts_json, default={})

    meta = summary.get("meta", {})
    big_vol = meta.get("big_volume_threshold", "")
    move_thr = meta.get("move_threshold_pct", "")
    band = meta.get("chop_band_pct", "")
    max_mult = meta.get("chop_max_range_mult", "")

    def chart_btn(date_str):
        if not date_str:
            return ""
        # always show button; modal shows “no chart” if missing
        return f"""<button class="btn" onclick="showChart('{date_str}')">View</button>"""

    # Columns for tables
    sell_cols = [
        ("date", "Date", None),
        ("fh_move_pct", "FH Move", lambda v: pct(v, 2)),
        ("first_hour_range", "FH Range (pts)", lambda v: fmt(v, 2)),
        ("final_abs_low", "Final Abs Low", lambda v: fmt(v, 2)),
        ("hours_to_final_abs_low", "Hrs→Final Low", lambda v: fmt(v, 2)),
        ("extension_points_below_fh_low", "Ext Below FH Low", lambda v: fmt(v, 2)),
        ("continuation_hours", "Continuation (hrs)", lambda v: fmt(v, 2)),
        ("chop_hours", "Chop (hrs)", lambda v: fmt(v, 2)),
        ("bounce_25_hours", "25% Bounce (hrs)", lambda v: fmt(v, 2)),
        ("bounce_50_hours", "50% Bounce (hrs)", lambda v: fmt(v, 2)),
        ("lh_move_pct", "Last Hr %", lambda v: pct(v, 2)),
        ("date", "Chart", lambda v: chart_btn(v)),
    ]

    buy_cols = [
        ("date", "Date", None),
        ("fh_move_pct", "FH Move", lambda v: pct(v, 2)),
        ("first_hour_range", "FH Range (pts)", lambda v: fmt(v, 2)),
        ("final_abs_high", "Final Abs High", lambda v: fmt(v, 2)),
        ("hours_to_final_abs_high", "Hrs→Final High", lambda v: fmt(v, 2)),
        ("extension_points_above_fh_high", "Ext Above FH High", lambda v: fmt(v, 2)),
        ("continuation_hours", "Continuation (hrs)", lambda v: fmt(v, 2)),
        ("chop_hours", "Chop (hrs)", lambda v: fmt(v, 2)),
        ("pullback_25_hours", "25% Pullback (hrs)", lambda v: fmt(v, 2)),
        ("pullback_50_hours", "50% Pullback (hrs)", lambda v: fmt(v, 2)),
        ("lh_move_pct", "Last Hr %", lambda v: pct(v, 2)),
        ("date", "Chart", lambda v: chart_btn(v)),
    ]

    chop_cols = [
        ("date", "Date", None),
        ("fh_move_pct", "FH Move", lambda v: pct(v, 2)),
        ("fh_high", "FH High", lambda v: fmt(v, 2)),
        ("fh_low", "FH Low", lambda v: fmt(v, 2)),
        ("final_abs_high", "Final Abs High", lambda v: fmt(v, 2)),
        ("hours_to_final_abs_high", "Hrs→Final High", lambda v: fmt(v, 2)),
        ("extension_points_above_fh_high", "Ext Above FH High", lambda v: fmt(v, 2)),
        ("made_new_high_after_fh", "New High?", None),
        ("final_abs_low", "Final Abs Low", lambda v: fmt(v, 2)),
        ("hours_to_final_abs_low", "Hrs→Final Low", lambda v: fmt(v, 2)),
        ("extension_points_below_fh_low", "Ext Below FH Low", lambda v: fmt(v, 2)),
        ("made_new_low_after_fh", "New Low?", None),
        ("lh_move_pct", "Last Hr %", lambda v: pct(v, 2)),
        ("date", "Chart", lambda v: chart_btn(v)),
    ]

    other_cols = [
        ("date", "Date", None),
        ("fh_move_pct", "FH Move", lambda v: pct(v, 2)),
        ("fh_volume", "FH Vol", lambda v: fmt(v, 0)),
        ("fh_vol_vs_day", "FH Vol % of Day", lambda v: pct(float(v)*100.0, 1) if v not in (None,"") else ""),
        ("fh_high", "FH High", lambda v: fmt(v, 2)),
        ("fh_low", "FH Low", lambda v: fmt(v, 2)),
        ("final_abs_high", "Final Abs High", lambda v: fmt(v, 2)),
        ("hours_to_final_abs_high", "Hrs→Final High", lambda v: fmt(v, 2)),
        ("extension_points_above_fh_high", "Ext Above FH High", lambda v: fmt(v, 2)),
        ("made_new_high_after_fh", "New High?", None),
        ("final_abs_low", "Final Abs Low", lambda v: fmt(v, 2)),
        ("hours_to_final_abs_low", "Hrs→Final Low", lambda v: fmt(v, 2)),
        ("extension_points_below_fh_low", "Ext Below FH Low", lambda v: fmt(v, 2)),
        ("made_new_low_after_fh", "New Low?", None),
        ("lh_move_pct", "Last Hr %", lambda v: pct(v, 2)),
        ("date", "Chart", lambda v: chart_btn(v)),
    ]

    # Top cards
    s = summary.get("selloffs", {})
    b = summary.get("buyups", {})
    c = summary.get("chop_days", {})
    o = summary.get("other_days", {})

    def card(title, value, sub=""):
        return f"""
        <div class="card">
          <div class="cardTitle">{title}</div>
          <div class="cardValue">{value}</div>
          <div class="cardSub">{sub}</div>
        </div>
        """

    cards_html = f"""
    <div class="cards">
      {card("Symbol / TF", f"{symbol} / {tf}", f"BigVol q={meta.get('volume_quantile','')} | MoveThr={move_thr}%")}
      {card("BigVol Threshold (FH)", f"{fmt(big_vol,0)}", f"Chop band={band} | maxRangeMult={max_mult}")}
      {card("Sell-offs", str(s.get("total_count",0)), f"Continuation rate: {pct(s.get('continuation_rate',0)*100,1)}")}
      {card("Buy-ups", str(b.get("total_count",0)), f"Continuation rate: {pct(b.get('continuation_rate',0)*100,1)}")}
      {card("BigVol Chop Days", str(c.get("total_count",0)), f"New high rate: {pct(c.get('new_high_rate',0)*100,1)}")}
      {card("Other/Normal Days", str(o.get("total_count",0)), f"New high rate: {pct(o.get('new_high_rate',0)*100,1)}")}
    </div>
    """

    # Tabs
    tabs = [
        ("sell", "Sell-offs", make_table(sell_rows, sell_cols)),
        ("buy", "Buy-ups", make_table(buy_rows, buy_cols)),
        ("chop", "BigVol Chop/No Breakout", make_table(chop_rows, chop_cols)),
        ("other", "Other / Normal Days", make_table(other_rows, other_cols)),
    ]

    tabs_buttons = "".join([f'<button class="tabBtn" onclick="openTab(\'{tid}\')">{name}</button>' for tid, name, _ in tabs])
    tabs_content = "".join([f'<div id="tab_{tid}" class="tabPane">{html}</div>' for tid, _, html in tabs])

    # Embed charts JSON directly
    charts_js = json.dumps(charts)

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{symbol} {tf} Analysis Dashboard</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 18px; background:#0b0e14; color:#e6e6e6; }}
  h1 {{ margin: 0 0 10px 0; font-size: 20px; }}
  .cards {{ display:grid; grid-template-columns: repeat(3, minmax(200px, 1fr)); gap: 12px; margin: 12px 0 18px; }}
  .card {{ background:#121826; border:1px solid #202a3a; border-radius: 12px; padding: 12px; }}
  .cardTitle {{ font-size: 12px; opacity: .85; }}
  .cardValue {{ font-size: 22px; margin-top: 6px; }}
  .cardSub {{ font-size: 12px; opacity: .75; margin-top: 6px; }}
  .tabRow {{ display:flex; gap:10px; margin: 10px 0 10px; flex-wrap: wrap; }}
  .tabBtn {{ background:#1b2434; color:#e6e6e6; border:1px solid #2a3850; padding:8px 10px; border-radius: 10px; cursor:pointer; }}
  .tabBtn:hover {{ background:#22304a; }}
  .tabPane {{ display:none; }}
  .tabPane.active {{ display:block; }}
  .tbl {{ width:100%; border-collapse: collapse; background:#0f1420; border:1px solid #202a3a; border-radius: 10px; overflow:hidden; }}
  .tbl th, .tbl td {{ border-bottom:1px solid #202a3a; padding:8px; font-size: 12px; }}
  .tbl th {{ text-align:left; background:#121826; position: sticky; top: 0; }}
  .btn {{ background:#2b3a57; border:1px solid #3b4f73; color:#fff; border-radius:10px; padding:6px 10px; cursor:pointer; }}
  .btn:hover {{ background:#334466; }}
  .modal {{ display:none; position:fixed; left:0; top:0; width:100%; height:100%; background:rgba(0,0,0,.65); }}
  .modalInner {{ width: min(1100px, 95vw); margin: 4vh auto; background:#0f1420; border:1px solid #202a3a; border-radius: 14px; padding: 12px; }}
  .modalTop {{ display:flex; justify-content: space-between; align-items:center; gap: 10px; }}
  .closeX {{ cursor:pointer; padding:6px 10px; border-radius:10px; background:#1b2434; border:1px solid #2a3850; }}
  img {{ max-width:100%; border-radius: 10px; border:1px solid #202a3a; }}
  .note {{ opacity:.8; font-size: 12px; margin-top: 6px; }}
</style>
</head>
<body>
  <h1>{symbol} First Hour Analysis Dashboard ({tf})</h1>
  {cards_html}

  <div class="tabRow">
    {tabs_buttons}
  </div>

  {tabs_content}

  <div id="chartModal" class="modal" onclick="hideModal(event)">
    <div class="modalInner">
      <div class="modalTop">
        <div id="chartTitle" style="font-weight:700;"></div>
        <div class="closeX" onclick="closeChart()">Close</div>
      </div>
      <div style="margin-top:10px;">
        <img id="chartImg" src="" alt="chart"/>
        <div id="chartNote" class="note"></div>
      </div>
    </div>
  </div>

<script>
  const CHARTS = {charts_js};

  function openTab(id) {{
    document.querySelectorAll('.tabPane').forEach(x => x.classList.remove('active'));
    const el = document.getElementById('tab_' + id);
    if (el) el.classList.add('active');
  }}

  // default tab
  openTab('sell');

  function showChart(dateStr) {{
    const b64 = CHARTS[dateStr];
    document.getElementById('chartTitle').innerText = "RTH Candles: " + dateStr;
    if (!b64) {{
      document.getElementById('chartImg').src = "";
      document.getElementById('chartNote').innerText = "No chart stored for this date (missing bars or chart generation disabled).";
    }} else {{
      document.getElementById('chartImg').src = "data:image/png;base64," + b64;
      document.getElementById('chartNote').innerText = "";
    }}
    document.getElementById('chartModal').style.display = 'block';
  }}

  function closeChart() {{
    document.getElementById('chartModal').style.display = 'none';
  }}

  function hideModal(ev) {{
    if (ev.target.id === 'chartModal') closeChart();
  }}
</script>
</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Wrote {out_html}")


if __name__ == "__main__":
    main()
