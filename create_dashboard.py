#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd


def fmt_num(x, nd=2, pct=False):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "—"
    try:
        v = float(x)
        if pct:
            return f"{v*100:.1f}%"
        if nd == 0:
            return f"{v:,.0f}"
        return f"{v:,.{nd}f}"
    except Exception:
        return str(x)


def card(title, value, subtitle=""):
    return f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value">{value}</div>
      <div class="card-sub">{subtitle}</div>
    </div>
    """


def df_to_html(df, cols, add_chart_col=True):
    if df is None or df.empty:
        return "<div class='muted'>No rows</div>"

    d = df.copy()
    for c in cols:
        if c not in d.columns:
            d[c] = None

    d = d[cols].copy()

    if add_chart_col and "date" in d.columns:
        d.insert(
            0,
            "Chart",
            d["date"].apply(lambda x: f"<a href='#' class='chartlink' data-date='{x}'>View</a>")
        )

    return d.to_html(index=False, classes="table", border=0, escape=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--tf", default="1h")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    tf = args.tf

    sell_path = f"{symbol}_{tf}_selloffs_detail.csv"
    buy_path  = f"{symbol}_{tf}_buyups_detail.csv"
    chop_path = f"{symbol}_{tf}_chop_days_detail.csv"
    sum_path  = f"{symbol}_{tf}_summary_stats.json"
    chart_path = f"{symbol}_{tf}_charts.json"
    out_html  = f"{symbol}_{tf}_analysis_dashboard.html"

    sell_df = pd.read_csv(sell_path) if os.path.exists(sell_path) else pd.DataFrame()
    buy_df  = pd.read_csv(buy_path)  if os.path.exists(buy_path)  else pd.DataFrame()
    chop_df = pd.read_csv(chop_path) if os.path.exists(chop_path) else pd.DataFrame()

    if not os.path.exists(sum_path):
        raise FileNotFoundError(f"Missing summary json: {sum_path}")

    with open(sum_path, "r") as f:
        summary = json.load(f)

    charts = {}
    if os.path.exists(chart_path):
        with open(chart_path, "r") as f:
            charts = json.load(f)

    meta = summary.get("meta", {})
    s = summary.get("selloffs", {}) or {}
    b = summary.get("buyups", {}) or {}
    c = summary.get("chop_days", {}) or {}

    # -------------------- TOP CARDS --------------------
    cards = []

    cards.append(card("Symbol", symbol, f"TF: {tf} | Bar hours: {meta.get('bar_hours','—')}"))
    cards.append(card("Lookback Days", fmt_num(meta.get("days"), 0), "Yahoo may cap intraday"))
    cards.append(card("Big Vol Threshold", fmt_num(meta.get("big_volume_threshold"), 0), f"Quantile: {meta.get('volume_quantile','—')}"))
    cards.append(card("FH Move Threshold", f"±{fmt_num(meta.get('move_threshold_pct'),2)}%", "Sell/Buy classification"))

    cards.append(card("Total Sell-Offs", fmt_num(s.get("total_count"),0), f"Continuation: {fmt_num(s.get('continuation_rate'),0,pct=True)}"))
    cards.append(card("Avg Sell-Off Size", fmt_num(s.get("avg_first_hour_range"),2), "Points (FH drop)"))
    cards.append(card("Total Buy-Ups", fmt_num(b.get("total_count"),0), f"Continuation: {fmt_num(b.get('continuation_rate'),0,pct=True)}"))
    cards.append(card("Avg Buy-Up Size", fmt_num(b.get("avg_first_hour_range"),2), "Points (FH rally)"))

    cards.append(card("Chop / No Breakout Days", fmt_num(c.get("total_count"),0),
                      f"NewHigh: {fmt_num(c.get('new_high_rate'),0,pct=True)} | NewLow: {fmt_num(c.get('new_low_rate'),0,pct=True)}"))
    cards.append(card("Chop Params", f"band={fmt_num(meta.get('chop_band_pct'),2)}", f"maxRangeMult={fmt_num(meta.get('chop_max_range_mult'),2)}"))

    # -------------------- TABLE COLUMNS --------------------
    lh_cols = ["lh_move_pct", "lh_range", "lh_volume", "lh_vol_vs_day"]

    sell_cols = [
        "date","tf","fh_move_pct","first_hour_range",
        "final_abs_low","hours_to_final_abs_low","extension_points_below_fh_low",
        "continuation_hours","chop_hours","bounce_25_hours","bounce_50_hours","failed_selloff",
        *lh_cols
    ]
    buy_cols = [
        "date","tf","fh_move_pct","first_hour_range",
        "final_abs_high","hours_to_final_abs_high","extension_points_above_fh_high",
        "continuation_hours","chop_hours","pullback_25_hours","pullback_50_hours","failed_buyup",
        *lh_cols
    ]
    chop_cols = [
        "date","tf","fh_move_pct",
        "fh_high","fh_low",
        "final_abs_high","hours_to_final_abs_high","extension_points_above_fh_high","made_new_high_after_fh",
        "final_abs_low","hours_to_final_abs_low","extension_points_below_fh_low","made_new_low_after_fh",
        *lh_cols
    ]

    # -------------------- SUMMARY BLOCKS --------------------
    def metric_row(label, value, details=""):
        return f"""
        <tr>
          <td><b>{label}</b></td>
          <td>{value}</td>
          <td class="muted">{details}</td>
        </tr>
        """

    sell_summary_html = f"""
    <h3>📉 Sell-Off Detailed Metrics</h3>
    <table class="table">
      <thead><tr><th>Metric</th><th>Value</th><th>Details</th></tr></thead>
      <tbody>
        {metric_row("Continued Lower", fmt_num(s.get("continuation_rate"),0,pct=True),
                    f"Avg time to final low: {fmt_num(s.get('avg_hours_to_final_abs_low'),2)} hours")}
        {metric_row("Entered Chop Zone", fmt_num(s.get("chop_rate"),0,pct=True),
                    f"Avg chop hours: {fmt_num(s.get('avg_chop_hours'),2)}")}
        {metric_row("50% Bounce Back", fmt_num(s.get("bounce_50_rate"),0,pct=True),
                    "")}
        {metric_row("25% Bounce Back", fmt_num(s.get("bounce_25_rate"),0,pct=True),
                    "")}
        {metric_row("Avg Extension Below FH Low", fmt_num(s.get("avg_extension_below_fh_low"),2),
                    "Points below first-hour low")}
        {metric_row("Avg Last-Hour Move %", fmt_num(s.get("avg_lh_move_pct"),2), "15:00–16:00 ET")}
      </tbody>
    </table>
    """

    buy_summary_html = f"""
    <h3>📈 Buy-Up Detailed Metrics</h3>
    <table class="table">
      <thead><tr><th>Metric</th><th>Value</th><th>Details</th></tr></thead>
      <tbody>
        {metric_row("Continued Higher", fmt_num(b.get("continuation_rate"),0,pct=True),
                    f"Avg time to final high: {fmt_num(b.get('avg_hours_to_final_abs_high'),2)} hours")}
        {metric_row("Entered Chop Zone", fmt_num(b.get("chop_rate"),0,pct=True),
                    f"Avg chop hours: {fmt_num(b.get('avg_chop_hours'),2)}")}
        {metric_row("50% Pullback", fmt_num(b.get("pullback_50_rate"),0,pct=True), "")}
        {metric_row("25% Pullback", fmt_num(b.get("pullback_25_rate"),0,pct=True), "")}
        {metric_row("Avg Extension Above FH High", fmt_num(b.get("avg_extension_above_fh_high"),2),
                    "Points above first-hour high")}
        {metric_row("Avg Last-Hour Move %", fmt_num(b.get("avg_lh_move_pct"),2), "15:00–16:00 ET")}
      </tbody>
    </table>
    """

    chop_summary_html = f"""
    <h3>🟨 Chop / No Breakout Detailed Metrics</h3>
    <table class="table">
      <thead><tr><th>Metric</th><th>Value</th><th>Details</th></tr></thead>
      <tbody>
        {metric_row("Made New High After FH", fmt_num(c.get("new_high_rate"),0,pct=True),
                    f"Avg hours to final high: {fmt_num(c.get('avg_hours_to_final_high'),2)}")}
        {metric_row("Made New Low After FH", fmt_num(c.get("new_low_rate"),0,pct=True),
                    f"Avg hours to final low: {fmt_num(c.get('avg_hours_to_final_low'),2)}")}
        {metric_row("Avg Ext Above FH High", fmt_num(c.get("avg_ext_above_fh_high"),2), "")}
        {metric_row("Avg Ext Below FH Low", fmt_num(c.get("avg_ext_below_fh_low"),2), "")}
        {metric_row("Avg Last-Hour Move %", fmt_num(c.get("avg_lh_move_pct"),2), "15:00–16:00 ET")}
      </tbody>
    </table>
    """

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{symbol} First Hour Volume Analysis ({tf})</title>
  <style>
    body {{ font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial; margin: 18px; }}
    h1 {{ margin: 0 0 6px 0; }}
    .muted {{ color: #666; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 12px 0; }}
    .card {{ border: 1px solid #ddd; border-radius: 10px; padding: 12px; background: #fff; }}
    .card-title {{ font-size: 12px; color: #555; }}
    .card-value {{ font-size: 22px; font-weight: 700; margin-top: 4px; }}
    .card-sub {{ font-size: 12px; color: #777; margin-top: 4px; }}
    .section {{ margin-top: 22px; }}

    .tabs {{ display: flex; gap: 8px; margin: 12px 0; }}
    .tab {{ padding: 8px 10px; border: 1px solid #ccc; border-radius: 10px; cursor: pointer; user-select: none; }}
    .tab.active {{ background: #111; color: #fff; border-color: #111; }}

    .panel {{ display: none; }}
    .panel.active {{ display: block; }}

    .table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    .table th, .table td {{ border-bottom: 1px solid #eee; padding: 8px; font-size: 12px; text-align: left; vertical-align: top; }}
    .table th {{ position: sticky; top: 0; background: #fafafa; }}

    .two-col {{ display:grid; grid-template-columns: 1fr 1fr; gap:16px; }}

    /* Modal */
    .modal-backdrop {{
      display:none; position:fixed; inset:0; background:rgba(0,0,0,0.55);
      align-items:center; justify-content:center; padding:20px;
    }}
    .modal {{
      background:#fff; border-radius:12px; max-width:1100px; width:100%;
      padding:14px; box-shadow:0 10px 30px rgba(0,0,0,0.35);
    }}
    .modal-head {{
      display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;
    }}
    .btn {{
      border:1px solid #ccc; border-radius:10px; padding:6px 10px; cursor:pointer; background:#fff;
    }}
    img.chartimg {{ width:100%; height:auto; border:1px solid #eee; border-radius:10px; }}
  </style>
</head>
<body>
  <h1>📊 {symbol} First Hour Volume Analysis ({tf})</h1>
  <div class="muted">RTH: 09:30–16:00 ET • First hour: 09:30–10:30 ET • Click <b>View</b> for that day’s candle chart.</div>

  <div class="grid">
    {''.join(cards)}
  </div>

  <div class="section two-col">
    <div>{sell_summary_html}</div>
    <div>{buy_summary_html}</div>
  </div>

  <div class="section">
    {chop_summary_html}
  </div>

  <div class="section">
    <h2>📋 Detailed Event Analysis</h2>
    <div class="tabs">
      <div class="tab active" data-panel="sell">Sell-off Events</div>
      <div class="tab" data-panel="buy">Buy-up Events</div>
      <div class="tab" data-panel="chop">Chop / No Breakout Days</div>
    </div>

    <div id="panel-sell" class="panel active">
      {df_to_html(sell_df, sell_cols)}
    </div>

    <div id="panel-buy" class="panel">
      {df_to_html(buy_df, buy_cols)}
    </div>

    <div id="panel-chop" class="panel">
      {df_to_html(chop_df, chop_cols)}
    </div>
  </div>

  <!-- Modal -->
  <div id="backdrop" class="modal-backdrop">
    <div class="modal">
      <div class="modal-head">
        <div><b id="modalTitle">Chart</b></div>
        <div class="btn" id="closeBtn">Close</div>
      </div>
      <div id="modalBody" class="muted">Loading...</div>
    </div>
  </div>

<script>
  const CHARTS = {json.dumps(charts)};

  function show(panel) {{
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    document.getElementById('panel-' + panel).classList.add('active');
    document.querySelector('.tab[data-panel="' + panel + '"]').classList.add('active');
  }}
  document.querySelectorAll('.tab').forEach(t => {{
    t.addEventListener('click', () => show(t.dataset.panel));
  }});

  const backdrop = document.getElementById('backdrop');
  const modalTitle = document.getElementById('modalTitle');
  const modalBody = document.getElementById('modalBody');
  const closeBtn = document.getElementById('closeBtn');

  function openModal(dateStr) {{
    modalTitle.textContent = "{symbol} {tf} | " + dateStr;
    const b64 = CHARTS[dateStr];
    if (!b64) {{
      modalBody.innerHTML = "<div class='muted'>No chart found for " + dateStr + "</div>";
    }} else {{
      modalBody.innerHTML = "<img class='chartimg' src='data:image/png;base64," + b64 + "' />";
    }}
    backdrop.style.display = "flex";
  }}

  function closeModal() {{
    backdrop.style.display = "none";
  }}

  closeBtn.addEventListener('click', closeModal);
  backdrop.addEventListener('click', (e) => {{
    if (e.target === backdrop) closeModal();
  }});

  document.addEventListener('click', (e) => {{
    const a = e.target.closest('.chartlink');
    if (!a) return;
    e.preventDefault();
    openModal(a.dataset.date);
  }});
</script>

</body>
</html>
"""

    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"✅ Wrote {out_html}")


if __name__ == "__main__":
    main()
