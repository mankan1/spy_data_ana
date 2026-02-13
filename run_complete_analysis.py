#!/usr/bin/env python3
"""
First Hour Volume Analysis - Complete Package (Multi-timeframe + Chop Days + Last Hour Reaction)

- Uses NY RTH time (America/New_York)
- Filters to 9:30–16:00 ET
- Defines first hour as 9:30–10:30 ET (time window)
- Supports timeframes: 15m, 30m, 1h, 2h

Event filtering:
- "Big first hour volume day" = first hour volume >= volume_quantile threshold (default q=0.75)

Buckets (on big-volume days):
1) Sell-off (FH move % <= -move_threshold_pct)
2) Buy-up   (FH move % >= +move_threshold_pct)
3) Chop/No Breakout (abs(FH move %) < move_threshold_pct)

Metrics per bucket:
- final_abs_low/high after first hour
- time-to-final-abs-low/high in HOURS (timeframe-aware)
- extension points beyond FH low/high
- continuation hours (time to final abs extreme if it extends beyond FH)
- bounce/pullback 25%/50% measured from running extremes (sell/buy)
- friendly chop hours (sell/buy) before 25% bounce/pullback

NEW: Last hour reaction (15:00–16:00 ET) on every event day:
- lh_open/high/low/close, lh_range, lh_move_pct
- lh_volume, lh_vol_vs_day

Outputs:
- SYMBOL_TF_selloffs_detail.csv
- SYMBOL_TF_buyups_detail.csv
- SYMBOL_TF_chop_days_detail.csv
- SYMBOL_TF_summary_stats.json
- SYMBOL_TF_analysis_dashboard.html (via create_dashboard.py)

Usage:
  python run_complete_analysis.py --symbol SPY --tf 30m
  python run_complete_analysis.py --symbol QQQ --tf 15m
  python run_complete_analysis.py --symbol SPY --tf 1h
  python run_complete_analysis.py --symbol SPY --tf 30m --chop_band_pct 0.7 --chop_max_range_mult 2.5

Requirements:
  yfinance pandas numpy
"""

import argparse
import subprocess
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import base64
from io import BytesIO

SUPPORTED_TF = ["15m", "30m", "1h", "2h"]

DEFAULTS_BY_TF = {
    "15m": {"chop_band_pct": 0.35, "chop_max_range_mult": 1.50},
    "30m": {"chop_band_pct": 0.45, "chop_max_range_mult": 1.75},
    "1h":  {"chop_band_pct": 0.80, "chop_max_range_mult": 2.50},
    "2h":  {"chop_band_pct": 1.10, "chop_max_range_mult": 3.25},
}


def check_and_install_dependencies():
    required = ["yfinance", "pandas", "numpy"]
    print("Checking dependencies...")
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")


def tf_to_hours(tf: str) -> float:
    if tf.endswith("m"):
        return int(tf[:-1]) / 60.0
    if tf.endswith("h"):
        return float(int(tf[:-1]))
    raise ValueError(f"Unsupported tf: {tf}")


def flatten_columns_if_needed(df):
    import pandas as pd
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if x not in (None, "")]).strip("_")
            for col in df.columns
        ]
    return df


def pick_col(df, base):
    base_lower = base.lower()
    if base in df.columns:
        return base
    cands = [c for c in df.columns if c.split("_")[0].lower() == base_lower]
    if not cands:
        cands = [c for c in df.columns if c.lower().startswith(base_lower)]
    if not cands:
        raise KeyError(f"Missing '{base}' column. Available columns: {list(df.columns)}")
    return cands[0]


def ensure_datetime_column(df):
    cols = list(df.columns)

    def col_to_str(c):
        if isinstance(c, tuple):
            return "_".join([str(x) for x in c if x not in (None, "")])
        return str(c)

    col_strs = [col_to_str(c) for c in cols]

    if "Datetime" in col_strs:
        idx = col_strs.index("Datetime")
        if cols[idx] != "Datetime":
            df = df.rename(columns={cols[idx]: "Datetime"})
        return df

    if "Date" in col_strs:
        idx = col_strs.index("Date")
        df = df.rename(columns={cols[idx]: "Datetime"})
        return df

    if "index" in col_strs:
        idx = col_strs.index("index")
        df = df.rename(columns={cols[idx]: "Datetime"})
        return df

    for c_real, c_str in zip(cols, col_strs):
        s = c_str.lower()
        if "datetime" in s or (("date" in s) and ("adj" not in s) and ("close" not in s)):
            df = df.rename(columns={c_real: "Datetime"})
            return df

    raise KeyError(f"Couldn't find a datetime column after reset_index(). Columns: {col_strs}")


def convert_to_ny_time(df):
    import pandas as pd
    dt = pd.to_datetime(df["Datetime"], errors="coerce")
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        dt = dt.dt.tz_convert("America/New_York")
    df["Datetime"] = dt
    return df


def compute_last_hour_metrics(day_data, rth_end_time):
    """
    Last hour defined as 15:00–16:00 ET.
    Returns dict with last-hour stats, or None-values if missing.
    """
    from datetime import time

    last_start = time(15, 0)
    last_end = rth_end_time  # 16:00

    lh = day_data[(day_data["Time"] >= last_start) & (day_data["Time"] <= last_end)].copy()
    lh = lh.reset_index(drop=True)

    if lh.empty:
        return {
            "lh_bars": 0,
            "lh_open": None, "lh_high": None, "lh_low": None, "lh_close": None,
            "lh_range": None, "lh_move_pct": None,
            "lh_volume": None, "lh_vol_vs_day": None
        }

    lh_open = float(lh.iloc[0]["Open"])
    lh_high = float(lh["High"].max())
    lh_low = float(lh["Low"].min())
    lh_close = float(lh.iloc[-1]["Close"])
    lh_range = lh_high - lh_low
    lh_move_pct = ((lh_close - lh_open) / lh_open) * 100.0 if lh_open else None

    lh_volume = float(lh["Volume"].sum())
    day_volume = float(day_data["Volume"].sum()) if len(day_data) else 0.0
    lh_vol_vs_day = (lh_volume / day_volume) if day_volume > 0 else None

    return {
        "lh_bars": int(len(lh)),
        "lh_open": round(lh_open, 4),
        "lh_high": round(lh_high, 4),
        "lh_low": round(lh_low, 4),
        "lh_close": round(lh_close, 4),
        "lh_range": round(lh_range, 4),
        "lh_move_pct": None if lh_move_pct is None else round(lh_move_pct, 4),
        "lh_volume": round(lh_volume, 2),
        "lh_vol_vs_day": None if lh_vol_vs_day is None else round(lh_vol_vs_day, 4),
    }


def safe_mean(series):
    s = series.dropna()
    return float(s.mean()) if len(s) else 0.0

def make_candles_base64(day_df, title=""):
    """
    Build a simple candlestick chart (no external libs) and return base64 PNG string.
    day_df must have columns: Datetime, Open, High, Low, Close
    """
    if day_df is None or day_df.empty:
        return None

    d = day_df.copy().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    # X positions
    xs = list(range(len(d)))

    for i, row in d.iterrows():
        o = float(row["Open"])
        h = float(row["High"])
        l = float(row["Low"])
        c = float(row["Close"])

        up = c >= o
        # Candle body
        body_low = min(o, c)
        body_high = max(o, c)
        body_h = max(body_high - body_low, 1e-9)

        # Wick
        ax.vlines(i, l, h, linewidth=1)

        # Body rectangle (no fixed colors requested; using default matplotlib cycle is ugly)
        # We'll keep it minimal: filled for up candles, unfilled for down candles.
        rect = Rectangle(
            (i - 0.30, body_low),
            0.60,
            body_h,
            fill=up,
            linewidth=1,
        )
        ax.add_patch(rect)

    # X labels as time stamps (sparse)
    times = [t.strftime("%H:%M") for t in d["Datetime"]]
    step = max(1, len(times) // 10)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels(times[::step], rotation=0)

    ax.set_xlim(-1, len(d))
    ax.set_ylabel("Price")

    # Tight bounds
    ymin = float(d["Low"].min())
    ymax = float(d["High"].max())
    pad = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")
    
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (SPY, QQQ, IWM, AAPL, etc.)")
    parser.add_argument("--tf", default="1h", choices=SUPPORTED_TF, help="Timeframe: 15m, 30m, 1h, 2h")
    parser.add_argument("--days", type=int, default=180, help="Lookback days (default 180)")

    parser.add_argument("--volume_quantile", type=float, default=0.75, help="Big volume threshold quantile (default 0.75)")
    parser.add_argument("--move_threshold_pct", type=float, default=0.30, help="FH breakout threshold in percent (default 0.30)")

    parser.add_argument("--chop_band_pct", type=float, default=None, help="Override chop band pct (else auto by timeframe)")
    parser.add_argument("--chop_max_range_mult", type=float, default=None, help="Override chop max bar range mult (else auto by timeframe)")

    args = parser.parse_args()
    symbol = args.symbol.upper()
    tf = args.tf

    defaults = DEFAULTS_BY_TF[tf]
    CHOP_BAND_PCT = defaults["chop_band_pct"] if args.chop_band_pct is None else float(args.chop_band_pct)
    CHOP_MAX_RANGE_MULT = defaults["chop_max_range_mult"] if args.chop_max_range_mult is None else float(args.chop_max_range_mult)

    BAR_HOURS = tf_to_hours(tf)
    move_thr = float(args.move_threshold_pct)

    print("=" * 80)
    print(f"{symbol} FIRST HOUR VOLUME ANALYSIS | TF={tf} | Lookback={args.days}d")
    print("=" * 80)
    print(f"Chop defaults for {tf}: band_pct={defaults['chop_band_pct']}, max_range_mult={defaults['chop_max_range_mult']}")
    print(f"Using chop params: band_pct={CHOP_BAND_PCT}, max_range_mult={CHOP_MAX_RANGE_MULT}")
    print(f"Bar size: {BAR_HOURS} hours")
    if tf == "2h":
        print("⚠️ Note: 2h bars blur the 09:30–10:30 window. 15m/30m recommended for best accuracy.")
    print()

    check_and_install_dependencies()
    print()

    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta, time
    import json

    print("=" * 80)
    print(f"STEP 1: Fetching {symbol} data from Yahoo Finance")
    print("=" * 80)

    # --- Yahoo intraday limits ---
    INTRADAY_MAX_DAYS = {
        "15m": 60,
        "30m": 60,
        "1h": 730,
        "2h": 730,
    }

    max_days = INTRADAY_MAX_DAYS.get(tf, args.days)
    if args.days > max_days:
        print(f"⚠️ Yahoo limit: {tf} supports ~{max_days} days. Reducing lookback from {args.days} -> {max_days}.")
        args.days = max_days

    # ✅ Use period for intraday to avoid Yahoo start/end rejection
    def days_to_period(d):
        # Keep inside Yahoo intraday window; 59d is often safer than 60d
        if d >= 60:
            return "59d"
        return f"{int(d)}d"

    period = days_to_period(args.days)

    print(f"Downloading {symbol} | interval={tf} | period={period} ...")

    try:
        df = yf.download(
            symbol,
            interval=tf,
            period=period,
            progress=True,
            auto_adjust=False,
            prepost=False,
            threads=True,
        )
        print(f"\n✓ Successfully downloaded {len(df)} bars")
    except Exception as e:
        print(f"\n✗ Error downloading data: {e}")
        sys.exit(1)

    if len(df) == 0:
        print("✗ No data received.")
        sys.exit(1)
        
    # print("=" * 80)
    # print(f"STEP 1: Fetching {symbol} data from Yahoo Finance")
    # print("=" * 80)

    # end_date = datetime.now()
    # start_date = end_date - timedelta(days=args.days)

    # print(f"Downloading {symbol} data from {start_date.date()} to {end_date.date()} | interval={tf} ...")

    # # --- Yahoo intraday limits (common): 15m/30m only ~60 days ---
    # INTRADAY_MAX_DAYS = {
    #     "15m": 60,
    #     "30m": 60,
    #     "1h": 730,   # Yahoo typically allows much longer for 60m
    #     "2h": 730,
    # }

    # max_days = INTRADAY_MAX_DAYS.get(tf, args.days)
    # if args.days > max_days:
    #     print(f"⚠️ Yahoo limit: {tf} supports ~{max_days} days. Reducing lookback from {args.days} -> {max_days}.")
    #     args.days = max_days

    # end_date = datetime.now()
    # start_date = end_date - timedelta(days=args.days)

    # try:
    #     df = yf.download(symbol, start=start_date, end=end_date, interval=tf, progress=True)
    #     print(f"\n✓ Successfully downloaded {len(df)} bars")
    # except Exception as e:
    #     print(f"\n✗ Error downloading data: {e}")
    #     sys.exit(1)

    if len(df) == 0:
        print("✗ No data received.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("STEP 2: Processing and Analyzing Data")
    print("=" * 80)

    df = df.reset_index()
    df = flatten_columns_if_needed(df)
    df = ensure_datetime_column(df)

    # Normalize OHLCV
    for base in ["Open", "High", "Low", "Close", "Volume"]:
        src = pick_col(df, base)
        if src != base:
            df[base] = df[src]

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert to NY time
    df = convert_to_ny_time(df)
    df = df.sort_values("Datetime").reset_index(drop=True)

    df["Time"] = df["Datetime"].dt.time
    df["Date"] = df["Datetime"].dt.date

    # Filter RTH 09:30–16:00
    rth_start = time(9, 30)
    rth_end = time(16, 0)
    df = df[(df["Time"] >= rth_start) & (df["Time"] <= rth_end)].copy()

    # First hour 09:30–10:30
    fh_end = time(10, 30)
    df["IsFirstHour"] = df["Time"].apply(lambda t: rth_start <= t < fh_end)

    # First hour volume threshold
    print("\nCalculating first-hour volume threshold...")
    first_hour_volume = df.loc[df["IsFirstHour"]].groupby("Date")["Volume"].sum()
    if len(first_hour_volume) == 0:
        print("✗ No first-hour bars found in 09:30–10:30 window. Use 15m/30m for best accuracy.")
        sys.exit(1)

    big_volume_threshold = float(first_hour_volume.quantile(args.volume_quantile))
    print(f"✓ Big volume threshold (q={args.volume_quantile:.2f}): {big_volume_threshold:,.0f}")

    print("\nAnalyzing event days...")
    results = {"selloffs": [], "buyups": [], "chop_days": []}

    # ✅ IMPORTANT: define charts storage ONCE (not inside loop)
    charts_by_date = {}

    for date in df["Date"].unique():
        day_data = df[df["Date"] == date].sort_values("Datetime").reset_index(drop=True)

        if len(day_data) < 2:
            continue

        fh = day_data[day_data["IsFirstHour"]].reset_index(drop=True)
        if len(fh) == 0:
            continue

        fh_volume = float(fh["Volume"].sum())
        if fh_volume < big_volume_threshold:
            continue

        # Build a chart for this whole RTH day (after we know it's a big-volume event day)
        chart_b64 = make_candles_base64(
            day_data,
            title=f"{symbol} {tf} | {date} (RTH)"
        )
        charts_by_date[str(date)] = chart_b64

        # last hour reaction for this day
        lh_metrics = compute_last_hour_metrics(day_data, rth_end_time=rth_end)

        fh_open = float(fh.iloc[0]["Open"])
        fh_high = float(fh["High"].max())
        fh_low = float(fh["Low"].min())
        fh_close = float(fh.iloc[-1]["Close"])

        fh_move_pct = ((fh_close - fh_open) / fh_open) * 100.0
        after_fh = day_data[~day_data["IsFirstHour"]].reset_index(drop=True)

        # final abs high/low AFTER first hour
        final_abs_high = None
        hours_to_final_abs_high = None
        ext_above = None

        final_abs_low = None
        hours_to_final_abs_low = None
        ext_below = None

        if len(after_fh) > 0:
            idx_high = int(after_fh["High"].values.argmax())
            idx_low = int(after_fh["Low"].values.argmin())

            final_abs_high = float(after_fh["High"].iloc[idx_high])
            final_abs_low = float(after_fh["Low"].iloc[idx_low])

            hours_to_final_abs_high = round((idx_high + 1) * BAR_HOURS, 4)
            hours_to_final_abs_low = round((idx_low + 1) * BAR_HOURS, 4)

            ext_above = round(final_abs_high - fh_high, 4)
            ext_below = round(fh_low - final_abs_low, 4)

        # ---------------- SELL-OFF ----------------
        if fh_move_pct <= -move_thr:
            fh_range = fh_open - fh_low

            continuation_found = (final_abs_low is not None) and (final_abs_low < fh_low)
            failed = not continuation_found
            continuation_hours = hours_to_final_abs_low if continuation_found else 0.0

            bounce_25_hours = None
            bounce_50_hours = None
            chop_hours = 0.0

            if len(after_fh) > 0 and fh_range > 0:
                abs_bottom = fh_low

                for j, row in after_fh.iterrows():
                    elapsed_hours = (j + 1) * BAR_HOURS

                    bar_low = float(row["Low"])
                    bar_high = float(row["High"])
                    bar_range = bar_high - bar_low

                    made_new_low = bar_low < abs_bottom
                    if made_new_low:
                        abs_bottom = bar_low

                    recovery = bar_high - abs_bottom
                    recovery_pct = (recovery / fh_range) * 100.0

                    if bounce_25_hours is None and recovery_pct >= 25:
                        bounce_25_hours = round(elapsed_hours, 4)
                    if bounce_50_hours is None and recovery_pct >= 50:
                        bounce_50_hours = round(elapsed_hours, 4)

                    # Friendly chop until 25% bounce occurs
                    if bounce_25_hours is None:
                        in_band = bar_high <= (abs_bottom + CHOP_BAND_PCT * fh_range)
                        not_huge = bar_range <= (CHOP_MAX_RANGE_MULT * fh_range)
                        if (not made_new_low) and in_band and not_huge:
                            chop_hours += BAR_HOURS

            results["selloffs"].append({
                "date": str(date),
                "tf": tf,
                "fh_move_pct": round(fh_move_pct, 4),
                "first_hour_range": round(fh_range, 4),

                "final_abs_low": None if final_abs_low is None else round(final_abs_low, 4),
                "hours_to_final_abs_low": hours_to_final_abs_low,
                "extension_points_below_fh_low": ext_below,

                "continuation_hours": continuation_hours,
                "chop_hours": round(chop_hours, 4),
                "bounce_25_hours": bounce_25_hours,
                "bounce_50_hours": bounce_50_hours,
                "failed_selloff": bool(failed),

                # last hour reaction
                **lh_metrics,
            })

        # ---------------- BUY-UP ----------------
        elif fh_move_pct >= move_thr:
            fh_range = fh_high - fh_open

            continuation_found = (final_abs_high is not None) and (final_abs_high > fh_high)
            failed = not continuation_found
            continuation_hours = hours_to_final_abs_high if continuation_found else 0.0

            pullback_25_hours = None
            pullback_50_hours = None
            chop_hours = 0.0

            if len(after_fh) > 0 and fh_range > 0:
                abs_top = fh_high

                for j, row in after_fh.iterrows():
                    elapsed_hours = (j + 1) * BAR_HOURS

                    bar_low = float(row["Low"])
                    bar_high = float(row["High"])
                    bar_range = bar_high - bar_low

                    made_new_high = bar_high > abs_top
                    if made_new_high:
                        abs_top = bar_high

                    pullback = abs_top - bar_low
                    pullback_pct = (pullback / fh_range) * 100.0

                    if pullback_25_hours is None and pullback_pct >= 25:
                        pullback_25_hours = round(elapsed_hours, 4)
                    if pullback_50_hours is None and pullback_pct >= 50:
                        pullback_50_hours = round(elapsed_hours, 4)

                    # Friendly chop until 25% pullback occurs
                    if pullback_25_hours is None:
                        in_band = bar_low >= (abs_top - CHOP_BAND_PCT * fh_range)
                        not_huge = bar_range <= (CHOP_MAX_RANGE_MULT * fh_range)
                        if (not made_new_high) and in_band and not_huge:
                            chop_hours += BAR_HOURS

            results["buyups"].append({
                "date": str(date),
                "tf": tf,
                "fh_move_pct": round(fh_move_pct, 4),
                "first_hour_range": round(fh_range, 4),

                "final_abs_high": None if final_abs_high is None else round(final_abs_high, 4),
                "hours_to_final_abs_high": hours_to_final_abs_high,
                "extension_points_above_fh_high": ext_above,

                "continuation_hours": continuation_hours,
                "chop_hours": round(chop_hours, 4),
                "pullback_25_hours": pullback_25_hours,
                "pullback_50_hours": pullback_50_hours,
                "failed_buyup": bool(failed),

                # last hour reaction
                **lh_metrics,
            })

        # ---------------- CHOP / NO BREAKOUT ----------------
        else:
            made_new_high = (final_abs_high is not None) and (final_abs_high > fh_high)
            made_new_low = (final_abs_low is not None) and (final_abs_low < fh_low)

            results["chop_days"].append({
                "date": str(date),
                "tf": tf,
                "fh_move_pct": round(fh_move_pct, 4),

                "fh_high": round(fh_high, 4),
                "fh_low": round(fh_low, 4),

                "final_abs_high": None if final_abs_high is None else round(final_abs_high, 4),
                "hours_to_final_abs_high": hours_to_final_abs_high,
                "extension_points_above_fh_high": ext_above,
                "made_new_high_after_fh": bool(made_new_high),

                "final_abs_low": None if final_abs_low is None else round(final_abs_low, 4),
                "hours_to_final_abs_low": hours_to_final_abs_low,
                "extension_points_below_fh_low": ext_below,
                "made_new_low_after_fh": bool(made_new_low),

                # last hour reaction
                **lh_metrics,
            })

    charts_json = f"{symbol}_{tf}_charts.json"
    with open(charts_json, "w") as f:
        json.dump(charts_by_date, f)

    print(f"✓ Saved charts: {charts_json} (base64 PNGs)")

    # ---------------- Save + Summary ----------------
    import pandas as pd
    import json

    sell_df = pd.DataFrame(results["selloffs"])
    buy_df = pd.DataFrame(results["buyups"])
    chop_df = pd.DataFrame(results["chop_days"])

    summary = {
        "meta": {
            "symbol": symbol,
            "tf": tf,
            "bar_hours": BAR_HOURS,
            "days": args.days,
            "volume_quantile": args.volume_quantile,
            "big_volume_threshold": big_volume_threshold,
            "move_threshold_pct": move_thr,
            "chop_band_pct": CHOP_BAND_PCT,
            "chop_max_range_mult": CHOP_MAX_RANGE_MULT,
        },
        "selloffs": {},
        "buyups": {},
        "chop_days": {},
    }

    if len(sell_df):
        summary["selloffs"] = {
            "total_count": int(len(sell_df)),
            "continuation_rate": float((sell_df["continuation_hours"] > 0).mean()),
            "avg_first_hour_range": safe_mean(sell_df["first_hour_range"]),
            "avg_extension_below_fh_low": safe_mean(sell_df["extension_points_below_fh_low"]),
            "avg_hours_to_final_abs_low": safe_mean(sell_df["hours_to_final_abs_low"]),
            "avg_chop_hours": safe_mean(sell_df["chop_hours"]),
            "chop_rate": float((sell_df["chop_hours"] > 0).mean()),
            "bounce_25_rate": float(sell_df["bounce_25_hours"].notna().mean()),
            "bounce_50_rate": float(sell_df["bounce_50_hours"].notna().mean()),
            "avg_lh_move_pct": safe_mean(sell_df["lh_move_pct"]) if "lh_move_pct" in sell_df.columns else 0.0,
        }

    if len(buy_df):
        summary["buyups"] = {
            "total_count": int(len(buy_df)),
            "continuation_rate": float((buy_df["continuation_hours"] > 0).mean()),
            "avg_first_hour_range": safe_mean(buy_df["first_hour_range"]),
            "avg_extension_above_fh_high": safe_mean(buy_df["extension_points_above_fh_high"]),
            "avg_hours_to_final_abs_high": safe_mean(buy_df["hours_to_final_abs_high"]),
            "avg_chop_hours": safe_mean(buy_df["chop_hours"]),
            "chop_rate": float((buy_df["chop_hours"] > 0).mean()),
            "pullback_25_rate": float(buy_df["pullback_25_hours"].notna().mean()),
            "pullback_50_rate": float(buy_df["pullback_50_hours"].notna().mean()),
            "avg_lh_move_pct": safe_mean(buy_df["lh_move_pct"]) if "lh_move_pct" in buy_df.columns else 0.0,
        }

    if len(chop_df):
        summary["chop_days"] = {
            "total_count": int(len(chop_df)),
            "new_high_rate": float(chop_df["made_new_high_after_fh"].mean()),
            "new_low_rate": float(chop_df["made_new_low_after_fh"].mean()),
            "avg_ext_above_fh_high": safe_mean(chop_df["extension_points_above_fh_high"]),
            "avg_ext_below_fh_low": safe_mean(chop_df["extension_points_below_fh_low"]),
            "avg_hours_to_final_high": safe_mean(chop_df["hours_to_final_abs_high"]),
            "avg_hours_to_final_low": safe_mean(chop_df["hours_to_final_abs_low"]),
            "avg_lh_move_pct": safe_mean(chop_df["lh_move_pct"]) if "lh_move_pct" in chop_df.columns else 0.0,
        }

    sell_csv = f"{symbol}_{tf}_selloffs_detail.csv"
    buy_csv = f"{symbol}_{tf}_buyups_detail.csv"
    chop_csv = f"{symbol}_{tf}_chop_days_detail.csv"
    sum_json = f"{symbol}_{tf}_summary_stats.json"

    sell_df.to_csv(sell_csv, index=False)
    buy_df.to_csv(buy_csv, index=False)
    chop_df.to_csv(chop_csv, index=False)
    with open(sum_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 3: Saved Outputs")
    print("=" * 80)
    print(f"✓ {sell_csv}")
    print(f"✓ {buy_csv}")
    print(f"✓ {chop_csv}")
    print(f"✓ {sum_json}")

    print("\n" + "=" * 80)
    print("STEP 4: Generating Dashboard")
    print("=" * 80)
    try:
        subprocess.run([sys.executable, "create_dashboard.py", "--symbol", symbol, "--tf", tf], check=True)
        print(f"\n✅ Dashboard created: {symbol}_{tf}_analysis_dashboard.html")
    except Exception as e:
        print(f"\nNote: Could not auto-generate dashboard: {e}")
        print(f"Run manually: python create_dashboard.py --symbol {symbol} --tf {tf}")


if __name__ == "__main__":
    main()
