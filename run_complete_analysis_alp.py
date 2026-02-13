#!/usr/bin/env python3
"""
run_complete_analysis.py  (ALPACA version)

First Hour Volume Analysis - Multi-timeframe + Chop Days + Other Days + Last Hour Reaction + Candle Charts

✅ Uses Alpaca Market Data (stocks) so you can:
- pull TRUE intraday history beyond Yahoo's 60-day limit (15m/30m)
- use --days and --offset_days to “go back in time”

Time & session rules:
- Convert timestamps to America/New_York
- Filter RTH 09:30–16:00 ET
- First hour window: 09:30–10:30 ET

Timeframes:
- 15m, 30m, 1h, 2h

Buckets (all days with first-hour bars):
- BigVol days: first-hour volume >= quantile threshold (default q=0.75)
  - Sell-off (FH move % <= -move_threshold_pct)
  - Buy-up   (FH move % >= +move_threshold_pct)
  - BigVol Chop/No Breakout (abs(FH move %) < move_threshold_pct)
- Other/Normal days: NOT big-vol days (FH volume < threshold)

Bounce/pullback logic:
- Sell-off: 25%/50% bounce measured from the RUNNING BOTTOM after first hour
- Buy-up:   25%/50% pullback measured from the RUNNING TOP after first hour
Also records:
- the "target price" (bottom + 0.25*FH_range, etc)
- the "hit price" (bar high/low that first met the threshold)
- hours are measured AFTER the first hour (timeframe-aware)

Last hour reaction (15:00–16:00 ET) for EVERY day included in outputs:
- lh_open/high/low/close, lh_range, lh_move_pct
- lh_volume, lh_vol_vs_day

Candlestick charts:
- stores base64 PNG for each day into SYMBOL_TF_charts.json
- create_dashboard.py can show them in a modal (your current one already supports this)

Outputs:
- SYMBOL_TF_selloffs_detail.csv
- SYMBOL_TF_buyups_detail.csv
- SYMBOL_TF_chop_days_detail.csv      (BigVol chop/no breakout)
- SYMBOL_TF_other_days_detail.csv     (non-big-vol days)
- SYMBOL_TF_summary_stats.json
- SYMBOL_TF_charts.json
- SYMBOL_TF_analysis_dashboard.html   (via create_dashboard.py)

Usage examples:
  # last 30 calendar days of 30m bars
  python run_complete_analysis.py --symbol SPY --tf 30m --days 30

  # 30-day window that ends 60 days ago
  python run_complete_analysis.py --symbol SPY --tf 30m --days 30 --offset_days 60

  # 180-day window ending 0 days ago (works fine on Alpaca)
  python run_complete_analysis.py --symbol SPY --tf 15m --days 180

Alpaca credentials (required):
  export ALPACA_API_KEY="..."
  export ALPACA_API_SECRET="..."
  export ALPACA_DATA_FEED="iex"   # or "sip" if you have it (optional, default "iex")

Dependencies:
  pip install alpaca-py pandas numpy matplotlib
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import datetime, timedelta, time
from io import BytesIO

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

SUPPORTED_TF = ["15m", "30m", "1h", "2h"]

DEFAULTS_BY_TF = {
    "15m": {"chop_band_pct": 0.35, "chop_max_range_mult": 1.50},
    "30m": {"chop_band_pct": 0.45, "chop_max_range_mult": 1.75},
    "1h":  {"chop_band_pct": 0.80, "chop_max_range_mult": 2.50},
    "2h":  {"chop_band_pct": 1.10, "chop_max_range_mult": 3.25},
}


def check_and_install_dependencies():
    # NOTE: You’re already using a venv. If something is missing, we’ll tell you what to pip install.
    required = ["alpaca", "pandas", "numpy", "matplotlib"]
    print("Checking dependencies...")
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
            print(f"✓ {pkg} is installed")
        except Exception:
            missing.append(pkg)
    if missing:
        print("\n✗ Missing packages:", ", ".join(missing))
        print("Install with:")
        print("  pip install alpaca-py pandas numpy matplotlib")
        sys.exit(1)


def tf_to_hours(tf: str) -> float:
    if tf.endswith("m"):
        return int(tf[:-1]) / 60.0
    if tf.endswith("h"):
        return float(int(tf[:-1]))
    raise ValueError(f"Unsupported tf: {tf}")


def safe_mean(series):
    s = series.dropna()
    return float(s.mean()) if len(s) else 0.0


def compute_last_hour_metrics(day_data, rth_end_time):
    """
    Last hour = 15:00–16:00 ET (inclusive end bar allowed).
    """
    last_start = time(15, 0)
    last_end = rth_end_time

    lh = day_data[(day_data["Time"] >= last_start) & (day_data["Time"] <= last_end)].copy().reset_index(drop=True)

    if lh.empty:
        return {
            "lh_bars": 0,
            "lh_open": None, "lh_high": None, "lh_low": None, "lh_close": None,
            "lh_range": None, "lh_move_pct": None,
            "lh_volume": None, "lh_vol_vs_day": None,
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


def make_candles_base64(day_df, title=""):
    """
    Simple candlestick chart (no external libs) => base64 PNG.
    """
    if day_df is None or day_df.empty:
        return None

    d = day_df.copy().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    for i, row in d.iterrows():
        o = float(row["Open"])
        h = float(row["High"])
        l = float(row["Low"])
        c = float(row["Close"])

        up = c >= o
        body_low = min(o, c)
        body_high = max(o, c)
        body_h = max(body_high - body_low, 1e-9)

        # wick
        ax.vlines(i, l, h, linewidth=1)

        # body (filled if up, hollow if down)
        rect = Rectangle(
            (i - 0.30, body_low),
            0.60,
            body_h,
            fill=up,
            linewidth=1,
        )
        ax.add_patch(rect)

    times = [t.strftime("%H:%M") for t in d["Datetime"]]
    step = max(1, len(times) // 10)
    ax.set_xticks(list(range(len(d)))[::step])
    ax.set_xticklabels(times[::step], rotation=0)
    ax.set_xlim(-1, len(d))
    ax.set_ylabel("Price")

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


def alpaca_timeframe(tf: str):
    # alpaca-py: TimeFrame(amount, unit)
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    if tf == "15m":
        return TimeFrame(15, TimeFrameUnit.Minute)
    if tf == "30m":
        return TimeFrame(30, TimeFrameUnit.Minute)
    if tf == "1h":
        return TimeFrame(1, TimeFrameUnit.Hour)
    if tf == "2h":
        return TimeFrame(2, TimeFrameUnit.Hour)
    raise ValueError(f"Unsupported tf: {tf}")


def fetch_bars_alpaca(symbol: str, tf: str, start_utc: datetime, end_utc: datetime, feed: str):
    """
    Returns pandas DataFrame with columns:
      Datetime (tz-aware UTC), Open, High, Low, Close, Volume
    """
    import pandas as pd
    from alpaca.data import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest

    api_key = os.getenv("ALPACA_API_KEY")
    api_secret = os.getenv("ALPACA_API_SECRET")
    if not api_key or not api_secret:
        print("✗ Missing Alpaca credentials. Set env vars:")
        print('  export ALPACA_API_KEY="..."')
        print('  export ALPACA_API_SECRET="..."')
        sys.exit(1)

    client = StockHistoricalDataClient(api_key, api_secret)

    req = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=alpaca_timeframe(tf),
        start=start_utc,
        end=end_utc,
        feed=feed,  # "iex" or "sip"
        adjustment="raw",
    )

    bars = client.get_stock_bars(req)

    # alpaca-py returns Bars with .df
    df = bars.df
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # If multiple symbols, alpaca uses multi-index; but we request single symbol.
    df = df.reset_index()

    # Usually columns: ["symbol","timestamp","open","high","low","close","volume","trade_count","vwap"]
    # Normalize:
    if "timestamp" in df.columns:
        df = df.rename(columns={"timestamp": "Datetime"})
    elif "time" in df.columns:
        df = df.rename(columns={"time": "Datetime"})
    else:
        # try common
        for c in df.columns:
            if "time" in c.lower() or "date" in c.lower():
                df = df.rename(columns={c: "Datetime"})
                break

    rename_map = {}
    for lc, std in [("open", "Open"), ("high", "High"), ("low", "Low"), ("close", "Close"), ("volume", "Volume")]:
        if lc in df.columns:
            rename_map[lc] = std
    df = df.rename(columns=rename_map)

    keep = ["Datetime", "Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # ensure datetime tz-aware UTC
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["Datetime", "Open", "High", "Low", "Close"]).sort_values("Datetime").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (SPY, QQQ, IWM, AAPL, etc.)")
    parser.add_argument("--tf", default="1h", choices=SUPPORTED_TF, help="Timeframe: 15m, 30m, 1h, 2h")

    parser.add_argument("--days", type=int, default=180, help="Window size in days (calendar days).")
    parser.add_argument("--offset_days", type=int, default=0,
                        help="How many days back from NOW the window ends. "
                             "Example: --days 30 --offset_days 60 => window ending 60 days ago.")

    parser.add_argument("--volume_quantile", type=float, default=0.75, help="Big volume threshold quantile (default 0.75)")
    parser.add_argument("--move_threshold_pct", type=float, default=0.30, help="FH breakout threshold in percent (default 0.30)")

    parser.add_argument("--chop_band_pct", type=float, default=None, help="Override chop band pct (else auto by timeframe)")
    parser.add_argument("--chop_max_range_mult", type=float, default=None, help="Override chop max bar range mult (else auto by timeframe)")

    parser.add_argument("--alpaca_feed", default=None, help='Alpaca feed: "iex" or "sip" (default uses env ALPACA_DATA_FEED or "iex")')

    args = parser.parse_args()
    symbol = args.symbol.upper()
    tf = args.tf

    defaults = DEFAULTS_BY_TF[tf]
    CHOP_BAND_PCT = defaults["chop_band_pct"] if args.chop_band_pct is None else float(args.chop_band_pct)
    CHOP_MAX_RANGE_MULT = defaults["chop_max_range_mult"] if args.chop_max_range_mult is None else float(args.chop_max_range_mult)

    BAR_HOURS = tf_to_hours(tf)
    move_thr = float(args.move_threshold_pct)

    print("=" * 80)
    print(f"{symbol} FIRST HOUR VOLUME ANALYSIS | TF={tf} | days={args.days} | offset_days={args.offset_days}")
    print("=" * 80)
    print(f"Chop defaults for {tf}: band_pct={defaults['chop_band_pct']}, max_range_mult={defaults['chop_max_range_mult']}")
    print(f"Using chop params: band_pct={CHOP_BAND_PCT}, max_range_mult={CHOP_MAX_RANGE_MULT}")
    print(f"Bar size: {BAR_HOURS} hours")
    if tf == "2h":
        print("⚠️ Note: 2h bars blur the 09:30–10:30 window. 15m/30m recommended for best accuracy.")
    print()

    check_and_install_dependencies()
    print()

    import pandas as pd

    feed = (args.alpaca_feed or os.getenv("ALPACA_DATA_FEED") or "iex").lower()
    if feed not in ("iex", "sip"):
        print('⚠️ alpaca_feed should be "iex" or "sip". Defaulting to "iex".')
        feed = "iex"

    print("=" * 80)
    print(f"STEP 1: Fetching {symbol} data from Alpaca (feed={feed})")
    print("=" * 80)

    # Build the window in UTC
    # End = now - offset_days; Start = end - days
    end_utc = datetime.utcnow() - timedelta(days=int(args.offset_days))
    start_utc = end_utc - timedelta(days=int(args.days))
    print(f"Request window (UTC): {start_utc.isoformat()} -> {end_utc.isoformat()} | tf={tf}")

    df = fetch_bars_alpaca(symbol, tf, start_utc=start_utc, end_utc=end_utc, feed=feed)

    print(f"\n✓ Successfully downloaded {len(df)} bars")
    if df.empty:
        print("✗ No data received from Alpaca.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("STEP 2: Processing and Analyzing Data")
    print("=" * 80)

    # Convert to NY time and filter RTH
    df["Datetime"] = df["Datetime"].dt.tz_convert("America/New_York")
    df = df.sort_values("Datetime").reset_index(drop=True)
    df["Time"] = df["Datetime"].dt.time
    df["Date"] = df["Datetime"].dt.date

    rth_start = time(9, 30)
    rth_end = time(16, 0)

    df = df[(df["Time"] >= rth_start) & (df["Time"] <= rth_end)].copy()

    # First hour window 09:30–10:30
    fh_end = time(10, 30)
    df["IsFirstHour"] = df["Time"].apply(lambda t: rth_start <= t < fh_end)

    # First hour volume threshold across ALL days (with FH bars)
    print("\nCalculating first-hour volume threshold...")
    first_hour_volume = df.loc[df["IsFirstHour"]].groupby("Date")["Volume"].sum()
    if len(first_hour_volume) == 0:
        print("✗ No first-hour bars found in 09:30–10:30 window. Try 15m/30m.")
        sys.exit(1)

    big_volume_threshold = float(first_hour_volume.quantile(args.volume_quantile))
    print(f"✓ Big volume threshold (q={args.volume_quantile:.2f}): {big_volume_threshold:,.0f}")

    print("\nAnalyzing days...")
    results = {"selloffs": [], "buyups": [], "chop_days": [], "other_days": []}
    charts_by_date = {}

    # iterate each trading date
    for date in sorted(df["Date"].unique()):
        day_data = df[df["Date"] == date].sort_values("Datetime").reset_index(drop=True)
        if len(day_data) < 2:
            continue

        fh = day_data[day_data["IsFirstHour"]].reset_index(drop=True)
        if fh.empty:
            continue

        fh_volume = float(fh["Volume"].sum())
        day_volume = float(day_data["Volume"].sum()) if len(day_data) else 0.0
        fh_vol_vs_day = (fh_volume / day_volume) if day_volume > 0 else None

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

        is_big_vol = fh_volume >= big_volume_threshold

        # Always store chart for the day (so you can click charts everywhere)
        chart_b64 = make_candles_base64(day_data, title=f"{symbol} {tf} | {date} (RTH)")
        charts_by_date[str(date)] = chart_b64

        # common fields for every bucket/day
        common = {
            "date": str(date),
            "tf": tf,

            "fh_move_pct": round(fh_move_pct, 4),
            "fh_open": round(fh_open, 4),
            "fh_close": round(fh_close, 4),
            "fh_high": round(fh_high, 4),
            "fh_low": round(fh_low, 4),

            "fh_volume": round(fh_volume, 2),
            "fh_vol_vs_day": None if fh_vol_vs_day is None else round(fh_vol_vs_day, 6),

            "final_abs_high": None if final_abs_high is None else round(final_abs_high, 4),
            "hours_to_final_abs_high": hours_to_final_abs_high,
            "extension_points_above_fh_high": ext_above,

            "final_abs_low": None if final_abs_low is None else round(final_abs_low, 4),
            "hours_to_final_abs_low": hours_to_final_abs_low,
            "extension_points_below_fh_low": ext_below,

            "is_big_vol_day": bool(is_big_vol),

            # last hour reaction
            **lh_metrics,
        }

        # ------------------------
        # OTHER / NORMAL days (not big volume)
        # ------------------------
        if not is_big_vol:
            made_new_high = (final_abs_high is not None) and (final_abs_high > fh_high)
            made_new_low = (final_abs_low is not None) and (final_abs_low < fh_low)

            results["other_days"].append({
                **common,
                "made_new_high_after_fh": bool(made_new_high),
                "made_new_low_after_fh": bool(made_new_low),
            })
            continue

        # BigVol day -> classify into sell/buy/chop
        # ---------------- SELL-OFF ----------------
        if fh_move_pct <= -move_thr:
            fh_range = fh_open - fh_low

            continuation_found = (final_abs_low is not None) and (final_abs_low < fh_low)
            failed = not continuation_found
            continuation_hours = hours_to_final_abs_low if continuation_found else 0.0

            bounce_25_hours = None
            bounce_50_hours = None
            bounce_25_target = None
            bounce_50_target = None
            bounce_25_hit_price = None
            bounce_50_hit_price = None
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

                    # bounce targets from current running bottom
                    t25 = abs_bottom + 0.25 * fh_range
                    t50 = abs_bottom + 0.50 * fh_range

                    if bounce_25_hours is None and bar_high >= t25:
                        bounce_25_hours = round(elapsed_hours, 4)
                        bounce_25_target = round(t25, 4)
                        bounce_25_hit_price = round(bar_high, 4)

                    if bounce_50_hours is None and bar_high >= t50:
                        bounce_50_hours = round(elapsed_hours, 4)
                        bounce_50_target = round(t50, 4)
                        bounce_50_hit_price = round(bar_high, 4)

                    # Friendly chop until 25% bounce occurs
                    if bounce_25_hours is None:
                        in_band = bar_high <= (abs_bottom + CHOP_BAND_PCT * fh_range)
                        not_huge = bar_range <= (CHOP_MAX_RANGE_MULT * fh_range)
                        if (not made_new_low) and in_band and not_huge:
                            chop_hours += BAR_HOURS

            results["selloffs"].append({
                **common,
                "first_hour_range": None if fh_range is None else round(fh_range, 4),

                "continuation_hours": continuation_hours,
                "chop_hours": round(chop_hours, 4),

                "bounce_25_hours": bounce_25_hours,
                "bounce_25_target_price": bounce_25_target,
                "bounce_25_hit_price": bounce_25_hit_price,

                "bounce_50_hours": bounce_50_hours,
                "bounce_50_target_price": bounce_50_target,
                "bounce_50_hit_price": bounce_50_hit_price,

                "failed_selloff": bool(failed),
            })

        # ---------------- BUY-UP ----------------
        elif fh_move_pct >= move_thr:
            fh_range = fh_high - fh_open

            continuation_found = (final_abs_high is not None) and (final_abs_high > fh_high)
            failed = not continuation_found
            continuation_hours = hours_to_final_abs_high if continuation_found else 0.0

            pullback_25_hours = None
            pullback_50_hours = None
            pullback_25_target = None
            pullback_50_target = None
            pullback_25_hit_price = None
            pullback_50_hit_price = None
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

                    # pullback targets from current running top
                    t25 = abs_top - 0.25 * fh_range
                    t50 = abs_top - 0.50 * fh_range

                    if pullback_25_hours is None and bar_low <= t25:
                        pullback_25_hours = round(elapsed_hours, 4)
                        pullback_25_target = round(t25, 4)
                        pullback_25_hit_price = round(bar_low, 4)

                    if pullback_50_hours is None and bar_low <= t50:
                        pullback_50_hours = round(elapsed_hours, 4)
                        pullback_50_target = round(t50, 4)
                        pullback_50_hit_price = round(bar_low, 4)

                    # Friendly chop until 25% pullback occurs
                    if pullback_25_hours is None:
                        in_band = bar_low >= (abs_top - CHOP_BAND_PCT * fh_range)
                        not_huge = bar_range <= (CHOP_MAX_RANGE_MULT * fh_range)
                        if (not made_new_high) and in_band and not_huge:
                            chop_hours += BAR_HOURS

            results["buyups"].append({
                **common,
                "first_hour_range": None if fh_range is None else round(fh_range, 4),

                "continuation_hours": continuation_hours,
                "chop_hours": round(chop_hours, 4),

                "pullback_25_hours": pullback_25_hours,
                "pullback_25_target_price": pullback_25_target,
                "pullback_25_hit_price": pullback_25_hit_price,

                "pullback_50_hours": pullback_50_hours,
                "pullback_50_target_price": pullback_50_target,
                "pullback_50_hit_price": pullback_50_hit_price,

                "failed_buyup": bool(failed),
            })

        # ---------------- BIGVOL CHOP / NO BREAKOUT ----------------
        else:
            made_new_high = (final_abs_high is not None) and (final_abs_high > fh_high)
            made_new_low = (final_abs_low is not None) and (final_abs_low < fh_low)

            results["chop_days"].append({
                **common,
                "made_new_high_after_fh": bool(made_new_high),
                "made_new_low_after_fh": bool(made_new_low),
            })

    # ---------------- Save outputs ----------------
    import pandas as pd

    sell_df = pd.DataFrame(results["selloffs"])
    buy_df = pd.DataFrame(results["buyups"])
    chop_df = pd.DataFrame(results["chop_days"])
    other_df = pd.DataFrame(results["other_days"])

    summary = {
        "meta": {
            "symbol": symbol,
            "tf": tf,
            "bar_hours": BAR_HOURS,
            "days": args.days,
            "offset_days": args.offset_days,
            "volume_quantile": args.volume_quantile,
            "big_volume_threshold": big_volume_threshold,
            "move_threshold_pct": move_thr,
            "chop_band_pct": CHOP_BAND_PCT,
            "chop_max_range_mult": CHOP_MAX_RANGE_MULT,
            "provider": "alpaca",
            "alpaca_feed": feed,
            "window_utc_start": start_utc.isoformat(),
            "window_utc_end": end_utc.isoformat(),
        },
        "selloffs": {},
        "buyups": {},
        "chop_days": {},
        "other_days": {},
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
            "new_high_rate": float(chop_df["made_new_high_after_fh"].mean()) if "made_new_high_after_fh" in chop_df.columns else 0.0,
            "new_low_rate": float(chop_df["made_new_low_after_fh"].mean()) if "made_new_low_after_fh" in chop_df.columns else 0.0,
            "avg_ext_above_fh_high": safe_mean(chop_df["extension_points_above_fh_high"]),
            "avg_ext_below_fh_low": safe_mean(chop_df["extension_points_below_fh_low"]),
            "avg_hours_to_final_high": safe_mean(chop_df["hours_to_final_abs_high"]),
            "avg_hours_to_final_low": safe_mean(chop_df["hours_to_final_abs_low"]),
            "avg_lh_move_pct": safe_mean(chop_df["lh_move_pct"]) if "lh_move_pct" in chop_df.columns else 0.0,
        }

    if len(other_df):
        summary["other_days"] = {
            "total_count": int(len(other_df)),
            "new_high_rate": float(other_df["made_new_high_after_fh"].mean()) if "made_new_high_after_fh" in other_df.columns else 0.0,
            "new_low_rate": float(other_df["made_new_low_after_fh"].mean()) if "made_new_low_after_fh" in other_df.columns else 0.0,
            "avg_ext_above_fh_high": safe_mean(other_df["extension_points_above_fh_high"]),
            "avg_ext_below_fh_low": safe_mean(other_df["extension_points_below_fh_low"]),
            "avg_hours_to_final_high": safe_mean(other_df["hours_to_final_abs_high"]),
            "avg_hours_to_final_low": safe_mean(other_df["hours_to_final_abs_low"]),
            "avg_lh_move_pct": safe_mean(other_df["lh_move_pct"]) if "lh_move_pct" in other_df.columns else 0.0,
        }

    sell_csv = f"{symbol}_{tf}_selloffs_detail.csv"
    buy_csv = f"{symbol}_{tf}_buyups_detail.csv"
    chop_csv = f"{symbol}_{tf}_chop_days_detail.csv"
    other_csv = f"{symbol}_{tf}_other_days_detail.csv"
    sum_json = f"{symbol}_{tf}_summary_stats.json"
    charts_json = f"{symbol}_{tf}_charts.json"

    sell_df.to_csv(sell_csv, index=False)
    buy_df.to_csv(buy_csv, index=False)
    chop_df.to_csv(chop_csv, index=False)
    other_df.to_csv(other_csv, index=False)

    with open(sum_json, "w") as f:
        json.dump(summary, f, indent=2)

    with open(charts_json, "w") as f:
        json.dump(charts_by_date, f)

    print("\n" + "=" * 80)
    print("STEP 3: Saved Outputs")
    print("=" * 80)
    print(f"✓ {sell_csv}")
    print(f"✓ {buy_csv}")
    print(f"✓ {chop_csv}")
    print(f"✓ {other_csv}")
    print(f"✓ {sum_json}")
    print(f"✓ {charts_json}  (base64 candle charts)")

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

