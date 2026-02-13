#!/usr/bin/env python3
"""
run_complete_analysis.py

First Hour Volume Analysis - Complete Package (Multi-timeframe + Sell/Buy/BigVol Chop + Other Days + Last Hour + Candle Cards + Offset Window)

✅ NY RTH time (America/New_York)
✅ RTH filter: 09:30–16:00 ET
✅ First hour window: 09:30–10:30 ET
✅ Timeframes: 15m, 30m, 1h, 2h
✅ Big first-hour volume day: first-hour volume >= quantile threshold (default q=0.75)

Buckets:
A) Sell-off (big-volume days): FH move % <= -move_threshold_pct
B) Buy-up  (big-volume days): FH move % >= +move_threshold_pct
C) BigVol Chop/No Breakout: abs(FH move %) < move_threshold_pct  (still big-volume)
D) Other/Normal days: NOT big-volume days (FH vol < threshold), but still has FH bars

Metrics:
- final_abs_high/low after first hour
- time-to-final-abs-high/low in HOURS (timeframe-aware)
- extension points beyond FH high/low
- continuation time (to final abs extreme if it extends beyond FH)
- bounce/pullback 25%/50% from RUNNING extremes (sell/buy)
- friendly chop hours (sell/buy) before 25% bounce/pullback
- last hour reaction (15:00–16:00): open/high/low/close, range, move %, volume, vol vs day
- candle chart base64 per date for dashboard “View Chart” modal

NEW:
✅ --offset_days to “go back in time”
   - We download enough recent bars using Yahoo period, then slice to your requested window.
   - For 15m/30m: Yahoo generally only supports ~60 days total (days + offset). Script auto-trims.

Outputs:
- SYMBOL_TF_selloffs_detail.csv
- SYMBOL_TF_buyups_detail.csv
- SYMBOL_TF_chop_days_detail.csv
- SYMBOL_TF_other_days_detail.csv
- SYMBOL_TF_summary_stats.json
- SYMBOL_TF_charts.json
- SYMBOL_TF_analysis_dashboard.html (via create_dashboard.py)

Usage examples:
  # latest window
  python run_complete_analysis.py --symbol SPY --tf 1h --days 180

  # 60-day window ending 30 days ago (only safe for 1h/2h; 15m/30m may trim)
  python run_complete_analysis.py --symbol SPY --tf 1h --days 60 --offset_days 30

  # 15m/30m max total history usually ~60d (days+offset), script trims if needed
  python run_complete_analysis.py --symbol SPY --tf 30m --days 40 --offset_days 10
"""

import argparse
import subprocess
import sys
import base64
from io import BytesIO
from datetime import datetime, timedelta, time

SUPPORTED_TF = ["15m", "30m", "1h", "2h"]

DEFAULTS_BY_TF = {
    "15m": {"chop_band_pct": 0.35, "chop_max_range_mult": 1.50},
    "30m": {"chop_band_pct": 0.45, "chop_max_range_mult": 1.75},
    "1h":  {"chop_band_pct": 0.80, "chop_max_range_mult": 2.50},
    "2h":  {"chop_band_pct": 1.10, "chop_max_range_mult": 3.25},
}

# Yahoo practical intraday windows (yfinance/Yahoo; can vary)
INTRADAY_MAX_DAYS = {
    "15m": 60,
    "30m": 60,
    "1h": 730,
    "2h": 730,
}


def check_and_install_dependencies():
    required = ["yfinance", "pandas", "numpy", "matplotlib"]
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
    # yfinance intraday often arrives tz-naive but is UTC-like
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        dt = dt.dt.tz_convert("America/New_York")
    df["Datetime"] = dt
    return df


def compute_last_hour_metrics(day_data, rth_end_time):
    """
    Last hour defined as 15:00–16:00 ET (inclusive of 16:00 bar if present).
    """
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
    Simple candlestick chart using matplotlib only (no mplfinance dependency).
    """
    if day_df is None or day_df.empty:
        return None

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

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

        # Wick
        ax.vlines(i, l, h, linewidth=1)

        # Body: filled if up, outline if down
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
    ax.set_xticks(list(range(0, len(times), step)))
    ax.set_xticklabels([times[i] for i in range(0, len(times), step)], rotation=0)

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


def fetch_yahoo_window(symbol: str, tf: str, days: int, offset_days: int):
    """
    Yahoo-safe fetching:
    - We fetch a RECENT chunk using period=... (avoids Yahoo start/end errors)
    - Then slice to the requested [start_dt, end_dt] window in NY time after conversion

    For 15m/30m: total needed (days+offset_days) is capped ~60 days (Yahoo limit).
    If you ask more, we auto-trim `days` to fit.
    """
    import yfinance as yf

    max_total = INTRADAY_MAX_DAYS.get(tf, days)
    total_needed = days + offset_days

    if total_needed > max_total:
        # trim days to fit
        new_days = max(1, max_total - offset_days)
        print(f"⚠️ Yahoo limit: tf={tf} total(days+offset)≈{max_total}d. Trimming days {days}->{new_days} to fit offset_days={offset_days}.")
        days = new_days
        total_needed = days + offset_days

    # window endpoints
    end_dt = datetime.now() - timedelta(days=offset_days)
    start_dt = end_dt - timedelta(days=days)

    # fetch a bit extra buffer (helps slicing / missing bars)
    buffer_days = 5
    fetch_days = min(max_total, total_needed + buffer_days)

    # Yahoo sometimes fails at exactly 60d; use 59d for 15m/30m for reliability
    if tf in ("15m", "30m") and fetch_days >= 60:
        fetch_days = 59

    period = f"{int(fetch_days)}d"
    print(f"Downloading {symbol} | interval={tf} | period={period} ...")

    df = yf.download(
        symbol,
        interval=tf,
        period=period,
        auto_adjust=False,
        prepost=False,
        progress=True,
        threads=True,
    )

    return df, start_dt, end_dt, days


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol (SPY, QQQ, IWM, AAPL, etc.)")
    parser.add_argument("--tf", default="1h", choices=SUPPORTED_TF, help="Timeframe: 15m, 30m, 1h, 2h")
    parser.add_argument("--days", type=int, default=180, help="Window length in days (default 180)")
    parser.add_argument("--offset_days", type=int, default=0, help="Shift end of window back by N days (0=end today)")
    parser.add_argument("--volume_quantile", type=float, default=0.75, help="Big volume threshold quantile (default 0.75)")
    parser.add_argument("--move_threshold_pct", type=float, default=0.30, help="FH breakout threshold in percent (default 0.30)")
    parser.add_argument("--chop_band_pct", type=float, default=None, help="Override chop band pct (else auto by timeframe)")
    parser.add_argument("--chop_max_range_mult", type=float, default=None, help="Override chop max bar range mult (else auto by timeframe)")
    args = parser.parse_args()

    symbol = args.symbol.upper()
    tf = args.tf
    BAR_HOURS = tf_to_hours(tf)
    move_thr = float(args.move_threshold_pct)

    defaults = DEFAULTS_BY_TF[tf]
    CHOP_BAND_PCT = defaults["chop_band_pct"] if args.chop_band_pct is None else float(args.chop_band_pct)
    CHOP_MAX_RANGE_MULT = defaults["chop_max_range_mult"] if args.chop_max_range_mult is None else float(args.chop_max_range_mult)

    print("=" * 80)
    print(f"{symbol} FIRST HOUR VOLUME ANALYSIS | TF={tf} | Window={args.days}d | Offset={args.offset_days}d")
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
    import numpy as np
    import json

    print("=" * 80)
    print(f"STEP 1: Fetching {symbol} data from Yahoo Finance")
    print("=" * 80)

    df_raw, start_dt, end_dt, effective_days = fetch_yahoo_window(symbol, tf, args.days, args.offset_days)
    print(f"\n✓ Downloaded {len(df_raw)} bars (pre-slice). Target window: {start_dt.date()} -> {end_dt.date()} (effective_days={effective_days})")

    if df_raw is None or len(df_raw) == 0:
        print("✗ No data received.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("STEP 2: Processing and Analyzing Data")
    print("=" * 80)

    df = df_raw.reset_index()
    df = flatten_columns_if_needed(df)
    df = ensure_datetime_column(df)

    # Normalize OHLCV
    for base in ["Open", "High", "Low", "Close", "Volume"]:
        src = pick_col(df, base)
        if src != base:
            df[base] = df[src]

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Convert to NY time and slice to [start_dt, end_dt]
    df = convert_to_ny_time(df)
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Slice to requested window
    # (start_dt/end_dt are naive; interpret them as "local now - offset" bounds in naive; apply as NY-aware bounds)
    start_bound = pd.Timestamp(start_dt).tz_localize("America/New_York")
    end_bound = pd.Timestamp(end_dt).tz_localize("America/New_York")
    df = df[(df["Datetime"] >= start_bound) & (df["Datetime"] <= end_bound)].copy()

    if df.empty:
        print("✗ After slicing to requested window, no bars remain.")
        print("  Try reducing --offset_days or --days (especially for 15m/30m).")
        sys.exit(1)

    # RTH filter + first hour window
    df["Time"] = df["Datetime"].dt.time
    df["Date"] = df["Datetime"].dt.date

    rth_start = time(9, 30)
    rth_end = time(16, 0)
    df = df[(df["Time"] >= rth_start) & (df["Time"] <= rth_end)].copy()

    fh_end = time(10, 30)
    df["IsFirstHour"] = df["Time"].apply(lambda t: rth_start <= t < fh_end)

    # First-hour volume threshold
    print("\nCalculating first-hour volume threshold...")
    first_hour_volume = df.loc[df["IsFirstHour"]].groupby("Date")["Volume"].sum()
    if len(first_hour_volume) == 0:
        print("✗ No first-hour bars found in 09:30–10:30 window (after RTH filter).")
        print("  For best results: use 15m/30m. For 1h/2h, Yahoo bar alignment may miss the exact window some days.")
        sys.exit(1)

    big_volume_threshold = float(first_hour_volume.quantile(args.volume_quantile))
    print(f"✓ Big volume threshold (q={args.volume_quantile:.2f}): {big_volume_threshold:,.0f}")

    # Results and charts
    results = {"selloffs": [], "buyups": [], "chop_days": [], "other_days": []}
    charts_by_date = {}  # <-- FIX: must be outside loop

    print("\nAnalyzing days...")

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

        fh_open = float(fh.iloc[0]["Open"])
        fh_high = float(fh["High"].max())
        fh_low = float(fh["Low"].min())
        fh_close = float(fh.iloc[-1]["Close"])
        fh_move_pct = ((fh_close - fh_open) / fh_open) * 100.0 if fh_open else 0.0

        after_fh = day_data[~day_data["IsFirstHour"]].reset_index(drop=True)

        # Candle chart for this day (we store for ALL days that make it into any bucket)
        # We'll decide whether to store based on whether it's big-volume or other-day.
        def store_chart():
            if str(date) not in charts_by_date:
                charts_by_date[str(date)] = make_candles_base64(
                    day_data,
                    title=f"{symbol} {tf} | {date} (RTH)"
                )

        # Last hour metrics
        lh_metrics = compute_last_hour_metrics(day_data, rth_end_time=rth_end)

        # final abs extremes AFTER first hour
        final_abs_high = None
        hours_to_final_abs_high = None
        ext_above = None

        final_abs_low = None
        hours_to_final_abs_low = None
        ext_below = None

        if not after_fh.empty:
            idx_high = int(np.argmax(after_fh["High"].values))
            idx_low = int(np.argmin(after_fh["Low"].values))

            final_abs_high = float(after_fh["High"].iloc[idx_high])
            final_abs_low = float(after_fh["Low"].iloc[idx_low])

            hours_to_final_abs_high = round((idx_high + 1) * BAR_HOURS, 4)
            hours_to_final_abs_low = round((idx_low + 1) * BAR_HOURS, 4)

            ext_above = round(final_abs_high - fh_high, 4)
            ext_below = round(fh_low - final_abs_low, 4)

        # ---------------------------
        # BIG VOLUME DAY?
        # ---------------------------
        is_big_vol = fh_volume >= big_volume_threshold

        if not is_big_vol:
            # OTHER / NORMAL DAY bucket (still compute same metrics)
            store_chart()

            made_new_high = (final_abs_high is not None) and (final_abs_high > fh_high)
            made_new_low = (final_abs_low is not None) and (final_abs_low < fh_low)

            results["other_days"].append({
                "date": str(date),
                "tf": tf,
                "fh_move_pct": round(fh_move_pct, 4),
                "fh_volume": round(fh_volume, 2),
                "fh_vol_vs_day": None if fh_vol_vs_day is None else round(fh_vol_vs_day, 4),

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

                **lh_metrics,
            })
            continue

        # Big-volume day => it is an “event day”
        store_chart()

        # ---------------- SELL-OFF ----------------
        if fh_move_pct <= -move_thr:
            fh_range = fh_open - fh_low

            continuation_found = (final_abs_low is not None) and (final_abs_low < fh_low)
            failed = not continuation_found
            continuation_hours = hours_to_final_abs_low if continuation_found else 0.0

            bounce_25_hours = None
            bounce_50_hours = None
            bounce_25_level = None
            bounce_50_level = None
            bounce_25_from_fh_low_pts = None
            bounce_50_from_fh_low_pts = None

            chop_hours = 0.0

            if not after_fh.empty and fh_range > 0:
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

                    # Trigger levels based on running bottom at this moment
                    lvl25 = abs_bottom + 0.25 * fh_range
                    lvl50 = abs_bottom + 0.50 * fh_range

                    if bounce_25_hours is None and recovery_pct >= 25:
                        bounce_25_hours = round(elapsed_hours, 4)
                        bounce_25_level = round(lvl25, 4)
                        bounce_25_from_fh_low_pts = round(lvl25 - fh_low, 4)

                    if bounce_50_hours is None and recovery_pct >= 50:
                        bounce_50_hours = round(elapsed_hours, 4)
                        bounce_50_level = round(lvl50, 4)
                        bounce_50_from_fh_low_pts = round(lvl50 - fh_low, 4)

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
                "fh_volume": round(fh_volume, 2),
                "fh_vol_vs_day": None if fh_vol_vs_day is None else round(fh_vol_vs_day, 4),

                "first_hour_range": round(fh_range, 4),

                "final_abs_low": None if final_abs_low is None else round(final_abs_low, 4),
                "hours_to_final_abs_low": hours_to_final_abs_low,
                "extension_points_below_fh_low": ext_below,

                "continuation_hours": continuation_hours,
                "chop_hours": round(chop_hours, 4),

                "bounce_25_hours": bounce_25_hours,
                "bounce_25_level": bounce_25_level,
                "bounce_25_from_fh_low_pts": bounce_25_from_fh_low_pts,

                "bounce_50_hours": bounce_50_hours,
                "bounce_50_level": bounce_50_level,
                "bounce_50_from_fh_low_pts": bounce_50_from_fh_low_pts,

                "failed_selloff": bool(failed),

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
            pullback_25_level = None
            pullback_50_level = None
            pullback_25_from_fh_high_pts = None
            pullback_50_from_fh_high_pts = None

            chop_hours = 0.0

            if not after_fh.empty and fh_range > 0:
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

                    # Trigger levels based on running top at this moment
                    lvl25 = abs_top - 0.25 * fh_range
                    lvl50 = abs_top - 0.50 * fh_range

                    if pullback_25_hours is None and pullback_pct >= 25:
                        pullback_25_hours = round(elapsed_hours, 4)
                        pullback_25_level = round(lvl25, 4)
                        pullback_25_from_fh_high_pts = round(fh_high - lvl25, 4)

                    if pullback_50_hours is None and pullback_pct >= 50:
                        pullback_50_hours = round(elapsed_hours, 4)
                        pullback_50_level = round(lvl50, 4)
                        pullback_50_from_fh_high_pts = round(fh_high - lvl50, 4)

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
                "fh_volume": round(fh_volume, 2),
                "fh_vol_vs_day": None if fh_vol_vs_day is None else round(fh_vol_vs_day, 4),

                "first_hour_range": round(fh_range, 4),

                "final_abs_high": None if final_abs_high is None else round(final_abs_high, 4),
                "hours_to_final_abs_high": hours_to_final_abs_high,
                "extension_points_above_fh_high": ext_above,

                "continuation_hours": continuation_hours,
                "chop_hours": round(chop_hours, 4),

                "pullback_25_hours": pullback_25_hours,
                "pullback_25_level": pullback_25_level,
                "pullback_25_from_fh_high_pts": pullback_25_from_fh_high_pts,

                "pullback_50_hours": pullback_50_hours,
                "pullback_50_level": pullback_50_level,
                "pullback_50_from_fh_high_pts": pullback_50_from_fh_high_pts,

                "failed_buyup": bool(failed),

                **lh_metrics,
            })

        # ---------------- BIGVOL CHOP / NO BREAKOUT ----------------
        else:
            made_new_high = (final_abs_high is not None) and (final_abs_high > fh_high)
            made_new_low = (final_abs_low is not None) and (final_abs_low < fh_low)

            results["chop_days"].append({
                "date": str(date),
                "tf": tf,

                "fh_move_pct": round(fh_move_pct, 4),
                "fh_volume": round(fh_volume, 2),
                "fh_vol_vs_day": None if fh_vol_vs_day is None else round(fh_vol_vs_day, 4),

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

                **lh_metrics,
            })

    # Write charts json
    import json
    charts_json = f"{symbol}_{tf}_charts.json"
    with open(charts_json, "w") as f:
        json.dump(charts_by_date, f)
    print(f"\n✓ Saved charts: {charts_json} (count={len(charts_by_date)})")

    # Build dataframes + summary
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
            "days_requested": args.days,
            "days_effective": effective_days,
            "offset_days": args.offset_days,
            "window_start": str(start_dt.date()),
            "window_end": str(end_dt.date()),
            "volume_quantile": args.volume_quantile,
            "big_volume_threshold": big_volume_threshold,
            "move_threshold_pct": move_thr,
            "chop_band_pct": CHOP_BAND_PCT,
            "chop_max_range_mult": CHOP_MAX_RANGE_MULT,
        },
        "selloffs": {},
        "buyups": {},
        "chop_days": {},
        "other_days": {},
    }

    if len(sell_df):
        summary["selloffs"] = {
            "total_count": int(len(sell_df)),
            "continuation_rate": float((sell_df["continuation_hours"] > 0).mean()) if "continuation_hours" in sell_df.columns else 0.0,
            "avg_first_hour_range": safe_mean(sell_df["first_hour_range"]) if "first_hour_range" in sell_df.columns else 0.0,
            "avg_extension_below_fh_low": safe_mean(sell_df["extension_points_below_fh_low"]) if "extension_points_below_fh_low" in sell_df.columns else 0.0,
            "avg_hours_to_final_abs_low": safe_mean(sell_df["hours_to_final_abs_low"]) if "hours_to_final_abs_low" in sell_df.columns else 0.0,
            "avg_chop_hours": safe_mean(sell_df["chop_hours"]) if "chop_hours" in sell_df.columns else 0.0,
            "chop_rate": float((sell_df["chop_hours"] > 0).mean()) if "chop_hours" in sell_df.columns else 0.0,
            "bounce_25_rate": float(sell_df["bounce_25_hours"].notna().mean()) if "bounce_25_hours" in sell_df.columns else 0.0,
            "bounce_50_rate": float(sell_df["bounce_50_hours"].notna().mean()) if "bounce_50_hours" in sell_df.columns else 0.0,
            "avg_lh_move_pct": safe_mean(sell_df["lh_move_pct"]) if "lh_move_pct" in sell_df.columns else 0.0,
        }

    if len(buy_df):
        summary["buyups"] = {
            "total_count": int(len(buy_df)),
            "continuation_rate": float((buy_df["continuation_hours"] > 0).mean()) if "continuation_hours" in buy_df.columns else 0.0,
            "avg_first_hour_range": safe_mean(buy_df["first_hour_range"]) if "first_hour_range" in buy_df.columns else 0.0,
            "avg_extension_above_fh_high": safe_mean(buy_df["extension_points_above_fh_high"]) if "extension_points_above_fh_high" in buy_df.columns else 0.0,
            "avg_hours_to_final_abs_high": safe_mean(buy_df["hours_to_final_abs_high"]) if "hours_to_final_abs_high" in buy_df.columns else 0.0,
            "avg_chop_hours": safe_mean(buy_df["chop_hours"]) if "chop_hours" in buy_df.columns else 0.0,
            "chop_rate": float((buy_df["chop_hours"] > 0).mean()) if "chop_hours" in buy_df.columns else 0.0,
            "pullback_25_rate": float(buy_df["pullback_25_hours"].notna().mean()) if "pullback_25_hours" in buy_df.columns else 0.0,
            "pullback_50_rate": float(buy_df["pullback_50_hours"].notna().mean()) if "pullback_50_hours" in buy_df.columns else 0.0,
            "avg_lh_move_pct": safe_mean(buy_df["lh_move_pct"]) if "lh_move_pct" in buy_df.columns else 0.0,
        }

    if len(chop_df):
        summary["chop_days"] = {
            "total_count": int(len(chop_df)),
            "new_high_rate": float(chop_df["made_new_high_after_fh"].mean()) if "made_new_high_after_fh" in chop_df.columns else 0.0,
            "new_low_rate": float(chop_df["made_new_low_after_fh"].mean()) if "made_new_low_after_fh" in chop_df.columns else 0.0,
            "avg_ext_above_fh_high": safe_mean(chop_df["extension_points_above_fh_high"]) if "extension_points_above_fh_high" in chop_df.columns else 0.0,
            "avg_ext_below_fh_low": safe_mean(chop_df["extension_points_below_fh_low"]) if "extension_points_below_fh_low" in chop_df.columns else 0.0,
            "avg_hours_to_final_high": safe_mean(chop_df["hours_to_final_abs_high"]) if "hours_to_final_abs_high" in chop_df.columns else 0.0,
            "avg_hours_to_final_low": safe_mean(chop_df["hours_to_final_abs_low"]) if "hours_to_final_abs_low" in chop_df.columns else 0.0,
            "avg_lh_move_pct": safe_mean(chop_df["lh_move_pct"]) if "lh_move_pct" in chop_df.columns else 0.0,
        }

    if len(other_df):
        summary["other_days"] = {
            "total_count": int(len(other_df)),
            "new_high_rate": float(other_df["made_new_high_after_fh"].mean()) if "made_new_high_after_fh" in other_df.columns else 0.0,
            "new_low_rate": float(other_df["made_new_low_after_fh"].mean()) if "made_new_low_after_fh" in other_df.columns else 0.0,
            "avg_ext_above_fh_high": safe_mean(other_df["extension_points_above_fh_high"]) if "extension_points_above_fh_high" in other_df.columns else 0.0,
            "avg_ext_below_fh_low": safe_mean(other_df["extension_points_below_fh_low"]) if "extension_points_below_fh_low" in other_df.columns else 0.0,
            "avg_hours_to_final_high": safe_mean(other_df["hours_to_final_abs_high"]) if "hours_to_final_abs_high" in other_df.columns else 0.0,
            "avg_hours_to_final_low": safe_mean(other_df["hours_to_final_abs_low"]) if "hours_to_final_abs_low" in other_df.columns else 0.0,
            "avg_lh_move_pct": safe_mean(other_df["lh_move_pct"]) if "lh_move_pct" in other_df.columns else 0.0,
        }

    # Save outputs
    sell_csv = f"{symbol}_{tf}_selloffs_detail.csv"
    buy_csv = f"{symbol}_{tf}_buyups_detail.csv"
    chop_csv = f"{symbol}_{tf}_chop_days_detail.csv"
    other_csv = f"{symbol}_{tf}_other_days_detail.csv"
    sum_json = f"{symbol}_{tf}_summary_stats.json"

    sell_df.to_csv(sell_csv, index=False)
    buy_df.to_csv(buy_csv, index=False)
    chop_df.to_csv(chop_csv, index=False)
    other_df.to_csv(other_csv, index=False)

    with open(sum_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 80)
    print("STEP 3: Saved Outputs")
    print("=" * 80)
    print(f"✓ {sell_csv}  ({len(sell_df)})")
    print(f"✓ {buy_csv}   ({len(buy_df)})")
    print(f"✓ {chop_csv}  ({len(chop_df)})")
    print(f"✓ {other_csv} ({len(other_df)})")
    print(f"✓ {sum_json}")
    print(f"✓ {charts_json}")

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
