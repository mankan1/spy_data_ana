#!/usr/bin/env python3
"""
First Hour Volume Analysis - Complete Package
(Multi-timeframe + Chop Days + Other Days + Last Hour Reaction + Day Charts)

Adds:
- Last-hour metrics everywhere: lh_move_pct, lh_range, lh_volume, lh_vol_vs_day
- Bounce/Pullback time since first hour + price level where threshold first met
- Other/Normal days: includes both "bounce-from-FH-low" and "pullback-from-FH-high" style metrics
"""

import argparse
import subprocess
import sys
import base64
from io import BytesIO

SUPPORTED_TF = ["15m", "30m", "1h", "2h"]

DEFAULTS_BY_TF = {
    "15m": {"chop_band_pct": 0.35, "chop_max_range_mult": 1.50},
    "30m": {"chop_band_pct": 0.45, "chop_max_range_mult": 1.75},
    "1h":  {"chop_band_pct": 0.80, "chop_max_range_mult": 2.50},
    "2h":  {"chop_band_pct": 1.10, "chop_max_range_mult": 3.25},
}

INTRADAY_MAX_DAYS = {"15m": 60, "30m": 60, "1h": 730, "2h": 730}


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
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        dt = dt.dt.tz_convert("America/New_York")
    df["Datetime"] = dt
    return df


def compute_last_hour_metrics(day_data, rth_end_time):
    from datetime import time
    last_start = time(15, 0)
    last_end = rth_end_time

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
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    if day_df is None or day_df.empty:
        return None

    d = day_df.copy().reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 4.2))
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

        ax.vlines(i, l, h, linewidth=1)
        rect = Rectangle((i - 0.30, body_low), 0.60, body_h, fill=up, linewidth=1)
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


def scan_sell_bounce(after_fh, fh_range_down, fh_low, bar_hours, band_pct, max_range_mult):
    """
    SELL-side: measure bounce from running bottom (starting fh_low).
    Return:
      bounce_25_hours, bounce_25_level, bounce_50_hours, bounce_50_level, chop_hours
    Where *level* is bar_high at first threshold hit.
    """
    bounce_25_hours = None
    bounce_50_hours = None
    bounce_25_level = None
    bounce_50_level = None
    chop_hours = 0.0

    if after_fh.empty or fh_range_down <= 0:
        return bounce_25_hours, bounce_25_level, bounce_50_hours, bounce_50_level, chop_hours

    abs_bottom = fh_low
    for j, row in after_fh.iterrows():
        elapsed_hours = (j + 1) * bar_hours

        bar_low = float(row["Low"])
        bar_high = float(row["High"])
        bar_range = bar_high - bar_low

        made_new_low = bar_low < abs_bottom
        if made_new_low:
            abs_bottom = bar_low

        recovery = bar_high - abs_bottom
        recovery_pct = (recovery / fh_range_down) * 100.0

        if bounce_25_hours is None and recovery_pct >= 25:
            bounce_25_hours = round(elapsed_hours, 4)
            bounce_25_level = round(bar_high, 4)

        if bounce_50_hours is None and recovery_pct >= 50:
            bounce_50_hours = round(elapsed_hours, 4)
            bounce_50_level = round(bar_high, 4)

        if bounce_25_hours is None:
            in_band = bar_high <= (abs_bottom + band_pct * fh_range_down)
            not_huge = bar_range <= (max_range_mult * fh_range_down)
            if (not made_new_low) and in_band and not_huge:
                chop_hours += bar_hours

    return bounce_25_hours, bounce_25_level, bounce_50_hours, bounce_50_level, round(chop_hours, 4)


def scan_buy_pullback(after_fh, fh_range_up, fh_high, bar_hours, band_pct, max_range_mult):
    """
    BUY-side: measure pullback from running top (starting fh_high).
    Return:
      pullback_25_hours, pullback_25_level, pullback_50_hours, pullback_50_level, chop_hours
    Where *level* is bar_low at first threshold hit.
    """
    pullback_25_hours = None
    pullback_50_hours = None
    pullback_25_level = None
    pullback_50_level = None
    chop_hours = 0.0

    if after_fh.empty or fh_range_up <= 0:
        return pullback_25_hours, pullback_25_level, pullback_50_hours, pullback_50_level, chop_hours

    abs_top = fh_high
    for j, row in after_fh.iterrows():
        elapsed_hours = (j + 1) * bar_hours

        bar_low = float(row["Low"])
        bar_high = float(row["High"])
        bar_range = bar_high - bar_low

        made_new_high = bar_high > abs_top
        if made_new_high:
            abs_top = bar_high

        pullback = abs_top - bar_low
        pullback_pct = (pullback / fh_range_up) * 100.0

        if pullback_25_hours is None and pullback_pct >= 25:
            pullback_25_hours = round(elapsed_hours, 4)
            pullback_25_level = round(bar_low, 4)

        if pullback_50_hours is None and pullback_pct >= 50:
            pullback_50_hours = round(elapsed_hours, 4)
            pullback_50_level = round(bar_low, 4)

        if pullback_25_hours is None:
            in_band = bar_low >= (abs_top - band_pct * fh_range_up)
            not_huge = bar_range <= (max_range_mult * fh_range_up)
            if (not made_new_high) and in_band and not_huge:
                chop_hours += bar_hours

    return pullback_25_hours, pullback_25_level, pullback_50_hours, pullback_50_level, round(chop_hours, 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--tf", default="1h", choices=SUPPORTED_TF)
    parser.add_argument("--days", type=int, default=180)

    parser.add_argument("--volume_quantile", type=float, default=0.75)
    parser.add_argument("--move_threshold_pct", type=float, default=0.30)

    parser.add_argument("--chop_band_pct", type=float, default=None)
    parser.add_argument("--chop_max_range_mult", type=float, default=None)

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
    print(f"Using chop params: band_pct={CHOP_BAND_PCT}, max_range_mult={CHOP_MAX_RANGE_MULT}")
    print(f"Bar size: {BAR_HOURS} hours\n")

    check_and_install_dependencies()
    print()

    import yfinance as yf
    import pandas as pd
    from datetime import time
    import json

    print("=" * 80)
    print(f"STEP 1: Fetching {symbol} data from Yahoo Finance")
    print("=" * 80)

    max_days = INTRADAY_MAX_DAYS.get(tf, args.days)
    if args.days > max_days:
        print(f"⚠️ Yahoo limit: {tf} supports ~{max_days} days. Reducing lookback from {args.days} -> {max_days}.")
        args.days = max_days

    def days_to_period(d):
        if tf in ("15m", "30m") and d >= 60:
            return "59d"
        return f"{int(d)}d"

    period = days_to_period(args.days)
    print(f"Downloading {symbol} | interval={tf} | period={period} ...")

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
    if len(df) == 0:
        print("✗ No data received.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("STEP 2: Processing and Analyzing Data")
    print("=" * 80)

    df = df.reset_index()
    df = flatten_columns_if_needed(df)
    df = ensure_datetime_column(df)

    for base in ["Open", "High", "Low", "Close", "Volume"]:
        src = pick_col(df, base)
        if src != base:
            df[base] = df[src]

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = convert_to_ny_time(df)
    df = df.sort_values("Datetime").reset_index(drop=True)

    df["Time"] = df["Datetime"].dt.time
    df["Date"] = df["Datetime"].dt.date

    rth_start = time(9, 30)
    rth_end = time(16, 0)
    df = df[(df["Time"] >= rth_start) & (df["Time"] <= rth_end)].copy()

    fh_end = time(10, 30)
    df["IsFirstHour"] = df["Time"].apply(lambda t: rth_start <= t < fh_end)

    first_hour_volume = df.loc[df["IsFirstHour"]].groupby("Date")["Volume"].sum()
    if len(first_hour_volume) == 0:
        print("✗ No first-hour bars found in 09:30–10:30 window.")
        sys.exit(1)

    big_volume_threshold = float(first_hour_volume.quantile(args.volume_quantile))
    print(f"\n✓ Big volume threshold (q={args.volume_quantile:.2f}): {big_volume_threshold:,.0f}")

    results = {"selloffs": [], "buyups": [], "chop_days": [], "other_days": []}
    charts_by_date = {}

    for date in df["Date"].unique():
        day_data = df[df["Date"] == date].sort_values("Datetime").reset_index(drop=True)
        if len(day_data) < 2:
            continue

        fh = day_data[day_data["IsFirstHour"]].reset_index(drop=True)
        if len(fh) == 0:
            continue

        fh_volume = float(fh["Volume"].sum())
        day_volume = float(day_data["Volume"].sum()) if len(day_data) else 0.0
        fh_vol_vs_day = (fh_volume / day_volume) if day_volume > 0 else None
        is_big_volume = fh_volume >= big_volume_threshold

        lh_metrics = compute_last_hour_metrics(day_data, rth_end_time=rth_end)

        fh_open = float(fh.iloc[0]["Open"])
        fh_high = float(fh["High"].max())
        fh_low = float(fh["Low"].min())
        fh_close = float(fh.iloc[-1]["Close"])
        fh_move_pct = ((fh_close - fh_open) / fh_open) * 100.0

        after_fh = day_data[~day_data["IsFirstHour"]].reset_index(drop=True)

        # post-first-hour final extremes
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

        made_new_high = (final_abs_high is not None) and (final_abs_high > fh_high)
        made_new_low = (final_abs_low is not None) and (final_abs_low < fh_low)

        # chart for all kept days
        chart_b64 = make_candles_base64(day_data, title=f"{symbol} {tf} | {date} (RTH)")
        if chart_b64:
            charts_by_date[str(date)] = chart_b64

        # SELL-OFF bucket (big-volume only)
        if is_big_volume and fh_move_pct <= -move_thr:
            fh_range_down = fh_open - fh_low
            continuation_found = (final_abs_low is not None) and (final_abs_low < fh_low)
            failed_selloff = not continuation_found
            continuation_hours = hours_to_final_abs_low if continuation_found else 0.0

            b25h, b25lvl, b50h, b50lvl, chop_hours = scan_sell_bounce(
                after_fh, fh_range_down, fh_low, BAR_HOURS, CHOP_BAND_PCT, CHOP_MAX_RANGE_MULT
            )

            results["selloffs"].append({
                "date": str(date), "tf": tf,
                "fh_move_pct": round(fh_move_pct, 4),
                "first_hour_range": round(fh_range_down, 4),

                "fh_volume": round(fh_volume, 2),
                "day_volume": round(day_volume, 2),
                "fh_vol_vs_day": None if fh_vol_vs_day is None else round(fh_vol_vs_day, 4),

                "final_abs_low": None if final_abs_low is None else round(final_abs_low, 4),
                "hours_to_final_abs_low": hours_to_final_abs_low,
                "extension_points_below_fh_low": ext_below,

                "continuation_hours": continuation_hours,
                "chop_hours": chop_hours,

                # time AFTER first hour
                "bounce_25_hours_after_fh": b25h,
                "bounce_50_hours_after_fh": b50h,

                # price level where bounce occurred
                "bounce_25_level": b25lvl,
                "bounce_50_level": b50lvl,

                "failed_selloff": bool(failed_selloff),

                **lh_metrics,
            })

        # BUY-UP bucket (big-volume only)
        elif is_big_volume and fh_move_pct >= move_thr:
            fh_range_up = fh_high - fh_open
            continuation_found = (final_abs_high is not None) and (final_abs_high > fh_high)
            failed_buyup = not continuation_found
            continuation_hours = hours_to_final_abs_high if continuation_found else 0.0

            p25h, p25lvl, p50h, p50lvl, chop_hours = scan_buy_pullback(
                after_fh, fh_range_up, fh_high, BAR_HOURS, CHOP_BAND_PCT, CHOP_MAX_RANGE_MULT
            )

            results["buyups"].append({
                "date": str(date), "tf": tf,
                "fh_move_pct": round(fh_move_pct, 4),
                "first_hour_range": round(fh_range_up, 4),

                "fh_volume": round(fh_volume, 2),
                "day_volume": round(day_volume, 2),
                "fh_vol_vs_day": None if fh_vol_vs_day is None else round(fh_vol_vs_day, 4),

                "final_abs_high": None if final_abs_high is None else round(final_abs_high, 4),
                "hours_to_final_abs_high": hours_to_final_abs_high,
                "extension_points_above_fh_high": ext_above,

                "continuation_hours": continuation_hours,
                "chop_hours": chop_hours,

                "pullback_25_hours_after_fh": p25h,
                "pullback_50_hours_after_fh": p50h,

                "pullback_25_level": p25lvl,
                "pullback_50_level": p50lvl,

                "failed_buyup": bool(failed_buyup),

                **lh_metrics,
            })

        # BIG-VOLUME CHOP/NO BREAKOUT
        elif is_big_volume:
            results["chop_days"].append({
                "date": str(date), "tf": tf,
                "fh_move_pct": round(fh_move_pct, 4),

                "fh_volume": round(fh_volume, 2),
                "day_volume": round(day_volume, 2),
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

        # OTHER/NORMAL DAYS (not big-volume)
        else:
            # optional bounce metrics on normal days:
            fh_range_down = fh_open - fh_low
            fh_range_up = fh_high - fh_open

            b25h, b25lvl, b50h, b50lvl, _ = scan_sell_bounce(
                after_fh, fh_range_down, fh_low, BAR_HOURS, CHOP_BAND_PCT, CHOP_MAX_RANGE_MULT
            )
            p25h, p25lvl, p50h, p50lvl, _ = scan_buy_pullback(
                after_fh, fh_range_up, fh_high, BAR_HOURS, CHOP_BAND_PCT, CHOP_MAX_RANGE_MULT
            )

            results["other_days"].append({
                "date": str(date), "tf": tf,
                "fh_move_pct": round(fh_move_pct, 4),

                "fh_volume": round(fh_volume, 2),
                "day_volume": round(day_volume, 2),
                "fh_vol_vs_day": None if fh_vol_vs_day is None else round(fh_vol_vs_day, 4),

                "fh_high": round(fh_high, 4),
                "fh_low": round(fh_low, 4),

                "final_abs_high": None if final_abs_high is None else round(final_abs_high, 4),
                "hours_to_final_abs_high": hours_to_final_abs_high,
                "extension_points_above_fh_high": ext_above,
                "made_new_high_after_fh": bool(made_new_high),
                "failed_new_high_after_fh": bool(not made_new_high),

                "final_abs_low": None if final_abs_low is None else round(final_abs_low, 4),
                "hours_to_final_abs_low": hours_to_final_abs_low,
                "extension_points_below_fh_low": ext_below,
                "made_new_low_after_fh": bool(made_new_low),
                "failed_new_low_after_fh": bool(not made_new_low),

                # "bounce from running bottom" (FH-low anchored)
                "bounce25_from_fh_low_hours_after_fh": b25h,
                "bounce25_from_fh_low_level": b25lvl,
                "bounce50_from_fh_low_hours_after_fh": b50h,
                "bounce50_from_fh_low_level": b50lvl,

                # "pullback from running top" (FH-high anchored)
                "pullback25_from_fh_high_hours_after_fh": p25h,
                "pullback25_from_fh_high_level": p25lvl,
                "pullback50_from_fh_high_hours_after_fh": p50h,
                "pullback50_from_fh_high_level": p50lvl,

                **lh_metrics,
            })

    charts_json = f"{symbol}_{tf}_charts.json"
    with open(charts_json, "w") as f:
        json.dump(charts_by_date, f)
    print(f"\n✓ Saved charts: {charts_json} ({len(charts_by_date)} days)")

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
            "continuation_rate": float((sell_df["continuation_hours"] > 0).mean()),
            "avg_first_hour_range": safe_mean(sell_df["first_hour_range"]),
            "avg_extension_below_fh_low": safe_mean(sell_df["extension_points_below_fh_low"]),
            "avg_hours_to_final_abs_low": safe_mean(sell_df["hours_to_final_abs_low"]),
            "avg_chop_hours": safe_mean(sell_df["chop_hours"]),
            "bounce_25_rate": float(sell_df["bounce_25_hours_after_fh"].notna().mean()),
            "bounce_50_rate": float(sell_df["bounce_50_hours_after_fh"].notna().mean()),
            "avg_lh_move_pct": safe_mean(sell_df["lh_move_pct"]) if "lh_move_pct" in sell_df.columns else 0.0,
            "avg_lh_range": safe_mean(sell_df["lh_range"]) if "lh_range" in sell_df.columns else 0.0,
            "avg_lh_vol_vs_day": safe_mean(sell_df["lh_vol_vs_day"]) if "lh_vol_vs_day" in sell_df.columns else 0.0,
        }

    if len(buy_df):
        summary["buyups"] = {
            "total_count": int(len(buy_df)),
            "continuation_rate": float((buy_df["continuation_hours"] > 0).mean()),
            "avg_first_hour_range": safe_mean(buy_df["first_hour_range"]),
            "avg_extension_above_fh_high": safe_mean(buy_df["extension_points_above_fh_high"]),
            "avg_hours_to_final_abs_high": safe_mean(buy_df["hours_to_final_abs_high"]),
            "avg_chop_hours": safe_mean(buy_df["chop_hours"]),
            "pullback_25_rate": float(buy_df["pullback_25_hours_after_fh"].notna().mean()),
            "pullback_50_rate": float(buy_df["pullback_50_hours_after_fh"].notna().mean()),
            "avg_lh_move_pct": safe_mean(buy_df["lh_move_pct"]) if "lh_move_pct" in buy_df.columns else 0.0,
            "avg_lh_range": safe_mean(buy_df["lh_range"]) if "lh_range" in buy_df.columns else 0.0,
            "avg_lh_vol_vs_day": safe_mean(buy_df["lh_vol_vs_day"]) if "lh_vol_vs_day" in buy_df.columns else 0.0,
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
            "avg_lh_range": safe_mean(chop_df["lh_range"]) if "lh_range" in chop_df.columns else 0.0,
            "avg_lh_vol_vs_day": safe_mean(chop_df["lh_vol_vs_day"]) if "lh_vol_vs_day" in chop_df.columns else 0.0,
        }

    if len(other_df):
        summary["other_days"] = {
            "total_count": int(len(other_df)),
            "new_high_rate": float(other_df["made_new_high_after_fh"].mean()),
            "new_low_rate": float(other_df["made_new_low_after_fh"].mean()),
            "avg_ext_above_fh_high": safe_mean(other_df["extension_points_above_fh_high"]),
            "avg_ext_below_fh_low": safe_mean(other_df["extension_points_below_fh_low"]),
            "avg_hours_to_final_high": safe_mean(other_df["hours_to_final_abs_high"]),
            "avg_hours_to_final_low": safe_mean(other_df["hours_to_final_abs_low"]),
            "avg_lh_move_pct": safe_mean(other_df["lh_move_pct"]) if "lh_move_pct" in other_df.columns else 0.0,
            "avg_lh_range": safe_mean(other_df["lh_range"]) if "lh_range" in other_df.columns else 0.0,
            "avg_lh_vol_vs_day": safe_mean(other_df["lh_vol_vs_day"]) if "lh_vol_vs_day" in other_df.columns else 0.0,
        }

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
    print(f"✓ {sell_csv}")
    print(f"✓ {buy_csv}")
    print(f"✓ {chop_csv}")
    print(f"✓ {other_csv}")
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
