#!/usr/bin/env python3
"""
SPY First Hour Volume Analysis - Complete Package
Downloads real data from Yahoo Finance and generates interactive dashboard

- Uses NY RTH time (America/New_York)
- Filters to 9:30–16:00 ET
- Defines first hour as 9:30–10:30 ET (time-window based)
- Adds:
    * final_abs_low / high AFTER the first hour
    * hours_to_final_abs_low / high
    * extension points beyond first-hour low/high

Usage:
    python run_complete_analysis.py

Requirements:
    yfinance pandas numpy
"""

import subprocess
import sys
import os


def check_and_install_dependencies():
    """Check and install required packages"""
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


def flatten_columns_if_needed(df):
    """Flatten yfinance MultiIndex columns to single strings."""
    import pandas as pd

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if x not in (None, "")]).strip("_")
            for col in df.columns
        ]
    return df


def pick_col(df, base):
    """Pick a column that matches base name (Open/High/Low/Close/Volume) even if suffixed (e.g., Open_SPY)."""
    base_lower = base.lower()
    if base in df.columns:
        return base

    # Prefer exact prefix match before underscore
    cands = [c for c in df.columns if c.split("_")[0].lower() == base_lower]
    if not cands:
        # fallback: startswith
        cands = [c for c in df.columns if c.lower().startswith(base_lower)]
    if not cands:
        raise KeyError(f"Missing '{base}' column. Available columns: {list(df.columns)}")
    return cands[0]

def ensure_datetime_column(spy):
    """Standardize the datetime column name to 'Datetime' after reset_index()."""
    cols = list(spy.columns)

    # If MultiIndex tuples are present, convert to strings for searching
    def col_to_str(c):
        if isinstance(c, tuple):
            return "_".join([str(x) for x in c if x not in (None, "")])
        return str(c)

    col_strs = [col_to_str(c) for c in cols]

    # Direct matches
    if "Datetime" in col_strs:
        # rename the underlying actual column to Datetime if needed
        idx = col_strs.index("Datetime")
        if cols[idx] != "Datetime":
            spy = spy.rename(columns={cols[idx]: "Datetime"})
        return spy

    if "Date" in col_strs:
        idx = col_strs.index("Date")
        spy = spy.rename(columns={cols[idx]: "Datetime"})
        return spy

    if "index" in col_strs:
        idx = col_strs.index("index")
        spy = spy.rename(columns={cols[idx]: "Datetime"})
        return spy

    # Fuzzy search
    for c_real, c_str in zip(cols, col_strs):
        s = c_str.lower()
        if "datetime" in s or (("date" in s) and ("adj" not in s) and ("close" not in s)):
            spy = spy.rename(columns={c_real: "Datetime"})
            return spy

    raise KeyError(f"Couldn't find a datetime column after reset_index(). Columns: {col_strs}")
    
# def ensure_datetime_column(spy):
    # """Standardize the datetime column name to 'Datetime' after reset_index."""
    # cols = list(spy.columns)
    # if "Datetime" in cols:
    #     return spy
    # if "Date" in cols and "Datetime" not in cols:
    #     # Some dataframes may use Date
    #     spy = spy.rename(columns={"Date": "Datetime"})
    #     return spy
    # if "index" in cols:
    #     spy = spy.rename(columns={"index": "Datetime"})
    #     return spy

    # # As a last resort, look for the first datetime-like column
    # for c in cols:
    #     if "time" in c.lower() or "date" in c.lower():
    #         spy = spy.rename(columns={c: "Datetime"})
    #         return spy

    # raise KeyError(f"Couldn't find a datetime column after reset_index(). Columns: {cols}")


def convert_to_ny_time(spy):
    """Convert Datetime to America/New_York, handling tz-naive and tz-aware."""
    import pandas as pd

    dt = pd.to_datetime(spy["Datetime"], errors="coerce")

    # If tz-naive, assume it's UTC (common with yfinance intraday), then convert.
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    else:
        dt = dt.dt.tz_convert("America/New_York")

    spy["Datetime"] = dt
    return spy


def main():
    print("=" * 80)
    print("SPY FIRST HOUR VOLUME ANALYSIS - COMPLETE PACKAGE")
    print("=" * 80)
    print()

    # Check dependencies
    check_and_install_dependencies()
    print()

    # Imports after dependencies
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta, time
    import json

    print("=" * 80)
    print("STEP 1: Fetching SPY Data from Yahoo Finance")
    print("=" * 80)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)

    print(f"Downloading SPY hourly data from {start_date.date()} to {end_date.date()}...")

    try:
        spy = yf.download("SPY", start=start_date, end=end_date, interval="1h", progress=True)
        print(f"\n✓ Successfully downloaded {len(spy)} hourly bars")
    except Exception as e:
        print(f"\n✗ Error downloading data: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Yahoo Finance API may be temporarily down")
        print("3. Try running again in a few minutes")
        sys.exit(1)

    if len(spy) == 0:
        print("✗ No data received. Please check your internet connection and try again.")
        sys.exit(1)

    print("\n" + "=" * 80)
    print("STEP 2: Processing and Analyzing Data")
    print("=" * 80)

    # # Reset index so datetime is a column
    # spy = spy.reset_index()
    # spy = ensure_datetime_column(spy)

    # # Flatten MultiIndex columns (yfinance often returns these)
    # spy = flatten_columns_if_needed(spy)

    spy = spy.reset_index()
    spy = flatten_columns_if_needed(spy)   # <-- move this up
    spy = ensure_datetime_column(spy)

    # Normalize OHLCV columns to canonical names
    for base in ["Open", "High", "Low", "Close", "Volume"]:
        src = pick_col(spy, base)
        if src != base:
            spy[base] = spy[src]

    # Ensure numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        spy[c] = pd.to_numeric(spy[c], errors="coerce")

    # Convert to NY timezone
    spy = convert_to_ny_time(spy)

    # Build time helpers
    spy = spy.sort_values("Datetime").reset_index(drop=True)
    spy["Time"] = spy["Datetime"].dt.time
    spy["Date"] = spy["Datetime"].dt.date

    # Filter to RTH: 9:30–16:00 ET
    rth_start = time(9, 30)
    rth_end = time(16, 0)
    spy = spy[(spy["Time"] >= rth_start) & (spy["Time"] <= rth_end)].copy()

    # Define first hour: 9:30–10:30 ET (time window)
    fh_end = time(10, 30)
    spy["IsFirstHour"] = spy["Time"].apply(lambda t: rth_start <= t < fh_end)

    print("\nIdentifying first hour of trading for each day... (RTH 09:30–10:30 ET)")
    # If a day has no bars in the first-hour window (can happen with 1h bars),
    # we will skip that day (you can switch to 30m interval for better precision).

    # Volume threshold (75th percentile of first-hour volume)
    print("Calculating volume thresholds...")
    first_hour_volume = spy.loc[spy["IsFirstHour"]].groupby("Date")["Volume"].sum()
    if len(first_hour_volume) == 0:
        print("✗ No first-hour bars found in 09:30–10:30 window. Switch to interval='30m' for accuracy.")
        sys.exit(1)

    big_volume_threshold = float(first_hour_volume.quantile(0.75))
    print(f"✓ Big volume threshold (75th percentile): {big_volume_threshold:,.0f}")

    # Analyze moves
    print("\nAnalyzing first hour moves...")

    results = {"selloffs": [], "buyups": []}

    for date in spy["Date"].unique():
        day_data = spy[spy["Date"] == date].sort_values("Datetime").reset_index(drop=True)
        if len(day_data) < 2:
            continue

        first_hour = day_data[day_data["IsFirstHour"]].reset_index(drop=True)
        if len(first_hour) == 0:
            # No clean 09:30–10:30 bars for this date (common with 1h interval). Skip.
            continue

        # First-hour metrics
        fh_volume = float(first_hour["Volume"].sum())
        fh_open = float(first_hour.iloc[0]["Open"])
        fh_high = float(first_hour["High"].max())
        fh_low = float(first_hour["Low"].min())
        fh_close = float(first_hour.iloc[-1]["Close"])

        price_change_pct = ((fh_close - fh_open) / fh_open) * 100.0

        if fh_volume < big_volume_threshold:
            continue

        after_fh = day_data[~day_data["IsFirstHour"]].reset_index(drop=True)

        # -------------------------
        # SELL-OFF (big down first hour)
        # -------------------------
        if price_change_pct < -0.3:
            fh_range = fh_open - fh_low  # points down in first hour

            continuation_hours = 0
            chop_hours = 0
            bounce_25_hours = None
            bounce_50_hours = None
            failed = False

            final_abs_low = None
            hours_to_final_abs_low = None
            extension_points = None  # how many points below fh_low

            if len(after_fh) > 0 and fh_range > 0:
                # Final abs low after first hour
                final_abs_low = float(after_fh["Low"].min())
                idx_low = int(after_fh["Low"].idxmin())
                hours_to_final_abs_low = idx_low + 1

                # Extension beyond first-hour low
                extension_points = float(fh_low - final_abs_low)  # positive means extended lower

                continuation_found = final_abs_low < fh_low
                continuation_hours = hours_to_final_abs_low if continuation_found else 0
                failed = not continuation_found

                # Bounce/chop measured from running abs_bottom
                abs_bottom = fh_low
                for j, row in after_fh.iterrows():
                    hours = j + 1

                    if row["Low"] < abs_bottom:
                        abs_bottom = row["Low"]

                    recovery = row["High"] - abs_bottom
                    recovery_pct = (recovery / fh_range) * 100.0

                    if bounce_25_hours is None and recovery_pct >= 25:
                        bounce_25_hours = hours
                    if bounce_50_hours is None and recovery_pct >= 50:
                        bounce_50_hours = hours

                    # Chop until bounce_25 occurs
                    if bounce_25_hours is None:
                        if (row["High"] - row["Low"]) < fh_range * 0.3:
                            chop_hours += 1
            else:
                failed = True

            results["selloffs"].append(
                {
                    "date": str(date),
                    "first_hour_range": round(fh_range, 2),
                    "final_abs_low": None if final_abs_low is None else round(final_abs_low, 2),
                    "hours_to_final_abs_low": hours_to_final_abs_low,
                    "extension_points_below_fh_low": None if extension_points is None else round(extension_points, 2),
                    "continuation_hours": continuation_hours,
                    "chop_hours": chop_hours,
                    "bounce_25_hours": bounce_25_hours,
                    "bounce_50_hours": bounce_50_hours,
                    "failed_selloff": failed,
                }
            )

        # -------------------------
        # BUY-UP (big up first hour)
        # -------------------------
        elif price_change_pct > 0.3:
            fh_range = fh_high - fh_open  # points up in first hour

            continuation_hours = 0
            chop_hours = 0
            pullback_25_hours = None
            pullback_50_hours = None
            failed = False

            final_abs_high = None
            hours_to_final_abs_high = None
            extension_points = None  # how many points above fh_high

            if len(after_fh) > 0 and fh_range > 0:
                # Final abs high after first hour
                final_abs_high = float(after_fh["High"].max())
                idx_high = int(after_fh["High"].idxmax())
                hours_to_final_abs_high = idx_high + 1

                # Extension beyond first-hour high
                extension_points = float(final_abs_high - fh_high)  # positive means extended higher

                continuation_found = final_abs_high > fh_high
                continuation_hours = hours_to_final_abs_high if continuation_found else 0
                failed = not continuation_found

                # Pullback/chop measured from running abs_top
                abs_top = fh_high
                for j, row in after_fh.iterrows():
                    hours = j + 1

                    if row["High"] > abs_top:
                        abs_top = row["High"]

                    pullback = abs_top - row["Low"]
                    pullback_pct = (pullback / fh_range) * 100.0

                    if pullback_25_hours is None and pullback_pct >= 25:
                        pullback_25_hours = hours
                    if pullback_50_hours is None and pullback_pct >= 50:
                        pullback_50_hours = hours

                    # Chop until pullback_25 occurs
                    if pullback_25_hours is None:
                        if (row["High"] - row["Low"]) < fh_range * 0.3:
                            chop_hours += 1
            else:
                failed = True

            results["buyups"].append(
                {
                    "date": str(date),
                    "first_hour_range": round(fh_range, 2),
                    "final_abs_high": None if final_abs_high is None else round(final_abs_high, 2),
                    "hours_to_final_abs_high": hours_to_final_abs_high,
                    "extension_points_above_fh_high": None if extension_points is None else round(extension_points, 2),
                    "continuation_hours": continuation_hours,
                    "chop_hours": chop_hours,
                    "pullback_25_hours": pullback_25_hours,
                    "pullback_50_hours": pullback_50_hours,
                    "failed_buyup": failed,
                }
            )

    # Summary stats
    print("\n" + "=" * 80)
    print("STEP 3: Calculating Statistics")
    print("=" * 80)

    import pandas as pd

    selloffs_df = pd.DataFrame(results["selloffs"])
    buyups_df = pd.DataFrame(results["buyups"])

    summary = {"selloffs": {}, "buyups": {}}

    if len(selloffs_df) > 0:
        ext = selloffs_df["extension_points_below_fh_low"].dropna()
        t_ext = selloffs_df["hours_to_final_abs_low"].dropna()

        summary["selloffs"] = {
            "total_count": int(len(selloffs_df)),
            "avg_selloff_points": float(selloffs_df["first_hour_range"].mean()),
            "continuation_count": int((selloffs_df["continuation_hours"] > 0).sum()),
            "avg_continuation_hours": float(
                selloffs_df.loc[selloffs_df["continuation_hours"] > 0, "continuation_hours"].mean()
            )
            if int((selloffs_df["continuation_hours"] > 0).sum()) > 0
            else 0.0,
            "chop_count": int((selloffs_df["chop_hours"] > 0).sum()),
            "avg_chop_hours": float(selloffs_df.loc[selloffs_df["chop_hours"] > 0, "chop_hours"].mean())
            if int((selloffs_df["chop_hours"] > 0).sum()) > 0
            else 0.0,
            "bounce_50_count": int(selloffs_df["bounce_50_hours"].notna().sum()),
            "avg_bounce_50_hours": float(selloffs_df.loc[selloffs_df["bounce_50_hours"].notna(), "bounce_50_hours"].mean())
            if int(selloffs_df["bounce_50_hours"].notna().sum()) > 0
            else 0.0,
            "bounce_25_count": int(selloffs_df["bounce_25_hours"].notna().sum()),
            "avg_bounce_25_hours": float(selloffs_df.loc[selloffs_df["bounce_25_hours"].notna(), "bounce_25_hours"].mean())
            if int(selloffs_df["bounce_25_hours"].notna().sum()) > 0
            else 0.0,
            "failed_selloff_count": int(selloffs_df["failed_selloff"].sum()),
            "avg_extension_points_below_fh_low": float(ext.mean()) if len(ext) else 0.0,
            "avg_hours_to_final_abs_low": float(t_ext.mean()) if len(t_ext) else 0.0,
        }

    if len(buyups_df) > 0:
        ext = buyups_df["extension_points_above_fh_high"].dropna()
        t_ext = buyups_df["hours_to_final_abs_high"].dropna()

        summary["buyups"] = {
            "total_count": int(len(buyups_df)),
            "avg_buyup_points": float(buyups_df["first_hour_range"].mean()),
            "continuation_count": int((buyups_df["continuation_hours"] > 0).sum()),
            "avg_continuation_hours": float(
                buyups_df.loc[buyups_df["continuation_hours"] > 0, "continuation_hours"].mean()
            )
            if int((buyups_df["continuation_hours"] > 0).sum()) > 0
            else 0.0,
            "chop_count": int((buyups_df["chop_hours"] > 0).sum()),
            "avg_chop_hours": float(buyups_df.loc[buyups_df["chop_hours"] > 0, "chop_hours"].mean())
            if int((buyups_df["chop_hours"] > 0).sum()) > 0
            else 0.0,
            "pullback_50_count": int(buyups_df["pullback_50_hours"].notna().sum()),
            "avg_pullback_50_hours": float(
                buyups_df.loc[buyups_df["pullback_50_hours"].notna(), "pullback_50_hours"].mean()
            )
            if int(buyups_df["pullback_50_hours"].notna().sum()) > 0
            else 0.0,
            "pullback_25_count": int(buyups_df["pullback_25_hours"].notna().sum()),
            "avg_pullback_25_hours": float(
                buyups_df.loc[buyups_df["pullback_25_hours"].notna(), "pullback_25_hours"].mean()
            )
            if int(buyups_df["pullback_25_hours"].notna().sum()) > 0
            else 0.0,
            "failed_buyup_count": int(buyups_df["failed_buyup"].sum()),
            "avg_extension_points_above_fh_high": float(ext.mean()) if len(ext) else 0.0,
            "avg_hours_to_final_abs_high": float(t_ext.mean()) if len(t_ext) else 0.0,
        }

    # Save results
    selloffs_df.to_csv("selloffs_detail.csv", index=False)
    buyups_df.to_csv("buyups_detail.csv", index=False)

    with open("summary_stats.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print results
    print("\n📉 SELL-OFF ANALYSIS:")
    if summary.get("selloffs"):
        s = summary["selloffs"]
        print(f"   Total sell-offs: {s['total_count']}")
        print(f"   Avg size (FH drop): {s['avg_selloff_points']:.2f} points")
        print(
            f"   Continuation rate: {s['continuation_count']}/{s['total_count']} "
            f"({(s['continuation_count']/s['total_count']*100):.1f}%)"
        )
        print(f"   Avg extension below FH low: {s['avg_extension_points_below_fh_low']:.2f} points")
        print(f"   Avg hours to final abs low: {s['avg_hours_to_final_abs_low']:.2f}")
        print(
            f"   50% bounce rate: {s['bounce_50_count']}/{s['total_count']} "
            f"({(s['bounce_50_count']/s['total_count']*100):.1f}%)"
        )
    else:
        print("   No sell-offs detected")

    print("\n📈 BUY-UP ANALYSIS:")
    if summary.get("buyups"):
        b = summary["buyups"]
        print(f"   Total buy-ups: {b['total_count']}")
        print(f"   Avg size (FH rally): {b['avg_buyup_points']:.2f} points")
        print(
            f"   Continuation rate: {b['continuation_count']}/{b['total_count']} "
            f"({(b['continuation_count']/b['total_count']*100):.1f}%)"
        )
        print(f"   Avg extension above FH high: {b['avg_extension_points_above_fh_high']:.2f} points")
        print(f"   Avg hours to final abs high: {b['avg_hours_to_final_abs_high']:.2f}")
        print(
            f"   50% pullback rate: {b['pullback_50_count']}/{b['total_count']} "
            f"({(b['pullback_50_count']/b['total_count']*100):.1f}%)"
        )
    else:
        print("   No buy-ups detected")

    # Generate dashboard
    print("\n" + "=" * 80)
    print("STEP 4: Generating Interactive Dashboard")
    print("=" * 80)

    try:
        subprocess.run([sys.executable, "create_dashboard.py"], check=True)
        print("\n✅ COMPLETE! Files generated:")
        print("   1. selloffs_detail.csv - Detailed sell-off data")
        print("   2. buyups_detail.csv - Detailed buy-up data")
        print("   3. summary_stats.json - Summary statistics")
        print("   4. spy_analysis_dashboard.html - Interactive dashboard")
        print("\n🌐 Open spy_analysis_dashboard.html in your browser to view results!")
    except Exception as e:
        print(f"\nNote: Could not auto-generate dashboard: {e}")
        print("Run 'python create_dashboard.py' manually to generate the HTML dashboard")


if __name__ == "__main__":
    main()

