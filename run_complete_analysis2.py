#!/usr/bin/env python3
"""
SPY First Hour Volume Analysis - Complete Package
Downloads real data from Yahoo Finance and generates interactive dashboard

Usage:
    python run_complete_analysis.py

Requirements:
    pip install yfinance pandas numpy
"""

import subprocess
import sys
import os

def check_and_install_dependencies():
    """Check and install required packages"""
    required = ['yfinance', 'pandas', 'numpy']
    
    print("Checking dependencies...")
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} not found. Installing...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")

def main():
    """Main execution function"""
    
    print("=" * 80)
    print("SPY FIRST HOUR VOLUME ANALYSIS - COMPLETE PACKAGE")
    print("=" * 80)
    print()
    
    # Check dependencies
    check_and_install_dependencies()
    print()
    
    # Now import after ensuring dependencies are installed
    import yfinance as yf
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    import json
    
    print("=" * 80)
    print("STEP 1: Fetching SPY Data from Yahoo Finance")
    print("=" * 80)
    
    # Fetch data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    
    print(f"Downloading SPY hourly data from {start_date.date()} to {end_date.date()}...")
    
    try:
        spy = yf.download('SPY', start=start_date, end=end_date, interval='1h', progress=True)
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
    
    # Process data
    print("\n" + "=" * 80)
    print("STEP 2: Processing and Analyzing Data")
    print("=" * 80)
    
    # Reset index to make Datetime a column
    spy = spy.reset_index()

    import pandas as pd

    # Make sure Datetime is a proper datetime
    spy["Datetime"] = pd.to_datetime(spy["Datetime"], utc=True, errors="coerce")

    # Convert to NY time
    spy["Datetime"] = spy["Datetime"].dt.tz_convert("America/New_York")

    # --- Ensure Volume is a 1-D numeric Series (yfinance sometimes returns MultiIndex cols) ---
    if isinstance(spy.columns, pd.MultiIndex):
        # Flatten MultiIndex columns like ('Volume','SPY') -> 'Volume_SPY'
        spy.columns = ['_'.join([str(x) for x in col if x not in (None, '')]).strip('_') for col in spy.columns]

    # If we have Volume_SPY (or similar), map it back to Volume for the rest of the script
    if 'Volume' not in spy.columns:
        vol_candidates = [c for c in spy.columns if c.lower().startswith('volume')]
        if not vol_candidates:
            raise ValueError(f"Couldn't find a Volume column. Columns: {list(spy.columns)}")
        spy['Volume'] = spy[vol_candidates[0]]

    spy['Volume'] = pd.to_numeric(spy['Volume'], errors='coerce')

    spy['Date'] = spy['Datetime'].dt.date
    spy['Hour'] = spy['Datetime'].dt.hour
    
    # # Identify first hour of each trading day
    # print("\nIdentifying first hour of trading for each day...")
    # spy['IsFirstHour'] = False
    
    # for date in spy['Date'].unique():
    #     day_data = spy[spy['Date'] == date]
    #     if len(day_data) > 0:
    #         first_hour_end = day_data.iloc[0]['Datetime'] + pd.Timedelta(hours=1)
    #         spy.loc[spy['Date'] == date, 'IsFirstHour'] = spy.loc[spy['Date'] == date, 'Datetime'] < first_hour_end

    # Identify first hour of each trading day (vectorized, robust)
    print("\nIdentifying first hour of trading for each day...")
    spy = spy.sort_values("Datetime").reset_index(drop=True)

    # Ensure Datetime is datetime64[ns]
    spy["Datetime"] = pd.to_datetime(spy["Datetime"])

    # First timestamp per Date
    first_dt = spy.groupby("Date")["Datetime"].transform("min")
    spy["IsFirstHour"] = spy["Datetime"] < (first_dt + pd.Timedelta(hours=1))

    # Calculate volume threshold
    print("Calculating volume thresholds...")
    # first_hour_volume = spy[spy['IsFirstHour']].groupby('Date')['Volume'].sum()
    # big_volume_threshold = first_hour_volume.quantile(0.75)
    
    # print(f"✓ Big volume threshold (75th percentile): {big_volume_threshold:,.0f}")
    first_hour_volume = spy.loc[spy['IsFirstHour']].groupby('Date')['Volume'].sum()
    big_volume_threshold = float(first_hour_volume.quantile(0.75))
    print(f"✓ Big volume threshold (75th percentile): {big_volume_threshold:,.0f}")

    # Analyze moves
    print("\nAnalyzing first hour moves...")
    
    results = {'selloffs': [], 'buyups': []}

    # --- Normalize yfinance columns to standard OHLCV names ---
    # 1) Flatten MultiIndex columns if present
    if isinstance(spy.columns, pd.MultiIndex):
        spy.columns = ['_'.join([str(x) for x in col if x not in (None, '')]).strip('_') for col in spy.columns]

    # 2) Helper to pick the correct column (handles Open, Open_SPY, etc.)
    def pick_col(df, base):
        if base in df.columns:
            return base
        cands = [c for c in df.columns if c.split('_')[0].lower() == base.lower()]
        if not cands:
            cands = [c for c in df.columns if c.lower().startswith(base.lower())]
        if not cands:
            raise KeyError(f"Missing '{base}' column. Available columns: {list(df.columns)}")
        return cands[0]

    # 3) Create canonical columns Open/High/Low/Close/Volume used throughout your script
    for base in ["Open", "High", "Low", "Close", "Volume"]:
        src = pick_col(spy, base)
        if src != base:
            spy[base] = spy[src]

    # Ensure numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        spy[c] = pd.to_numeric(spy[c], errors="coerce")

    for date in spy['Date'].unique():
        day_data = spy[spy['Date'] == date].sort_values('Datetime').reset_index(drop=True)
        
        if len(day_data) < 2:
            continue
        
        first_hour = day_data[day_data['IsFirstHour']]
        
        if len(first_hour) == 0:
            continue
        
        # Calculate metrics
        fh_volume = first_hour['Volume'].sum()
        fh_open = first_hour.iloc[0]['Open']
        fh_high = first_hour['High'].max()
        fh_low = first_hour['Low'].min()
        fh_close = first_hour.iloc[-1]['Close']
        
        price_change_pct = ((fh_close - fh_open) / fh_open) * 100
        
        if fh_volume < big_volume_threshold:
            continue
        
        # Analyze sell-offs
        if price_change_pct < -0.3:
            fh_range = fh_open - fh_low
            after_fh = day_data[~day_data['IsFirstHour']]
            
            continuation_hours = 0
            chop_hours = 0
            bounce_25_hours = None
            bounce_50_hours = None
            failed = False
            
            if len(after_fh) > 0:
                abs_bottom = fh_low
                continuation_found = False
                
                # for i, row in after_fh.iterrows():
                #     hours = i + 1
                after_fh = after_fh.reset_index(drop=True)
                for j, row in after_fh.iterrows():
                    hours = j + 1           

                    if row['Low'] < abs_bottom:
                        abs_bottom = row['Low']
                        continuation_hours = hours
                        continuation_found = True
                    
                    recovery = row['High'] - abs_bottom
                    recovery_pct = (recovery / fh_range * 100) if fh_range > 0 else 0
                    
                    if bounce_50_hours is None and recovery_pct >= 50:
                        bounce_50_hours = hours
                    if bounce_25_hours is None and recovery_pct >= 25:
                        bounce_25_hours = hours
                    
                    # if continuation_found and bounce_25_hours is None:
                    #     if row['High'] - row['Low'] < fh_range * 0.3:
                    #         chop_hours += 1
                    after_fh = after_fh.reset_index(drop=True)

                    for j, row in after_fh.iterrows():
                        hours = j + 1

                        # update bottom / continuation
                        if row['Low'] < abs_bottom:
                            abs_bottom = row['Low']
                            continuation_hours = hours
                            continuation_found = True

                        # bounce tests
                        recovery = row['High'] - abs_bottom
                        recovery_pct = (recovery / fh_range * 100) if fh_range > 0 else 0

                        if bounce_50_hours is None and recovery_pct >= 50:
                            bounce_50_hours = hours
                        if bounce_25_hours is None and recovery_pct >= 25:
                            bounce_25_hours = hours

                        # chop: count small-range bars UNTIL bounce_25 happens
                        if bounce_25_hours is None:
                            if (row['High'] - row['Low']) < fh_range * 0.3:
                                chop_hours += 1   

                if not continuation_found:
                    failed = True
            
            results['selloffs'].append({
                'date': str(date),
                'first_hour_range': round(fh_range, 2),
                'continuation_hours': continuation_hours,
                'chop_hours': chop_hours,
                'bounce_25_hours': bounce_25_hours,
                'bounce_50_hours': bounce_50_hours,
                'failed_selloff': failed
            })
        
        # Analyze buy-ups
        elif price_change_pct > 0.3:
            fh_range = fh_high - fh_open
            after_fh = day_data[~day_data['IsFirstHour']]
            
            continuation_hours = 0
            chop_hours = 0
            pullback_25_hours = None
            pullback_50_hours = None
            failed = False
            
            if len(after_fh) > 0:
                abs_top = fh_high
                continuation_found = False
                
                # for i, row in after_fh.iterrows():
                #     hours = i + 1
                after_fh = after_fh.reset_index(drop=True)
                for j, row in after_fh.iterrows():
                    hours = j + 1

                    if row['High'] > abs_top:
                        abs_top = row['High']
                        continuation_hours = hours
                        continuation_found = True
                    
                    pullback = abs_top - row['Low']
                    pullback_pct = (pullback / fh_range * 100) if fh_range > 0 else 0
                    
                    if pullback_50_hours is None and pullback_pct >= 50:
                        pullback_50_hours = hours
                    if pullback_25_hours is None and pullback_pct >= 25:
                        pullback_25_hours = hours
                    
                    # if continuation_found and pullback_25_hours is None:
                    #     if row['High'] - row['Low'] < fh_range * 0.3:
                    #         chop_hours += 1
                    after_fh = after_fh.reset_index(drop=True)

                    for j, row in after_fh.iterrows():
                        hours = j + 1

                        if row['High'] > abs_top:
                            abs_top = row['High']
                            continuation_hours = hours
                            continuation_found = True

                        pullback = abs_top - row['Low']
                        pullback_pct = (pullback / fh_range * 100) if fh_range > 0 else 0

                        if pullback_50_hours is None and pullback_pct >= 50:
                            pullback_50_hours = hours
                        if pullback_25_hours is None and pullback_pct >= 25:
                            pullback_25_hours = hours

                        if pullback_25_hours is None:
                            if (row['High'] - row['Low']) < fh_range * 0.3:
                                chop_hours += 1

                if not continuation_found:
                    failed = True
            
            results['buyups'].append({
                'date': str(date),
                'first_hour_range': round(fh_range, 2),
                'continuation_hours': continuation_hours,
                'chop_hours': chop_hours,
                'pullback_25_hours': pullback_25_hours,
                'pullback_50_hours': pullback_50_hours,
                'failed_buyup': failed
            })
    
    # Calculate summary statistics
    print("\n" + "=" * 80)
    print("STEP 3: Calculating Statistics")
    print("=" * 80)
    
    selloffs_df = pd.DataFrame(results['selloffs'])
    buyups_df = pd.DataFrame(results['buyups'])
    
    summary = {'selloffs': {}, 'buyups': {}}
    
    if len(selloffs_df) > 0:
        summary['selloffs'] = {
            'total_count': len(selloffs_df),
            'avg_selloff_points': float(selloffs_df['first_hour_range'].mean()),
            'continuation_count': int((selloffs_df['continuation_hours'] > 0).sum()),
            'avg_continuation_hours': float(selloffs_df[selloffs_df['continuation_hours'] > 0]['continuation_hours'].mean()) if (selloffs_df['continuation_hours'] > 0).sum() > 0 else 0,
            'chop_count': int((selloffs_df['chop_hours'] > 0).sum()),
            'avg_chop_hours': float(selloffs_df[selloffs_df['chop_hours'] > 0]['chop_hours'].mean()) if (selloffs_df['chop_hours'] > 0).sum() > 0 else 0,
            'bounce_50_count': int(selloffs_df['bounce_50_hours'].notna().sum()),
            'avg_bounce_50_hours': float(selloffs_df[selloffs_df['bounce_50_hours'].notna()]['bounce_50_hours'].mean()) if selloffs_df['bounce_50_hours'].notna().sum() > 0 else 0,
            'bounce_25_count': int(selloffs_df['bounce_25_hours'].notna().sum()),
            'avg_bounce_25_hours': float(selloffs_df[selloffs_df['bounce_25_hours'].notna()]['bounce_25_hours'].mean()) if selloffs_df['bounce_25_hours'].notna().sum() > 0 else 0,
            'failed_selloff_count': int(selloffs_df['failed_selloff'].sum())
        }
    
    if len(buyups_df) > 0:
        summary['buyups'] = {
            'total_count': len(buyups_df),
            'avg_buyup_points': float(buyups_df['first_hour_range'].mean()),
            'continuation_count': int((buyups_df['continuation_hours'] > 0).sum()),
            'avg_continuation_hours': float(buyups_df[buyups_df['continuation_hours'] > 0]['continuation_hours'].mean()) if (buyups_df['continuation_hours'] > 0).sum() > 0 else 0,
            'chop_count': int((buyups_df['chop_hours'] > 0).sum()),
            'avg_chop_hours': float(buyups_df[buyups_df['chop_hours'] > 0]['chop_hours'].mean()) if (buyups_df['chop_hours'] > 0).sum() > 0 else 0,
            'pullback_50_count': int(buyups_df['pullback_50_hours'].notna().sum()),
            'avg_pullback_50_hours': float(buyups_df[buyups_df['pullback_50_hours'].notna()]['pullback_50_hours'].mean()) if buyups_df['pullback_50_hours'].notna().sum() > 0 else 0,
            'pullback_25_count': int(buyups_df['pullback_25_hours'].notna().sum()),
            'avg_pullback_25_hours': float(buyups_df[buyups_df['pullback_25_hours'].notna()]['pullback_25_hours'].mean()) if buyups_df['pullback_25_hours'].notna().sum() > 0 else 0,
            'failed_buyup_count': int(buyups_df['failed_buyup'].sum())
        }
    
    # Save results
    selloffs_df.to_csv('selloffs_detail.csv', index=False)
    buyups_df.to_csv('buyups_detail.csv', index=False)
    
    with open('summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print results
    print("\n📉 SELL-OFF ANALYSIS:")
    if summary.get('selloffs'):
        s = summary['selloffs']
        print(f"   Total sell-offs: {s['total_count']}")
        print(f"   Avg size: {s['avg_selloff_points']:.2f} points")
        print(f"   Continuation rate: {s['continuation_count']}/{s['total_count']} ({s['continuation_count']/s['total_count']*100:.1f}%)")
        print(f"   50% bounce rate: {s['bounce_50_count']}/{s['total_count']} ({s['bounce_50_count']/s['total_count']*100:.1f}%)")
    else:
        print("   No sell-offs detected")
    
    print("\n📈 BUY-UP ANALYSIS:")
    if summary.get('buyups'):
        b = summary['buyups']
        print(f"   Total buy-ups: {b['total_count']}")
        print(f"   Avg size: {b['avg_buyup_points']:.2f} points")
        print(f"   Continuation rate: {b['continuation_count']}/{b['total_count']} ({b['continuation_count']/b['total_count']*100:.1f}%)")
        print(f"   50% pullback rate: {b['pullback_50_count']}/{b['total_count']} ({b['pullback_50_count']/b['total_count']*100:.1f}%)")
    else:
        print("   No buy-ups detected")
    
    # Generate dashboard
    print("\n" + "=" * 80)
    print("STEP 4: Generating Interactive Dashboard")
    print("=" * 80)
    
    # Import and run dashboard generator
    try:
        subprocess.run([sys.executable, 'create_dashboard.py'], check=True)
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
