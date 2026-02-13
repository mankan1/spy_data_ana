# SPY First Hour Volume Analysis Dashboard

## 📊 Overview

This comprehensive analysis tool examines SPY (S&P 500 ETF) trading patterns, specifically focusing on high-volume moves during the first hour of trading and tracking subsequent price action throughout the day.

## 🎯 What This Tool Analyzes

### For SELL-OFFS (Big Volume Drops in First Hour):
1. **Detection**: Identifies days with high-volume sell-offs in the first trading hour (9:30-10:30 AM ET)
2. **Continuation**: Tracks how many hours the sell-off continued (new lows)
3. **Chop Periods**: Measures low-volatility consolidation periods after sell-offs
4. **Bounce Analysis**:
   - 50% recoveries from the bottom (frequency and timing)
   - 25% recoveries from the bottom (frequency and timing)
5. **Failed Sell-offs**: Counts instances where buyers stepped in immediately (no new lows)
6. **Point Measurements**: Calculates exact point movements for each event

### For BUY-UPS (Big Volume Rallies in First Hour):
1. **Detection**: Identifies days with high-volume buy-ups in the first trading hour
2. **Continuation**: Tracks how many hours the rally continued (new highs)
3. **Chop Periods**: Measures consolidation after rallies
4. **Pullback Analysis**:
   - 50% pullbacks from the top (frequency and timing)
   - 25% pullbacks from the top (frequency and timing)
5. **Failed Buy-ups**: Counts instances where sellers stepped in immediately
6. **Point Measurements**: Calculates exact point movements

## 🚀 Quick Start (With Real Yahoo Finance Data)

### Step 1: Install Required Libraries

```bash
pip install yfinance pandas numpy
```

### Step 2: Create the Main Analysis Script

Save this as `spy_analysis_real.py`:

```python
"""
SPY First Hour Volume Analysis - Real Yahoo Finance Data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

def fetch_spy_data(days=180):
    """Fetch SPY hourly data from Yahoo Finance"""
    print(f"Fetching SPY data for last {days} days...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Download hourly data
    spy = yf.download('SPY', start=start_date, end=end_date, interval='1h', progress=True)
    
    print(f"Downloaded {len(spy)} hourly bars")
    return spy

def identify_first_hour(df):
    """Identify first hour of trading (9:30-10:30 AM ET)"""
    df = df.reset_index()
    df['Date'] = df['Datetime'].dt.date
    df['Hour'] = df['Datetime'].dt.hour
    df['Minute'] = df['Datetime'].dt.minute
    
    # First hour is typically the 9:30 hour (hour 9 or hour 13/14 UTC depending on DST)
    # We'll mark the first trading hour of each day
    df['IsFirstHour'] = False
    
    for date in df['Date'].unique():
        day_data = df[df['Date'] == date]
        if len(day_data) > 0:
            first_hour_end = day_data.iloc[0]['Datetime'] + pd.Timedelta(hours=1)
            df.loc[df['Date'] == date, 'IsFirstHour'] = df.loc[df['Date'] == date, 'Datetime'] < first_hour_end
    
    return df

def calculate_volume_metrics(df):
    """Calculate volume percentiles for first hour"""
    first_hour_data = df[df['IsFirstHour']].copy()
    
    # Calculate daily first-hour volume
    daily_volume = first_hour_data.groupby('Date')['Volume'].sum()
    
    # Define "big volume" as 75th percentile
    big_volume_threshold = daily_volume.quantile(0.75)
    
    return big_volume_threshold, daily_volume.describe()

def analyze_first_hour_moves(df, big_volume_threshold):
    """Analyze first hour big volume sell-offs and buy-ups"""
    results = {
        'selloffs': [],
        'buyups': []
    }
    
    dates = df['Date'].unique()
    
    for date in dates:
        day_data = df[df['Date'] == date].copy()
        day_data = day_data.sort_values('Datetime').reset_index(drop=True)
        
        if len(day_data) < 2:
            continue
            
        # Get first hour data
        first_hour = day_data[day_data['IsFirstHour']]
        
        if len(first_hour) == 0:
            continue
        
        # Calculate first hour metrics
        first_hour_volume = first_hour['Volume'].sum()
        first_hour_open = first_hour.iloc[0]['Open']
        first_hour_high = first_hour['High'].max()
        first_hour_low = first_hour['Low'].min()
        first_hour_close = first_hour.iloc[-1]['Close']
        
        # Calculate price change
        price_change = first_hour_close - first_hour_open
        price_change_pct = (price_change / first_hour_open) * 100
        
        # Check if big volume
        if first_hour_volume < big_volume_threshold:
            continue
        
        # Determine if sell-off or buy-up (threshold: 0.3% move)
        if price_change_pct < -0.3:
            # SELL-OFF detected
            analyze_selloff_continuation(day_data, first_hour, first_hour_close, 
                                        first_hour_low, first_hour_open, results['selloffs'], date)
        elif price_change_pct > 0.3:
            # BUY-UP detected
            analyze_buyup_continuation(day_data, first_hour, first_hour_close,
                                      first_hour_high, first_hour_open, results['buyups'], date)
    
    return results

def analyze_selloff_continuation(day_data, first_hour, first_hour_close, first_hour_low, 
                                 first_hour_open, results_list, date):
    """Analyze what happens after a first-hour sell-off"""
    
    # Get data after first hour
    after_first_hour = day_data[~day_data['IsFirstHour']].copy()
    
    if len(after_first_hour) == 0:
        return
    
    first_hour_range = first_hour_open - first_hour_low
    selloff_bottom = first_hour_low
    
    # Initialize tracking variables
    continuation_hours = 0
    chop_hours = 0
    bounce_25_hours = None
    bounce_50_hours = None
    bounce_25_achieved = False
    bounce_50_achieved = False
    failed_selloff = False
    in_chop = False
    continuation_found = False
    
    # Track the absolute bottom after first hour
    absolute_bottom = selloff_bottom
    hours_since_first = 0
    
    for idx, row in after_first_hour.iterrows():
        hours_since_first += 1
        
        # Update absolute bottom
        if row['Low'] < absolute_bottom:
            absolute_bottom = row['Low']
            continuation_hours = hours_since_first
            continuation_found = True
            in_chop = False
        
        # Calculate recovery from absolute bottom
        recovery_from_bottom = row['High'] - absolute_bottom
        recovery_pct = (recovery_from_bottom / first_hour_range) * 100 if first_hour_range > 0 else 0
        
        # Check for 50% bounce
        if not bounce_50_achieved and recovery_pct >= 50:
            bounce_50_hours = hours_since_first
            bounce_50_achieved = True
        
        # Check for 25% bounce
        if not bounce_25_achieved and recovery_pct >= 25:
            bounce_25_hours = hours_since_first
            bounce_25_achieved = True
        
        # Check for chop (price range within 30% of first hour range)
        if continuation_found and not bounce_25_achieved:
            hour_range = row['High'] - row['Low']
            if hour_range < first_hour_range * 0.3:
                if not in_chop:
                    in_chop = True
                chop_hours += 1
    
    # Check if buyers stepped in
    if not continuation_found:
        failed_selloff = True
    
    results_list.append({
        'date': str(date),
        'first_hour_range': round(first_hour_range, 2),
        'continuation_hours': continuation_hours if continuation_found else 0,
        'chop_hours': chop_hours,
        'bounce_25_hours': bounce_25_hours if bounce_25_achieved else None,
        'bounce_50_hours': bounce_50_hours if bounce_50_achieved else None,
        'failed_selloff': failed_selloff,
        'absolute_bottom': round(absolute_bottom, 2)
    })

def analyze_buyup_continuation(day_data, first_hour, first_hour_close, first_hour_high,
                               first_hour_open, results_list, date):
    """Analyze what happens after a first-hour buy-up"""
    
    after_first_hour = day_data[~day_data['IsFirstHour']].copy()
    
    if len(after_first_hour) == 0:
        return
    
    first_hour_range = first_hour_high - first_hour_open
    buyup_top = first_hour_high
    
    continuation_hours = 0
    chop_hours = 0
    pullback_25_hours = None
    pullback_50_hours = None
    pullback_25_achieved = False
    pullback_50_achieved = False
    failed_buyup = False
    in_chop = False
    continuation_found = False
    
    absolute_top = buyup_top
    hours_since_first = 0
    
    for idx, row in after_first_hour.iterrows():
        hours_since_first += 1
        
        if row['High'] > absolute_top:
            absolute_top = row['High']
            continuation_hours = hours_since_first
            continuation_found = True
            in_chop = False
        
        pullback_from_top = absolute_top - row['Low']
        pullback_pct = (pullback_from_top / first_hour_range) * 100 if first_hour_range > 0 else 0
        
        if not pullback_50_achieved and pullback_pct >= 50:
            pullback_50_hours = hours_since_first
            pullback_50_achieved = True
        
        if not pullback_25_achieved and pullback_pct >= 25:
            pullback_25_hours = hours_since_first
            pullback_25_achieved = True
        
        if continuation_found and not pullback_25_achieved:
            hour_range = row['High'] - row['Low']
            if hour_range < first_hour_range * 0.3:
                if not in_chop:
                    in_chop = True
                chop_hours += 1
    
    if not continuation_found:
        failed_buyup = True
    
    results_list.append({
        'date': str(date),
        'first_hour_range': round(first_hour_range, 2),
        'continuation_hours': continuation_hours if continuation_found else 0,
        'chop_hours': chop_hours,
        'pullback_25_hours': pullback_25_hours if pullback_25_achieved else None,
        'pullback_50_hours': pullback_50_hours if pullback_50_achieved else None,
        'failed_buyup': failed_buyup,
        'absolute_top': round(absolute_top, 2)
    })

def calculate_summary_stats(results):
    """Calculate summary statistics"""
    
    selloffs = pd.DataFrame(results['selloffs'])
    buyups = pd.DataFrame(results['buyups'])
    
    summary = {
        'selloffs': {},
        'buyups': {}
    }
    
    # Sell-off statistics
    if len(selloffs) > 0:
        summary['selloffs'] = {
            'total_count': len(selloffs),
            'avg_selloff_points': float(selloffs['first_hour_range'].mean()),
            'avg_continuation_hours': float(selloffs[selloffs['continuation_hours'] > 0]['continuation_hours'].mean()) if len(selloffs[selloffs['continuation_hours'] > 0]) > 0 else 0,
            'continuation_count': int(len(selloffs[selloffs['continuation_hours'] > 0])),
            'avg_chop_hours': float(selloffs[selloffs['chop_hours'] > 0]['chop_hours'].mean()) if len(selloffs[selloffs['chop_hours'] > 0]) > 0 else 0,
            'chop_count': int(len(selloffs[selloffs['chop_hours'] > 0])),
            'bounce_50_count': int(len(selloffs[selloffs['bounce_50_hours'].notna()])),
            'avg_bounce_50_hours': float(selloffs[selloffs['bounce_50_hours'].notna()]['bounce_50_hours'].mean()) if len(selloffs[selloffs['bounce_50_hours'].notna()]) > 0 else 0,
            'bounce_25_count': int(len(selloffs[selloffs['bounce_25_hours'].notna()])),
            'avg_bounce_25_hours': float(selloffs[selloffs['bounce_25_hours'].notna()]['bounce_25_hours'].mean()) if len(selloffs[selloffs['bounce_25_hours'].notna()]) > 0 else 0,
            'failed_selloff_count': int(len(selloffs[selloffs['failed_selloff'] == True]))
        }
    
    # Buy-up statistics
    if len(buyups) > 0:
        summary['buyups'] = {
            'total_count': len(buyups),
            'avg_buyup_points': float(buyups['first_hour_range'].mean()),
            'avg_continuation_hours': float(buyups[buyups['continuation_hours'] > 0]['continuation_hours'].mean()) if len(buyups[buyups['continuation_hours'] > 0]) > 0 else 0,
            'continuation_count': int(len(buyups[buyups['continuation_hours'] > 0])),
            'avg_chop_hours': float(buyups[buyups['chop_hours'] > 0]['chop_hours'].mean()) if len(buyups[buyups['chop_hours'] > 0]) > 0 else 0,
            'chop_count': int(len(buyups[buyups['chop_hours'] > 0])),
            'pullback_50_count': int(len(buyups[buyups['pullback_50_hours'].notna()])),
            'avg_pullback_50_hours': float(buyups[buyups['pullback_50_hours'].notna()]['pullback_50_hours'].mean()) if len(buyups[buyups['pullback_50_hours'].notna()]) > 0 else 0,
            'pullback_25_count': int(len(buyups[buyups['pullback_25_hours'].notna()])),
            'avg_pullback_25_hours': float(buyups[buyups['pullback_25_hours'].notna()]['pullback_25_hours'].mean()) if len(buyups[buyups['pullback_25_hours'].notna()]) > 0 else 0,
            'failed_buyup_count': int(len(buyups[buyups['failed_buyup'] == True]))
        }
    
    return summary, selloffs, buyups

def print_summary(summary):
    """Print summary statistics"""
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\n📉 SELL-OFF ANALYSIS:")
    if summary['selloffs']:
        s = summary['selloffs']
        print(f"   Total sell-offs detected: {s['total_count']}")
        print(f"   Average sell-off size: {s['avg_selloff_points']:.2f} points")
        print(f"   Continued lower: {s['continuation_count']} times ({s['continuation_count']/s['total_count']*100:.1f}%)")
        if s['continuation_count'] > 0:
            print(f"   Average continuation: {s['avg_continuation_hours']:.1f} hours")
        print(f"   Entered chop: {s['chop_count']} times ({s['chop_count']/s['total_count']*100:.1f}%)")
        if s['chop_count'] > 0:
            print(f"   Average chop duration: {s['avg_chop_hours']:.1f} hours")
        print(f"   50% bounce achieved: {s['bounce_50_count']} times ({s['bounce_50_count']/s['total_count']*100:.1f}%)")
        if s['bounce_50_count'] > 0:
            print(f"   Average time to 50% bounce: {s['avg_bounce_50_hours']:.1f} hours")
        print(f"   25% bounce achieved: {s['bounce_25_count']} times ({s['bounce_25_count']/s['total_count']*100:.1f}%)")
        if s['bounce_25_count'] > 0:
            print(f"   Average time to 25% bounce: {s['avg_bounce_25_hours']:.1f} hours")
        print(f"   Failed sell-offs (buyers stepped in): {s['failed_selloff_count']} times ({s['failed_selloff_count']/s['total_count']*100:.1f}%)")
    else:
        print("   No sell-offs detected in the period")
    
    print("\n📈 BUY-UP ANALYSIS:")
    if summary['buyups']:
        b = summary['buyups']
        print(f"   Total buy-ups detected: {b['total_count']}")
        print(f"   Average buy-up size: {b['avg_buyup_points']:.2f} points")
        print(f"   Continued higher: {b['continuation_count']} times ({b['continuation_count']/b['total_count']*100:.1f}%)")
        if b['continuation_count'] > 0:
            print(f"   Average continuation: {b['avg_continuation_hours']:.1f} hours")
        print(f"   Entered chop: {b['chop_count']} times ({b['chop_count']/b['total_count']*100:.1f}%)")
        if b['chop_count'] > 0:
            print(f"   Average chop duration: {b['avg_chop_hours']:.1f} hours")
        print(f"   50% pullback: {b['pullback_50_count']} times ({b['pullback_50_count']/b['total_count']*100:.1f}%)")
        if b['pullback_50_count'] > 0:
            print(f"   Average time to 50% pullback: {b['avg_pullback_50_hours']:.1f} hours")
        print(f"   25% pullback: {b['pullback_25_count']} times ({b['pullback_25_count']/b['total_count']*100:.1f}%)")
        if b['pullback_25_count'] > 0:
            print(f"   Average time to 25% pullback: {b['avg_pullback_25_hours']:.1f} hours")
        print(f"   Failed buy-ups (sellers stepped in): {b['failed_buyup_count']} times ({b['failed_buyup_count']/b['total_count']*100:.1f}%)")
    else:
        print("   No buy-ups detected in the period")

def main():
    print("SPY First Hour Volume Analysis")
    print("=" * 80)
    
    # Fetch data
    df = fetch_spy_data(days=180)
    
    # Identify first hour
    print("\nIdentifying first hour of trading...")
    df = identify_first_hour(df)
    
    # Calculate volume metrics
    print("\nCalculating volume thresholds...")
    big_volume_threshold, volume_stats = calculate_volume_metrics(df)
    print(f"Big volume threshold: {big_volume_threshold:,.0f}")
    
    # Analyze
    print("\nAnalyzing first hour big volume moves...")
    results = analyze_first_hour_moves(df, big_volume_threshold)
    
    # Calculate statistics
    print("\nCalculating summary statistics...")
    summary, selloffs_df, buyups_df = calculate_summary_stats(results)
    
    # Save results
    selloffs_df.to_csv('selloffs_detail.csv', index=False)
    buyups_df.to_csv('buyups_detail.csv', index=False)
    
    with open('summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print_summary(summary)
    
    print("\n✅ Analysis complete! Files saved:")
    print("   - selloffs_detail.csv")
    print("   - buyups_detail.csv")
    print("   - summary_stats.json")

if __name__ == "__main__":
    main()
```

### Step 3: Run the Analysis

```bash
python spy_analysis_real.py
```

### Step 4: Generate the Interactive Dashboard

Use the `create_dashboard.py` script (included) to generate the HTML dashboard:

```bash
python create_dashboard.py
```

This will create `spy_analysis_dashboard.html` that you can open in any web browser.

## 📈 Understanding the Metrics

### Key Definitions

- **Big Volume**: Defined as 75th percentile of first-hour daily volume
- **First Hour**: 9:30 AM - 10:30 AM ET (first hour of market open)
- **Continuation**: Price makes new lows (sell-offs) or new highs (buy-ups) after first hour
- **Chop Zone**: Low volatility period where hourly range < 30% of first-hour range
- **Bounce/Pullback**: Recovery of 25% or 50% of the first-hour move
- **Failed Move**: Price never makes new extremes after first hour

### Color Coding in Dashboard

- **Green/High**: Favorable metric (>60% for continuations, <15% for failures)
- **Yellow/Medium**: Moderate metric (30-60% for continuations, 15-30% for failures)
- **Red/Low**: Unfavorable metric (<30% for continuations, >30% for failures)

## 🔍 How to Use the Results

### For Traders:

1. **Sell-Off Days**: If you see a big volume sell-off in the first hour:
   - Check continuation rate to see if more downside is likely
   - Look at average bounce times to plan entries
   - Note failed sell-off rate for quick reversal opportunities

2. **Buy-Up Days**: If you see a big volume rally in the first hour:
   - Check continuation rate for momentum plays
   - Look at pullback rates to avoid getting caught
   - Note failed buy-up rate for reversal opportunities

3. **Chop Awareness**: High chop rates after moves suggest sideways trading is common

## 📊 Output Files

1. **selloffs_detail.csv**: Every sell-off event with detailed metrics
2. **buyups_detail.csv**: Every buy-up event with detailed metrics
3. **summary_stats.json**: Aggregate statistics in JSON format
4. **spy_analysis_dashboard.html**: Interactive visual dashboard

## 🛠 Customization

### Change Analysis Parameters

Edit these variables in the script:

```python
# Change analysis period (default 180 days)
df = fetch_spy_data(days=365)  # 1 year

# Change volume threshold percentile (default 75th)
big_volume_threshold = daily_volume.quantile(0.80)  # Top 20%

# Change price move threshold (default 0.3%)
if price_change_pct < -0.5:  # More significant moves only

# Change chop definition (default 30% of first hour range)
if hour_range < first_hour_range * 0.2:  # Tighter chop
```

## 📝 Notes

- The analysis uses hourly bars from Yahoo Finance
- First hour is defined as the first trading hour starting at market open
- Volume spikes are relative to each day's first hour, not absolute
- Results may vary with different time periods and market conditions

## 🐛 Troubleshooting

If you encounter issues:

1. **No data fetched**: Check your internet connection
2. **Empty results**: Try lowering the volume threshold percentile
3. **Wrong timezone**: Yahoo Finance data is in market timezone (ET)
4. **Missing hours**: Some days may have less data due to holidays/half days

## 📧 Support

For questions or issues with the analysis, review the code comments or adjust parameters to fit your trading style.

---

**Disclaimer**: This tool is for educational and research purposes only. Past performance does not guarantee future results. Always do your own analysis before making trading decisions.
