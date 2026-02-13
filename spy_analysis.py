"""
SPY First Hour Volume Analysis Dashboard
Analyzes big volume moves in the first hour and subsequent price action
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

# For the demo, I'll create sample data structure
# In real use, you'd fetch from Yahoo Finance using yfinance

def create_sample_data():
    """Create sample SPY hourly data for demonstration"""
    dates = pd.date_range(start='2024-08-16', end='2025-02-12', freq='1h')
    # Filter to market hours (9:30 AM - 4:00 PM ET)
    dates = dates[(dates.hour >= 9) & (dates.hour <= 16)]
    dates = dates[dates.dayofweek < 5]  # Weekdays only
    
    np.random.seed(42)
    n = len(dates)
    
    # Generate realistic SPY price movement
    base_price = 580
    price_walk = np.cumsum(np.random.randn(n) * 0.5)
    
    data = pd.DataFrame({
        'Datetime': dates,
        'Open': base_price + price_walk + np.random.randn(n) * 0.2,
        'High': base_price + price_walk + np.abs(np.random.randn(n)) * 0.5,
        'Low': base_price + price_walk - np.abs(np.random.randn(n)) * 0.5,
        'Close': base_price + price_walk + np.random.randn(n) * 0.2,
        'Volume': np.random.randint(1000000, 10000000, n)
    })
    
    # Ensure High is highest and Low is lowest
    data['High'] = data[['Open', 'High', 'Close']].max(axis=1)
    data['Low'] = data[['Open', 'Low', 'Close']].min(axis=1)
    
    return data

def identify_first_hour(df):
    """Identify first hour of trading (9:30-10:30 AM ET)"""
    df['Date'] = df['Datetime'].dt.date
    df['Hour'] = df['Datetime'].dt.hour
    df['Minute'] = df['Datetime'].dt.minute
    
    # First hour is 9:30 AM hour (hour 9)
    df['IsFirstHour'] = (df['Hour'] == 9) | ((df['Hour'] == 10) & (df['Minute'] < 30))
    return df

def calculate_volume_metrics(df):
    """Calculate volume percentiles for first hour"""
    first_hour_data = df[df['IsFirstHour']].copy()
    
    # Calculate average volume for first hour across all days
    volume_stats = first_hour_data.groupby('Date')['Volume'].sum().describe()
    
    # Define "big volume" as 75th percentile
    big_volume_threshold = volume_stats['75%']
    
    return big_volume_threshold, volume_stats

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
                                        first_hour_low, results['selloffs'], date)
        elif price_change_pct > 0.3:
            # BUY-UP detected
            analyze_buyup_continuation(day_data, first_hour, first_hour_close,
                                      first_hour_high, results['buyups'], date)
    
    return results

def analyze_selloff_continuation(day_data, first_hour, first_hour_close, first_hour_low, results_list, date):
    """Analyze what happens after a first-hour sell-off"""
    
    # Get data after first hour
    after_first_hour = day_data[~day_data['IsFirstHour']].copy()
    
    if len(after_first_hour) == 0:
        return
    
    first_hour_range = first_hour.iloc[0]['Open'] - first_hour_low
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
    
    for idx, row in after_first_hour.iterrows():
        hours_elapsed = idx - first_hour.index[-1]
        
        # Update absolute bottom
        if row['Low'] < absolute_bottom:
            absolute_bottom = row['Low']
            continuation_hours = hours_elapsed
            continuation_found = True
            in_chop = False  # Reset chop if we make new lows
        
        # Calculate recovery from absolute bottom
        recovery_from_bottom = row['High'] - absolute_bottom
        recovery_pct = (recovery_from_bottom / first_hour_range) * 100 if first_hour_range > 0 else 0
        
        # Check for 50% bounce
        if not bounce_50_achieved and recovery_pct >= 50:
            bounce_50_hours = hours_elapsed
            bounce_50_achieved = True
        
        # Check for 25% bounce
        if not bounce_25_achieved and recovery_pct >= 25:
            bounce_25_hours = hours_elapsed
            bounce_25_achieved = True
        
        # Check for chop (price range within 20% of first hour range)
        if continuation_found and not bounce_25_achieved:
            hour_range = row['High'] - row['Low']
            if hour_range < first_hour_range * 0.3:
                if not in_chop:
                    in_chop = True
                chop_hours += 1
    
    # Check if buyers stepped in (never made new lows after first hour)
    if not continuation_found:
        failed_selloff = True
    
    results_list.append({
        'date': date,
        'first_hour_range': round(first_hour_range, 2),
        'continuation_hours': continuation_hours if continuation_found else 0,
        'chop_hours': chop_hours,
        'bounce_25_hours': bounce_25_hours if bounce_25_achieved else None,
        'bounce_50_hours': bounce_50_hours if bounce_50_achieved else None,
        'failed_selloff': failed_selloff,
        'absolute_bottom': round(absolute_bottom, 2)
    })

def analyze_buyup_continuation(day_data, first_hour, first_hour_close, first_hour_high, results_list, date):
    """Analyze what happens after a first-hour buy-up"""
    
    after_first_hour = day_data[~day_data['IsFirstHour']].copy()
    
    if len(after_first_hour) == 0:
        return
    
    first_hour_range = first_hour_high - first_hour.iloc[0]['Open']
    buyup_top = first_hour_high
    
    # Initialize tracking variables
    continuation_hours = 0
    chop_hours = 0
    pullback_25_hours = None
    pullback_50_hours = None
    pullback_25_achieved = False
    pullback_50_achieved = False
    failed_buyup = False
    in_chop = False
    continuation_found = False
    
    # Track the absolute top after first hour
    absolute_top = buyup_top
    
    for idx, row in after_first_hour.iterrows():
        hours_elapsed = idx - first_hour.index[-1]
        
        # Update absolute top
        if row['High'] > absolute_top:
            absolute_top = row['High']
            continuation_hours = hours_elapsed
            continuation_found = True
            in_chop = False
        
        # Calculate pullback from absolute top
        pullback_from_top = absolute_top - row['Low']
        pullback_pct = (pullback_from_top / first_hour_range) * 100 if first_hour_range > 0 else 0
        
        # Check for 50% pullback
        if not pullback_50_achieved and pullback_pct >= 50:
            pullback_50_hours = hours_elapsed
            pullback_50_achieved = True
        
        # Check for 25% pullback
        if not pullback_25_achieved and pullback_pct >= 25:
            pullback_25_hours = hours_elapsed
            pullback_25_achieved = True
        
        # Check for chop
        if continuation_found and not pullback_25_achieved:
            hour_range = row['High'] - row['Low']
            if hour_range < first_hour_range * 0.3:
                if not in_chop:
                    in_chop = True
                chop_hours += 1
    
    # Check if sellers stepped in
    if not continuation_found:
        failed_buyup = True
    
    results_list.append({
        'date': date,
        'first_hour_range': round(first_hour_range, 2),
        'continuation_hours': continuation_hours if continuation_found else 0,
        'chop_hours': chop_hours,
        'pullback_25_hours': pullback_25_hours if pullback_25_achieved else None,
        'pullback_50_hours': pullback_50_hours if pullback_50_achieved else None,
        'failed_buyup': failed_buyup,
        'absolute_top': round(absolute_top, 2)
    })

def calculate_summary_stats(results):
    """Calculate summary statistics for the analysis"""
    
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
            'avg_selloff_points': selloffs['first_hour_range'].mean(),
            'avg_continuation_hours': selloffs[selloffs['continuation_hours'] > 0]['continuation_hours'].mean(),
            'continuation_count': len(selloffs[selloffs['continuation_hours'] > 0]),
            'avg_chop_hours': selloffs[selloffs['chop_hours'] > 0]['chop_hours'].mean(),
            'chop_count': len(selloffs[selloffs['chop_hours'] > 0]),
            'bounce_50_count': len(selloffs[selloffs['bounce_50_hours'].notna()]),
            'avg_bounce_50_hours': selloffs[selloffs['bounce_50_hours'].notna()]['bounce_50_hours'].mean(),
            'bounce_25_count': len(selloffs[selloffs['bounce_25_hours'].notna()]),
            'avg_bounce_25_hours': selloffs[selloffs['bounce_25_hours'].notna()]['bounce_25_hours'].mean(),
            'failed_selloff_count': len(selloffs[selloffs['failed_selloff'] == True])
        }
    
    # Buy-up statistics
    if len(buyups) > 0:
        summary['buyups'] = {
            'total_count': len(buyups),
            'avg_buyup_points': buyups['first_hour_range'].mean(),
            'avg_continuation_hours': buyups[buyups['continuation_hours'] > 0]['continuation_hours'].mean(),
            'continuation_count': len(buyups[buyups['continuation_hours'] > 0]),
            'avg_chop_hours': buyups[buyups['chop_hours'] > 0]['chop_hours'].mean(),
            'chop_count': len(buyups[buyups['chop_hours'] > 0]),
            'pullback_50_count': len(buyups[buyups['pullback_50_hours'].notna()]),
            'avg_pullback_50_hours': buyups[buyups['pullback_50_hours'].notna()]['pullback_50_hours'].mean(),
            'pullback_25_count': len(buyups[buyups['pullback_25_hours'].notna()]),
            'avg_pullback_25_hours': buyups[buyups['pullback_25_hours'].notna()]['pullback_25_hours'].mean(),
            'failed_buyup_count': len(buyups[buyups['failed_buyup'] == True])
        }
    
    return summary, selloffs, buyups

def main():
    print("SPY First Hour Volume Analysis")
    print("=" * 80)
    
    # Create sample data (replace with yfinance download in real use)
    print("\n1. Loading SPY data...")
    df = create_sample_data()
    print(f"   Loaded {len(df)} hourly bars")
    
    # Identify first hour
    print("\n2. Identifying first hour of trading...")
    df = identify_first_hour(df)
    
    # Calculate volume metrics
    print("\n3. Calculating volume thresholds...")
    big_volume_threshold, volume_stats = calculate_volume_metrics(df)
    print(f"   Big volume threshold: {big_volume_threshold:,.0f}")
    
    # Analyze first hour moves
    print("\n4. Analyzing first hour big volume moves...")
    results = analyze_first_hour_moves(df, big_volume_threshold)
    
    # Calculate summary statistics
    print("\n5. Calculating summary statistics...")
    summary, selloffs_df, buyups_df = calculate_summary_stats(results)
    
    # Save results
    selloffs_df.to_csv('selloffs_detail.csv', index=False)
    buyups_df.to_csv('buyups_detail.csv', index=False)
    
    with open('summary_stats.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    print("\n📉 SELL-OFF ANALYSIS:")
    if summary['selloffs']:
        s = summary['selloffs']
        print(f"   Total sell-offs detected: {s['total_count']}")
        print(f"   Average sell-off size: {s['avg_selloff_points']:.2f} points")
        print(f"   Continued lower: {s['continuation_count']} times ({s['continuation_count']/s['total_count']*100:.1f}%)")
        print(f"   Average continuation: {s['avg_continuation_hours']:.1f} hours")
        print(f"   Entered chop: {s['chop_count']} times ({s['chop_count']/s['total_count']*100:.1f}%)")
        print(f"   Average chop duration: {s['avg_chop_hours']:.1f} hours")
        print(f"   50% bounce achieved: {s['bounce_50_count']} times ({s['bounce_50_count']/s['total_count']*100:.1f}%)")
        print(f"   Average time to 50% bounce: {s['avg_bounce_50_hours']:.1f} hours")
        print(f"   25% bounce achieved: {s['bounce_25_count']} times ({s['bounce_25_count']/s['total_count']*100:.1f}%)")
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
        print(f"   Average continuation: {b['avg_continuation_hours']:.1f} hours")
        print(f"   Entered chop: {b['chop_count']} times ({b['chop_count']/b['total_count']*100:.1f}%)")
        print(f"   Average chop duration: {b['avg_chop_hours']:.1f} hours")
        print(f"   50% pullback: {b['pullback_50_count']} times ({b['pullback_50_count']/b['total_count']*100:.1f}%)")
        print(f"   Average time to 50% pullback: {b['avg_pullback_50_hours']:.1f} hours")
        print(f"   25% pullback: {b['pullback_25_count']} times ({b['pullback_25_count']/b['total_count']*100:.1f}%)")
        print(f"   Average time to 25% pullback: {b['avg_pullback_25_hours']:.1f} hours")
        print(f"   Failed buy-ups (sellers stepped in): {b['failed_buyup_count']} times ({b['failed_buyup_count']/b['total_count']*100:.1f}%)")
    else:
        print("   No buy-ups detected in the period")
    
    print("\n✅ Analysis complete! Files saved:")
    print("   - selloffs_detail.csv")
    print("   - buyups_detail.csv")
    print("   - summary_stats.json")

if __name__ == "__main__":
    main()
