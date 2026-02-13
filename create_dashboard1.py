"""
Create Interactive HTML Dashboard for SPY Analysis
"""

import pandas as pd
import json
from datetime import datetime

def create_html_dashboard():
    """Generate interactive HTML dashboard"""
    
    # Load data
    try:
        selloffs_df = pd.read_csv('selloffs_detail.csv')
    except:
        selloffs_df = pd.DataFrame()
    
    try:
        buyups_df = pd.read_csv('buyups_detail.csv')
    except:
        buyups_df = pd.DataFrame()
    
    try:
        with open('summary_stats.json', 'r') as f:
            summary = json.load(f)
    except:
        summary = {'selloffs': {}, 'buyups': {}}
    
    # Generate HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SPY First Hour Volume Analysis Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
            padding: 20px;
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
        }}
        
        .header {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            color: #1e3c72;
            margin-bottom: 10px;
        }}
        
        .header p {{
            color: #666;
            font-size: 1.1em;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .stat-card {{
            background: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0,0,0,0.15);
        }}
        
        .stat-card.selloff {{
            border-left: 5px solid #e74c3c;
        }}
        
        .stat-card.buyup {{
            border-left: 5px solid #27ae60;
        }}
        
        .stat-card h3 {{
            font-size: 1.3em;
            margin-bottom: 15px;
            color: #333;
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            margin: 10px 0;
        }}
        
        .stat-card.selloff .stat-value {{
            color: #e74c3c;
        }}
        
        .stat-card.buyup .stat-value {{
            color: #27ae60;
        }}
        
        .stat-label {{
            font-size: 0.9em;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        
        .section {{
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }}
        
        .section h2 {{
            font-size: 1.8em;
            margin-bottom: 20px;
            color: #1e3c72;
            border-bottom: 3px solid #1e3c72;
            padding-bottom: 10px;
        }}
        
        .metrics-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        .metrics-table th {{
            background: #f8f9fa;
            padding: 15px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid #dee2e6;
        }}
        
        .metrics-table td {{
            padding: 12px 15px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .metrics-table tr:hover {{
            background: #f8f9fa;
        }}
        
        .percentage {{
            font-weight: 600;
            padding: 5px 10px;
            border-radius: 5px;
            display: inline-block;
        }}
        
        .percentage.high {{
            background: #d4edda;
            color: #155724;
        }}
        
        .percentage.medium {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .percentage.low {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        .detail-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
            font-size: 0.9em;
        }}
        
        .detail-table th {{
            background: #e9ecef;
            padding: 10px;
            text-align: left;
            font-weight: 600;
        }}
        
        .detail-table td {{
            padding: 8px 10px;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .detail-table tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        
        .tabs {{
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }}
        
        .tab {{
            padding: 12px 24px;
            background: #e9ecef;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: 600;
            transition: all 0.3s;
        }}
        
        .tab:hover {{
            background: #dee2e6;
        }}
        
        .tab.active {{
            background: #1e3c72;
            color: white;
        }}
        
        .tab-content {{
            display: none;
        }}
        
        .tab-content.active {{
            display: block;
        }}
        
        .legend {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin-top: 20px;
        }}
        
        .legend h4 {{
            margin-bottom: 15px;
            color: #1e3c72;
        }}
        
        .legend-item {{
            margin: 8px 0;
            padding: 8px;
            background: white;
            border-radius: 5px;
        }}
        
        .legend-item strong {{
            color: #1e3c72;
        }}
        
        @media (max-width: 768px) {{
            .stats-grid {{
                grid-template-columns: 1fr;
            }}
            
            .header h1 {{
                font-size: 1.8em;
            }}
            
            .stat-value {{
                font-size: 2em;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 SPY First Hour Volume Analysis</h1>
            <p>Comprehensive analysis of big volume moves in the first trading hour and subsequent price action</p>
            <p style="margin-top: 10px; color: #999;">Analysis Period: Last 180 Days | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
"""
    
    # Add sell-off summary stats
    if summary.get('selloffs'):
        s = summary['selloffs']
        html += f"""
        <div class="stats-grid">
            <div class="stat-card selloff">
                <div class="stat-label">Total Sell-Offs Detected</div>
                <div class="stat-value">{s['total_count']}</div>
            </div>
            <div class="stat-card selloff">
                <div class="stat-label">Average Sell-Off Size</div>
                <div class="stat-value">{s['avg_selloff_points']:.2f}</div>
                <div class="stat-label">Points</div>
            </div>
            <div class="stat-card selloff">
                <div class="stat-label">Continuation Rate</div>
                <div class="stat-value">{s['continuation_count']/s['total_count']*100:.1f}%</div>
                <div class="stat-label">{s['continuation_count']} times</div>
            </div>
            <div class="stat-card selloff">
                <div class="stat-label">50% Bounce Rate</div>
                <div class="stat-value">{s['bounce_50_count']/s['total_count']*100:.1f}%</div>
                <div class="stat-label">{s['bounce_50_count']} times</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📉 Sell-Off Detailed Metrics</h2>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Details</th>
                </tr>
                <tr>
                    <td><strong>Continued Lower</strong></td>
                    <td><span class="percentage {'high' if s['continuation_count']/s['total_count'] > 0.6 else 'medium' if s['continuation_count']/s['total_count'] > 0.3 else 'low'}">{s['continuation_count']/s['total_count']*100:.1f}%</span></td>
                    <td>{s['continuation_count']} out of {s['total_count']} times | Avg: {s['avg_continuation_hours']:.1f} hours</td>
                </tr>
                <tr>
                    <td><strong>Entered Chop Zone</strong></td>
                    <td><span class="percentage {'high' if s['chop_count']/s['total_count'] > 0.4 else 'medium' if s['chop_count']/s['total_count'] > 0.2 else 'low'}">{s['chop_count']/s['total_count']*100:.1f}%</span></td>
                    <td>{s['chop_count']} times | Avg duration: {s['avg_chop_hours']:.1f} hours</td>
                </tr>
                <tr>
                    <td><strong>50% Bounce Back</strong></td>
                    <td><span class="percentage {'high' if s['bounce_50_count']/s['total_count'] > 0.5 else 'medium' if s['bounce_50_count']/s['total_count'] > 0.25 else 'low'}">{s['bounce_50_count']/s['total_count']*100:.1f}%</span></td>
                    <td>{s['bounce_50_count']} times | Avg time: {s['avg_bounce_50_hours']:.1f} hours</td>
                </tr>
                <tr>
                    <td><strong>25% Bounce Back</strong></td>
                    <td><span class="percentage {'high' if s['bounce_25_count']/s['total_count'] > 0.6 else 'medium' if s['bounce_25_count']/s['total_count'] > 0.3 else 'low'}">{s['bounce_25_count']/s['total_count']*100:.1f}%</span></td>
                    <td>{s['bounce_25_count']} times | Avg time: {s['avg_bounce_25_hours']:.1f} hours</td>
                </tr>
                <tr>
                    <td><strong>Failed Sell-Offs (Buyers Stepped In)</strong></td>
                    <td><span class="percentage {'low' if s['failed_selloff_count']/s['total_count'] > 0.3 else 'medium' if s['failed_selloff_count']/s['total_count'] > 0.15 else 'high'}">{s['failed_selloff_count']/s['total_count']*100:.1f}%</span></td>
                    <td>{s['failed_selloff_count']} times - Price never made new lows after first hour</td>
                </tr>
            </table>
        </div>
"""
    
    # Add buy-up summary stats
    if summary.get('buyups'):
        b = summary['buyups']
        html += f"""
        <div class="stats-grid">
            <div class="stat-card buyup">
                <div class="stat-label">Total Buy-Ups Detected</div>
                <div class="stat-value">{b['total_count']}</div>
            </div>
            <div class="stat-card buyup">
                <div class="stat-label">Average Buy-Up Size</div>
                <div class="stat-value">{b['avg_buyup_points']:.2f}</div>
                <div class="stat-label">Points</div>
            </div>
            <div class="stat-card buyup">
                <div class="stat-label">Continuation Rate</div>
                <div class="stat-value">{b['continuation_count']/b['total_count']*100:.1f}%</div>
                <div class="stat-label">{b['continuation_count']} times</div>
            </div>
            <div class="stat-card buyup">
                <div class="stat-label">50% Pullback Rate</div>
                <div class="stat-value">{b['pullback_50_count']/b['total_count']*100:.1f}%</div>
                <div class="stat-label">{b['pullback_50_count']} times</div>
            </div>
        </div>
        
        <div class="section">
            <h2>📈 Buy-Up Detailed Metrics</h2>
            <table class="metrics-table">
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                    <th>Details</th>
                </tr>
                <tr>
                    <td><strong>Continued Higher</strong></td>
                    <td><span class="percentage {'high' if b['continuation_count']/b['total_count'] > 0.6 else 'medium' if b['continuation_count']/b['total_count'] > 0.3 else 'low'}">{b['continuation_count']/b['total_count']*100:.1f}%</span></td>
                    <td>{b['continuation_count']} out of {b['total_count']} times | Avg: {b['avg_continuation_hours']:.1f} hours</td>
                </tr>
                <tr>
                    <td><strong>Entered Chop Zone</strong></td>
                    <td><span class="percentage {'high' if b['chop_count']/b['total_count'] > 0.4 else 'medium' if b['chop_count']/b['total_count'] > 0.2 else 'low'}">{b['chop_count']/b['total_count']*100:.1f}%</span></td>
                    <td>{b['chop_count']} times | Avg duration: {b['avg_chop_hours']:.1f} hours</td>
                </tr>
                <tr>
                    <td><strong>50% Pullback</strong></td>
                    <td><span class="percentage {'low' if b['pullback_50_count']/b['total_count'] > 0.5 else 'medium' if b['pullback_50_count']/b['total_count'] > 0.25 else 'high'}">{b['pullback_50_count']/b['total_count']*100:.1f}%</span></td>
                    <td>{b['pullback_50_count']} times | Avg time: {b['avg_pullback_50_hours']:.1f} hours</td>
                </tr>
                <tr>
                    <td><strong>25% Pullback</strong></td>
                    <td><span class="percentage {'low' if b['pullback_25_count']/b['total_count'] > 0.6 else 'medium' if b['pullback_25_count']/b['total_count'] > 0.3 else 'high'}">{b['pullback_25_count']/b['total_count']*100:.1f}%</span></td>
                    <td>{b['pullback_25_count']} times | Avg time: {b['avg_pullback_25_hours']:.1f} hours</td>
                </tr>
                <tr>
                    <td><strong>Failed Buy-Ups (Sellers Stepped In)</strong></td>
                    <td><span class="percentage {'low' if b['failed_buyup_count']/b['total_count'] > 0.3 else 'medium' if b['failed_buyup_count']/b['total_count'] > 0.15 else 'high'}">{b['failed_buyup_count']/b['total_count']*100:.1f}%</span></td>
                    <td>{b['failed_buyup_count']} times - Price never made new highs after first hour</td>
                </tr>
            </table>
        </div>
"""
    
    # Add detailed data tables
    html += """
        <div class="section">
            <h2>📋 Detailed Event Analysis</h2>
            <div class="tabs">
                <button class="tab active" onclick="showTab('selloffs')">Sell-Off Events</button>
                <button class="tab" onclick="showTab('buyups')">Buy-Up Events</button>
            </div>
            
            <div id="selloffs-content" class="tab-content active">
"""
    
    if len(selloffs_df) > 0:
        html += """
                <table class="detail-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Sell-Off Points</th>
                            <th>Continuation (hrs)</th>
                            <th>Chop (hrs)</th>
                            <th>25% Bounce (hrs)</th>
                            <th>50% Bounce (hrs)</th>
                            <th>Failed?</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        for _, row in selloffs_df.iterrows():
            html += f"""
                        <tr>
                            <td>{row['date']}</td>
                            <td>{row['first_hour_range']:.2f}</td>
                            <td>{row['continuation_hours']}</td>
                            <td>{row['chop_hours']}</td>
                            <td>{row['bounce_25_hours'] if pd.notna(row['bounce_25_hours']) else '-'}</td>
                            <td>{row['bounce_50_hours'] if pd.notna(row['bounce_50_hours']) else '-'}</td>
                            <td>{'Yes' if row['failed_selloff'] else 'No'}</td>
                        </tr>
"""
        html += """
                    </tbody>
                </table>
"""
    else:
        html += "<p>No sell-off events detected in the analysis period.</p>"
    
    html += """
            </div>
            
            <div id="buyups-content" class="tab-content">
"""
    
    if len(buyups_df) > 0:
        html += """
                <table class="detail-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th>Buy-Up Points</th>
                            <th>Continuation (hrs)</th>
                            <th>Chop (hrs)</th>
                            <th>25% Pullback (hrs)</th>
                            <th>50% Pullback (hrs)</th>
                            <th>Failed?</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        for _, row in buyups_df.iterrows():
            html += f"""
                        <tr>
                            <td>{row['date']}</td>
                            <td>{row['first_hour_range']:.2f}</td>
                            <td>{row['continuation_hours']}</td>
                            <td>{row['chop_hours']}</td>
                            <td>{row['pullback_25_hours'] if pd.notna(row['pullback_25_hours']) else '-'}</td>
                            <td>{row['pullback_50_hours'] if pd.notna(row['pullback_50_hours']) else '-'}</td>
                            <td>{'Yes' if row['failed_buyup'] else 'No'}</td>
                        </tr>
"""
        html += """
                    </tbody>
                </table>
"""
    else:
        html += "<p>No buy-up events detected in the analysis period.</p>"
    
    html += """
            </div>
        </div>
        
        <div class="section">
            <div class="legend">
                <h4>📖 Metrics Explanation</h4>
                <div class="legend-item">
                    <strong>Big Volume Threshold:</strong> Defined as 75th percentile of first-hour daily volume
                </div>
                <div class="legend-item">
                    <strong>First Hour:</strong> 9:30 AM - 10:30 AM ET trading session
                </div>
                <div class="legend-item">
                    <strong>Continuation:</strong> Price continues in the direction of the first-hour move (new lows for sell-offs, new highs for buy-ups)
                </div>
                <div class="legend-item">
                    <strong>Chop Zone:</strong> Period where hourly price range is less than 30% of the first-hour range (low volatility consolidation)
                </div>
                <div class="legend-item">
                    <strong>25%/50% Bounce/Pullback:</strong> Recovery of 25% or 50% of the first-hour move from the absolute bottom (sell-offs) or top (buy-ups)
                </div>
                <div class="legend-item">
                    <strong>Failed Move:</strong> Price never made new extremes after the first hour (buyers/sellers stepped in immediately)
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function showTab(tabName) {
            // Hide all tab contents
            const contents = document.querySelectorAll('.tab-content');
            contents.forEach(content => content.classList.remove('active'));
            
            // Deactivate all tabs
            const tabs = document.querySelectorAll('.tab');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Show selected tab content
            document.getElementById(tabName + '-content').classList.add('active');
            
            // Activate selected tab
            event.target.classList.add('active');
        }
    </script>
</body>
</html>
"""
    
    # Save HTML file
    with open('spy_analysis_dashboard.html', 'w') as f:
        f.write(html)
    
    print("✅ Interactive HTML dashboard created: spy_analysis_dashboard.html")

if __name__ == "__main__":
    create_html_dashboard()
