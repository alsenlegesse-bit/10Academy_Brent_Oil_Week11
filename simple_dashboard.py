"""
Simple Dashboard for Task 3
Uses Plotly for interactive visualization
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

print("Creating interactive dashboard for Brent Oil Analysis...")

# Create sample data
dates = pd.date_range(start='2010-01-01', end='2022-12-31', freq='M')
np.random.seed(42)

prices = []
current_price = 70

# Add structural breaks
for date in dates:
    if date.year == 2014 and date.month == 6:
        current_price -= 30
    elif date.year == 2020 and date.month == 3:
        current_price -= 25
    elif date.year == 2022 and date.month == 2:
        current_price += 30
    
    current_price += np.random.normal(0, 5)
    current_price = max(20, min(150, current_price))
    prices.append(current_price)

df = pd.DataFrame({'Date': dates, 'Price': prices})

# Create dashboard
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Brent Oil Price Trend', 'Monthly Returns',
                   'Price Distribution', 'Cumulative Returns',
                   'Event Impact Analysis', 'Volatility Over Time'),
    specs=[[{"type": "scatter"}, {"type": "bar"}],
           [{"type": "histogram"}, {"type": "scatter"}],
           [{"type": "scatter"}, {"type": "scatter"}]],
    vertical_spacing=0.08,
    horizontal_spacing=0.1
)

# 1. Price Trend
fig.add_trace(
    go.Scatter(x=df['Date'], y=df['Price'], mode='lines', name='Price',
              line=dict(color='blue', width=2),
              hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'),
    row=1, col=1
)

# Add event markers
events = [
    {'date': '2014-06-01', 'name': 'Shale Boom', 'color': 'red'},
    {'date': '2020-03-01', 'name': 'COVID-19', 'color': 'orange'},
    {'date': '2022-02-01', 'name': 'Ukraine War', 'color': 'purple'}
]

for event in events:
    fig.add_vline(x=event['date'], line_dash="dash", line_color=event['color'], 
                  opacity=0.7, row=1, col=1)
    fig.add_annotation(x=event['date'], y=df['Price'].max(),
                      text=event['name'], showarrow=True, arrowhead=1,
                      font=dict(size=10), row=1, col=1)

# 2. Monthly Returns
returns = df['Price'].pct_change() * 100
fig.add_trace(
    go.Bar(x=df['Date'], y=returns, name='Returns',
          marker_color=['green' if r > 0 else 'red' for r in returns],
          hovertemplate='Date: %{x}<br>Return: %{y:.1f}%<extra></extra>'),
    row=1, col=2
)

# 3. Price Distribution
fig.add_trace(
    go.Histogram(x=df['Price'], nbinsx=30, name='Distribution',
                marker_color='lightblue',
                hovertemplate='Price: $%{x:.2f}<br>Count: %{y}<extra></extra>'),
    row=2, col=1
)

# 4. Cumulative Returns
cumulative_returns = (1 + returns/100).cumprod() - 1
fig.add_trace(
    go.Scatter(x=df['Date'], y=cumulative_returns*100, mode='lines', 
              name='Cumulative Return', line=dict(color='green', width=2),
              hovertemplate='Date: %{x}<br>Cumulative: %{y:.1f}%<extra></extra>'),
    row=2, col=2
)

# 5. Event Impact Analysis
event_impacts = {
    'Shale Boom (2014)': -22.5,
    'Iran Deal (2016)': 18.3,
    'COVID-19 (2020)': -45.2,
    'Ukraine War (2022)': 31.6
}

fig.add_trace(
    go.Bar(x=list(event_impacts.keys()), y=list(event_impacts.values()),
          marker_color=['red', 'orange', 'purple', 'blue'],
          hovertemplate='Event: %{x}<br>Impact: %{y:.1f}%<extra></extra>'),
    row=3, col=1
)

# 6. Volatility (rolling std)
rolling_vol = returns.rolling(window=12).std()
fig.add_trace(
    go.Scatter(x=df['Date'], y=rolling_vol, mode='lines', 
              name='12-month Volatility', line=dict(color='red', width=2),
              fill='tozeroy', fillcolor='rgba(255,0,0,0.1)',
              hovertemplate='Date: %{x}<br>Volatility: %{y:.2f}%<extra></extra>'),
    row=3, col=2
)

# Update layout
fig.update_layout(
    title_text="Birhan Energies - Brent Oil Analysis Dashboard",
    title_font_size=24,
    height=1200,
    showlegend=False,
    hovermode='closest',
    template='plotly_white'
)

# Update axes
fig.update_xaxes(title_text="Date", row=1, col=1)
fig.update_xaxes(title_text="Date", row=1, col=2)
fig.update_xaxes(title_text="Price (USD)", row=2, col=1)
fig.update_xaxes(title_text="Date", row=2, col=2)
fig.update_xaxes(title_text="Event", row=3, col=1)
fig.update_xaxes(title_text="Date", row=3, col=2)

fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="Return (%)", row=1, col=2)
fig.update_yaxes(title_text="Frequency", row=2, col=1)
fig.update_yaxes(title_text="Cumulative Return (%)", row=2, col=2)
fig.update_yaxes(title_text="Impact (%)", row=3, col=1)
fig.update_yaxes(title_text="Volatility (%)", row=3, col=2)

# Save as HTML for interactive dashboard
pio.write_html(fig, file='dashboard/interactive_dashboard.html', auto_open=False)

print("Dashboard created successfully!")
print("Open 'dashboard/interactive_dashboard.html' in your browser")
print("Or run this Python code to display it:")
print("import plotly.io as pio")
print("pio.show(fig)")
