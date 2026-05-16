import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Create some dummy datetime data
dates = pd.date_range("2025-01-01", periods=10, freq="D")
y = np.random.randn(10)

fig = go.Figure(go.Scatter(x=dates, y=y))

# Try adding vline with different types
try:
    fig.add_vline(x=dates[5], line_dash="dot")
    print("SUCCESS with Timestamp object")
except Exception as e:
    print(f"FAILED with Timestamp object: {e}")

try:
    fig_str = go.Figure(go.Scatter(x=dates, y=y))
    fig_str.add_vline(x=dates[5].strftime("%Y-%m-%d"), line_dash="dot")
    print("SUCCESS with string")
except Exception as e:
    print(f"FAILED with string: {e}")

try:
    fig_float = go.Figure(go.Scatter(x=dates, y=y))
    fig_float.add_vline(x=dates[5].timestamp() * 1000, line_dash="dot")
    print("SUCCESS with float timestamp")
except Exception as e:
    print(f"FAILED with float timestamp: {e}")

