"""
MCA Financial Analytics Capstone Project
Risk Analytics Dashboard - 10 Module Professional Dashboard
Built with Python, Streamlit, Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Page config
st.set_page_config(
    page_title="Risk Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; }
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e, #16213e);
        border: 1px solid #2d3561;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 4px 0;
        text-align: center;
    }
    .kpi-label { color: #8892b0; font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { color: #ccd6f6; font-size: 22px; font-weight: 700; margin: 4px 0; }
    .kpi-change { font-size: 11px; }
    .positive { color: #64ffda; }
    .negative { color: #ff6b6b; }
    .neutral  { color: #ffd93d; }
    .signal-buy  { background: linear-gradient(135deg, #0d4f3c, #1a7a5e); border: 2px solid #64ffda; border-radius: 10px; padding: 16px; text-align: center; }
    .signal-sell { background: linear-gradient(135deg, #4f0d0d, #7a1a1a); border: 2px solid #ff6b6b; border-radius: 10px; padding: 16px; text-align: center; }
    .signal-hold { background: linear-gradient(135deg, #4f3d0d, #7a5c1a); border: 2px solid #ffd93d; border-radius: 10px; padding: 16px; text-align: center; }
    .signal-text { color: white; font-size: 28px; font-weight: 900; letter-spacing: 3px; }
    .module-header {
        background: linear-gradient(90deg, #1a1f2e, #16213e);
        border-left: 4px solid #64ffda;
        padding: 8px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0 16px 0;
    }
    .module-title  { color: #64ffda; font-size: 16px; font-weight: 700; margin: 0; }
    .module-marks  { color: #8892b0; font-size: 12px; }
    .metric-row {
        background: #1a1f2e;
        border: 1px solid #2d3561;
        border-radius: 8px;
        padding: 10px 14px;
        text-align: center;
        margin-bottom: 6px;
    }
    .metric-label { color: #8892b0; font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-value { color: #ccd6f6; font-size: 18px; font-weight: 700; }
    div[data-testid="stSidebar"] { background-color: #0a0e1a !important; }
    .insight-box {
        background: #1a1f2e;
        border: 1px solid #2d3561;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 4px 0;
        color: #ccd6f6;
        font-size: 13px;
    }
    .footer { text-align: center; color: #4a5568; font-size: 11px; margin-top: 20px; padding: 10px; border-top: 1px solid #2d3561; }
</style>
""", unsafe_allow_html=True)

# ── Import modules ──────────────────────────────────────────
from modules.data_loader import load_data, get_ticker_info
from modules.module1_executive import render_executive_summary
from modules.module2_arima import render_arima
from modules.module3_garch import render_garch
from modules.module4_dcf import render_dcf
from modules.module5_montecarlo import render_monte_carlo
from modules.module6_var import render_var
from modules.module7_credit import render_credit_risk
from modules.module8_portfolio import render_portfolio
from modules.module9_stress import render_stress_testing
from modules.module10_heatmap import render_heatmap

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0;'>
        <h2 style='color:#64ffda; margin:0;'>📊 RISK ANALYTICS</h2>
        <p style='color:#8892b0; font-size:11px; margin:4px 0;'>Comprehensive Risk & Investment Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    TICKERS = {
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS":       "Tata Consultancy Services",
        "INFY.NS":      "Infosys",
        "HDFCBANK.NS":  "HDFC Bank",
        "WIPRO.NS":     "Wipro",
    }

    selected_ticker = st.selectbox(
        "🎯 Select Ticker",
        list(TICKERS.keys()),
        format_func=lambda x: f"{x} — {TICKERS[x]}",
        key="ticker"
    )

    from datetime import datetime, timedelta
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("📅 Start Date",
            value=datetime.now() - timedelta(days=730),
            max_value=datetime.now() - timedelta(days=1))
    with col2:
        end_date = st.date_input("📅 End Date",
            value=datetime.now(), max_value=datetime.now())

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#4a5568;'>
    <b style='color:#8892b0;'>Modules:</b><br>
    1. Executive Summary Panel<br>
    2. ARIMA Forecasting<br>
    3. GARCH Volatility Modeling<br>
    4. DCF Valuation<br>
    5. Monte Carlo Simulation<br>
    6. Value at Risk (VaR)<br>
    7. Credit Risk Modeling<br>
    8. Portfolio Optimization<br>
    9. Stress Testing<br>
    10. Correlation Heatmap
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("*All calculations based on daily data | Source: Yahoo Finance / Synthetic NSE data*")

# ── Load data ───────────────────────────────────────────────
@st.cache_data(ttl=300, show_spinner=False)
def get_data(ticker, start, end):
    return load_data(ticker, str(start), str(end))

with st.spinner(f"⏳ Loading data for {selected_ticker}..."):
    try:
        df, all_tickers_df = get_data(selected_ticker, start_date, end_date)
        data_ok = df is not None and len(df) > 30
    except Exception as e:
        data_ok = False
        st.error(f"Data loading error: {e}")

if not data_ok:
    st.error("❌ Failed to load or generate data. Please try a different date range.")
    st.stop()

# Banner if using synthetic data
try:
    real_data = False
    import yfinance as yf
    test = yf.download(selected_ticker, period="5d", progress=False)
    if test is not None and len(test) > 2:
        real_data = True
except Exception:
    pass

if not real_data:
    st.warning(
        "⚠️ **Yahoo Finance is not reachable in this environment.** "
        "The dashboard is running on **realistic synthetic NSE data** (GBM-based) "
        "so all 10 modules are fully functional. On a machine with internet access, "
        "it will automatically switch to live data.",
        icon="📡"
    )

# ── Tabs ─────────────────────────────────────────────────────
tabs = st.tabs([
    "📋 Overview",
    "📈 ARIMA",
    "🌊 GARCH",
    "💰 DCF",
    "🎲 Monte Carlo",
    "⚠️ VaR",
    "🏦 Credit Risk",
    "📊 Portfolio",
    "🔥 Stress Test",
    "🔗 Correlation"
])

with tabs[0]:
    render_executive_summary(df, selected_ticker, all_tickers_df)
with tabs[1]:
    render_arima(df, selected_ticker)
with tabs[2]:
    render_garch(df, selected_ticker)
with tabs[3]:
    render_dcf(df, selected_ticker)
with tabs[4]:
    render_monte_carlo(df, selected_ticker)
with tabs[5]:
    render_var(df, selected_ticker)
with tabs[6]:
    render_credit_risk(df, selected_ticker)
with tabs[7]:
    render_portfolio(all_tickers_df, selected_ticker)
with tabs[8]:
    render_stress_testing(df, all_tickers_df, selected_ticker)
with tabs[9]:
    render_heatmap(all_tickers_df, selected_ticker)

st.markdown("""
<div class='footer'>
    Risk Analytics Dashboard | Built with Python, Streamlit, Plotly | MCA Financial Analytics Capstone Project
</div>
""", unsafe_allow_html=True)
