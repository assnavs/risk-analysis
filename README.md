# Risk Analytics Dashboard
**MCA · Financial Analytics · Capstone Project**

> A fully interactive 10-module Risk Analytics Dashboard built with Python, Streamlit, and Plotly — analysing NSE/BSE listed stocks in real time.

---

## 🚀 Quick Start

```bash
# 1. Clone the repository
git clone <your-repo-url>
cd risk_dashboard

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501` in your browser.

---

## 📋 Project Overview

| Field | Detail |
|-------|--------|
| **Total Marks** | 150 (130 base + 20 bonus for live deployment) |
| **Duration** | 10 Days |
| **Framework** | Streamlit + Plotly |
| **Data Source** | Yahoo Finance (yfinance), NSE |
| **Tickers** | RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, WIPRO.NS |

---

## 📊 Dashboard Modules

| Module | Name | Marks |
|--------|------|-------|
| 1 | Executive Summary Panel | 15 |
| 2 | ARIMA Forecasting | 15 |
| 3 | GARCH Volatility Modeling | 12 |
| 4 | DCF Valuation | 13 |
| 5 | Monte Carlo Simulation | 13 |
| 6 | Value at Risk (VaR) | 15 |
| 7 | Credit Risk Modeling | 13 |
| 8 | Portfolio Optimization | 12 |
| 9 | Stress Testing & Scenario Analysis | 10 |
| 10 | Correlation Heatmap | 8 |
| **Total** | | **126** |
| **Code Quality** | Docstrings, modular code, requirements | 20 |
| **Data Pipeline** | Clean loading, log returns | 10 |
| **Bonus** | Live deployment | +20 |

---

## 🏗️ Project Structure

```
risk_dashboard/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Pinned dependencies
├── README.md                   # This file
└── modules/
    ├── __init__.py
    ├── data_loader.py           # Data fetching & preprocessing
    ├── module1_executive.py    # Executive Summary Panel
    ├── module2_arima.py        # ARIMA Forecasting
    ├── module3_garch.py        # GARCH Volatility Modeling
    ├── module4_dcf.py          # DCF Valuation
    ├── module5_montecarlo.py   # Monte Carlo Simulation
    ├── module6_var.py          # Value at Risk
    ├── module7_credit.py       # Credit Risk Modeling
    ├── module8_portfolio.py    # Portfolio Optimization
    ├── module9_stress.py       # Stress Testing
    └── module10_heatmap.py     # Correlation Heatmap
```

---

## 🔬 Module Details

### Module 1 — Executive Summary Panel
- KPI cards: Current Price, ARIMA Forecast, DCF Intrinsic Value with sparklines
- Live metrics: Expected Return, Portfolio Return, Risk, VaR 95%, Probability of Default, Sharpe Ratio
- Investment Signal: BUY/HOLD/SELL logic with margin of safety and VaR thresholds
- Risk Level: LOW / MEDIUM / HIGH from VaR percentile
- Date picker + Ticker selector (5 NSE tickers)

### Module 2 — ARIMA Forecasting
- `pmdarima.auto_arima` with `stepwise=True, seasonal=False`
- 90-day forecast with 95% confidence interval (Plotly)
- Metrics: RMSE, MAE, MAPE, Direction
- Walk-forward validation: 80% train / 20% test, in-sample vs OOS RMSE

### Module 3 — GARCH Volatility Modeling
- `arch` library GARCH(1,1) on log returns
- Annualised conditional volatility + 20-day rolling volatility
- Highest spike marked with vertical dashed line + annotation
- Regime detection: High (>P75), Low (<P25), Moderate otherwise

### Module 4 — DCF Valuation
- Real operating cash flow from `yfinance.cashflow`
- User sliders: Forecast Period (1–10y), WACC (5–25%), Terminal Growth Rate (1–5%)
- Gordon Growth Model: TV = FCF_n × (1 + g) / (WACC - g)
- Waterfall bar chart: Year 1–5 PV (green), Terminal Value (blue), Enterprise Value (purple)
- Margin of Safety indicator: Undervalued / Fairly Valued / Overvalued

### Module 5 — Monte Carlo Simulation
- GBM: S(t+1) = S(t) × exp((μ - 0.5σ²)dt + σ√dt × Z), Z ~ N(0,1)
- 1,000–10,000 paths configurable, 63–252 day horizon
- Top 5% in green, bottom 5% in red, others translucent blue
- Probability summary: Expected Price, Median, P95, P5, P(+10%), P(-10%)

### Module 6 — Value at Risk (VaR)
- Three methods at 95% and 99%: Historical Simulation, Parametric Normal, Monte Carlo
- Expected Shortfall (CVaR) at 95%
- Loss distribution histogram with VaR threshold line and tail shading
- Kupiec POF backtesting over 252 days

### Module 7 — Credit Risk Modeling
- Logistic regression on synthetic 500-company dataset (D/E, ICR, Current Ratio, ROE, NPM)
- Real stock ratios from `yfinance.info` for inference
- Credit Score (300–850 scale) mapped to Risk Grade (AAA–D)
- Semicircular gauge chart, Confusion Matrix, AUC-ROC, 15-month PD trend

### Module 8 — Portfolio Optimization
- Efficient frontier: 5,000 random weight combinations
- Max Sharpe Ratio optimisation via `scipy.optimize.minimize`
- Colour-coded scatter plot + optimal allocation pie chart
- Interactive asset toggle to re-run optimisation

### Module 9 — Stress Testing
- 5 predefined scenarios + 1 user-defined custom scenario
- Factor betas from historical regression (market, rates, oil)
- Colour-coded horizontal bar chart (red/green)
- Dynamic risk interpretation paragraph

### Module 10 — Correlation Heatmap
- Correlation matrix via `pandas .corr()` on log returns
- Plotly heatmap: red (≥0.5), white (≈0), blue (negative)
- ⚠ annotation where |correlation| > 0.70
- Most diversifying pair + most redundant pair (dynamic)

---

## 🛠️ Technology Stack

| Category | Libraries |
|----------|-----------|
| Data & APIs | pandas, numpy, yfinance, scipy |
| Statistical Modeling | statsmodels, arch (GARCH), pmdarima, scikit-learn |
| Visualization | plotly, plotly.express, matplotlib, seaborn |
| Dashboard | streamlit, streamlit-option-menu |
| Portfolio Optimization | scipy.optimize |
| Deployment (Bonus) | Streamlit Cloud |

---

## 📦 Data Sources

- **Yahoo Finance** via `yfinance`: OHLCV prices, financial ratios, cash flow statements
- **NSE Tickers**: RELIANCE.NS, TCS.NS, INFY.NS, HDFCBANK.NS, WIPRO.NS
- **Gold Proxy**: GLD ETF from Yahoo Finance
- **Bond Proxy**: Simulated low-volatility asset (approx. 10Y Gsec)

---

## 🤖 AI Tools Declaration

Portions of this code were generated with AI assistance (Claude by Anthropic). All code has been reviewed, understood, and is explainable by the student. The student can explain every function and algorithm in the codebase.

---

## 👤 Team

| Name | Roll Number | Contribution |
|------|-------------|--------------|
| [Your Name] | [Your Roll Number] | All 10 modules, data pipeline, UI/UX |

---

## 🌐 Live Dashboard

*Link will be added after deployment to Streamlit Cloud.*

---

## 📝 Notes

- All calculations use daily OHLCV data from Yahoo Finance
- Risk-free rate set to 6.5% (India 10Y Government Securities)
- ARIMA may take 30–60 seconds depending on series length
- Monte Carlo with 10,000 paths takes ~2–3 seconds
