"""
Module 7 – Credit Risk Modeling (13 marks)
Logistic regression PD model, credit score gauge, confusion matrix, PD trend.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler


def generate_synthetic_dataset(n=500, seed=42):
    """Generate synthetic company dataset for PD model training."""
    np.random.seed(seed)

    de_ratio = np.random.uniform(0.1, 8.0, n)          # Debt-to-Equity
    icr = np.random.uniform(0.5, 15.0, n)              # Interest Coverage Ratio
    current_ratio = np.random.uniform(0.5, 4.0, n)     # Current Ratio
    roe = np.random.uniform(-0.2, 0.35, n)             # Return on Equity
    npm = np.random.uniform(-0.15, 0.25, n)            # Net Profit Margin

    # Label: default if risky on multiple dimensions
    default_score = (
        (de_ratio > 4.0).astype(int) +
        (icr < 2.0).astype(int) +
        (current_ratio < 1.0).astype(int) +
        (roe < 0).astype(int) +
        (npm < 0).astype(int)
    )
    default = (default_score >= 3).astype(int)
    # Add noise
    flip = np.random.binomial(1, 0.05, n)
    default = np.abs(default - flip)

    X = np.column_stack([de_ratio, icr, current_ratio, roe, npm])
    return X, default


def pd_to_credit_score(pd_pct):
    """Map PD percentage to 300-850 credit score."""
    # Lower PD → higher score
    pd_clamp = max(0.01, min(pd_pct, 100))
    score = 850 - (pd_clamp / 100) * (850 - 300)
    return int(score)


def score_to_grade(score):
    """Map credit score to risk grade."""
    if score >= 800: return "AAA"
    if score >= 750: return "AA"
    if score >= 700: return "A"
    if score >= 650: return "BBB"
    if score >= 600: return "BB"
    if score >= 550: return "B"
    if score >= 500: return "CCC"
    return "D"


def render_credit_risk(df, ticker):
    st.markdown("""
    <div class='module-header'>
        <p class='module-title'>MODULE 7 — CREDIT RISK MODELING</p>
        <p class='module-marks'>13 marks | Logistic Regression PD · Credit Score Gauge · Confusion Matrix · PD Trend</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Training credit risk model..."):
        # Train model (7a)
        X, y = generate_synthetic_dataset(500)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = LogisticRegression(random_state=42, max_iter=500, C=1.0)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, y_prob))
        acc = float(accuracy_score(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred)

        # Fetch real ratios (7b)
        try:
            from modules.data_loader import get_ticker_info; info = get_ticker_info(ticker) or {}
            de = info.get("debtToEquity", 50) or 50
            de = de / 100  # yfinance returns in %, normalize
            icr_val = info.get("ebitdaMargins", 0.15) / 0.08 if info.get("ebitdaMargins") else 3.5
            curr_r = info.get("currentRatio", 1.5) or 1.5
            roe_val = info.get("returnOnEquity", 0.10) or 0.10
            npm_val = info.get("profitMargins", 0.08) or 0.08
        except Exception:
            de, icr_val, curr_r, roe_val, npm_val = 0.8, 3.5, 1.5, 0.12, 0.09

        live_features = np.array([[de, icr_val, curr_r, roe_val, npm_val]])
        live_scaled = scaler.transform(live_features)
        pd_prob = float(model.predict_proba(live_scaled)[0][1]) * 100

        credit_score = pd_to_credit_score(pd_prob)
        grade = score_to_grade(credit_score)

    # ── Layout: Gauge + PD + Grade ────────────────────────
    st.markdown("#### 🏦 Credit Risk Assessment")
    g_col, m_col = st.columns([1, 1])

    with g_col:
        # Semicircular gauge (7c)
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credit_score,
            title={"text": "Credit Score (300–850)", "font": {"color": "#ccd6f6", "size": 14}},
            number={"font": {"color": "#64ffda", "size": 42}},
            gauge={
                "axis": {"range": [300, 850], "tickcolor": "#8892b0",
                          "tickvals": [300, 400, 500, 600, 700, 800, 850]},
                "bar": {"color": "#64ffda", "thickness": 0.25},
                "bgcolor": "#1a1f2e",
                "steps": [
                    {"range": [300, 450], "color": "#4f0d0d"},
                    {"range": [450, 550], "color": "#7a2c1a"},
                    {"range": [550, 650], "color": "#4f3d0d"},
                    {"range": [650, 750], "color": "#0d3a4f"},
                    {"range": [750, 850], "color": "#0d4f3c"},
                ],
                "threshold": {
                    "line": {"color": "white", "width": 3},
                    "thickness": 0.75, "value": credit_score
                }
            }
        ))
        gauge_fig.update_layout(
            paper_bgcolor="#0e1117", font=dict(color="#ccd6f6"),
            height=320, margin=dict(l=20, r=20, t=50, b=0)
        )
        st.plotly_chart(gauge_fig, width="stretch")

    with m_col:
        pd_color = "#64ffda" if pd_prob < 5 else "#ffd93d" if pd_prob < 15 else "#ff6b6b"
        risk_level = "LOW" if pd_prob < 5 else "MODERATE" if pd_prob < 15 else "HIGH"
        risk_color = "#64ffda" if pd_prob < 5 else "#ffd93d" if pd_prob < 15 else "#ff6b6b"

        st.markdown(f"""
        <div style='padding: 10px 0;'>
            <div class='metric-row' style='margin-bottom:10px; border-color:{pd_color};'>
                <div class='metric-label'>Probability of Default (PD)</div>
                <div class='metric-value' style='color:{pd_color}; font-size:36px;'>{pd_prob:.2f}%</div>
            </div>
            <div class='metric-row' style='margin-bottom:10px;'>
                <div class='metric-label'>Risk Grade</div>
                <div class='metric-value' style='color:#ffd93d;'>{grade}</div>
            </div>
            <div class='metric-row' style='border-color:{risk_color};'>
                <div class='metric-label'>Risk Level</div>
                <div class='metric-value' style='color:{risk_color};'>{risk_level}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='insight-box' style='margin-top:10px;'>
            <b>Financial Ratios Used:</b><br>
            D/E Ratio: {de:.2f} | ICR: {icr_val:.2f}<br>
            Current Ratio: {curr_r:.2f} | ROE: {roe_val:.2%}<br>
            Net Profit Margin: {npm_val:.2%}
        </div>
        """, unsafe_allow_html=True)

    # ── Model Performance + Confusion Matrix (7d) ─────────
    st.markdown("#### 📊 Model Performance")
    perf_col, cm_col = st.columns([1, 1])

    with perf_col:
        st.markdown(f"""
        <div class='metric-row' style='margin-bottom:8px;'>
            <div class='metric-label'>AUC-ROC Score</div>
            <div class='metric-value positive'>{auc:.4f}</div>
        </div>
        <div class='metric-row'>
            <div class='metric-label'>Classification Accuracy</div>
            <div class='metric-value positive'>{acc*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

        st.info(f"Model AUC = {auc:.4f} {'✅ (>0.70 threshold met)' if auc >= 0.70 else '❌ (Below 0.70 threshold)'}")

    with cm_col:
        # Confusion matrix heatmap
        cm_fig = go.Figure(data=go.Heatmap(
            z=cm, x=["Predicted: Non-Default", "Predicted: Default"],
            y=["Actual: Non-Default", "Actual: Default"],
            colorscale="Blues",
            text=cm, texttemplate="%{text}",
            textfont=dict(size=20, color="white")
        ))
        cm_fig.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#ccd6f6"), height=250,
            margin=dict(l=10, r=10, t=30, b=10),
            title=dict(text="Confusion Matrix", font=dict(color="#64ffda", size=13))
        )
        st.plotly_chart(cm_fig, width="stretch")

    # ── PD Trend (15-month simulation) ────────────────────
    st.markdown("#### 📈 15-Month PD Trend (Simulated)")

    months = 15
    month_labels = pd.date_range(end=pd.Timestamp.now(), periods=months, freq="MS").strftime("%b %Y").tolist()
    pd_trend = []
    base_features = live_features.flatten()

    for i in range(months):
        noise = np.random.normal(0, [0.05, 0.15, 0.05, 0.005, 0.005])
        varied = base_features + noise * (i / months)
        try:
            varied_scaled = scaler.transform([varied])
            pd_val = float(model.predict_proba(varied_scaled)[0][1]) * 100
        except Exception:
            pd_val = pd_prob
        pd_trend.append(max(0.1, min(pd_val, 50)))

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=month_labels, y=pd_trend,
        mode="lines+markers",
        line=dict(color="#ff9800", width=2),
        marker=dict(size=8, color="#ff9800"),
        fill="tozeroy", fillcolor="rgba(255,152,0,0.1)",
        name="PD (%)"
    ))
    trend_fig.add_hline(y=5, line_dash="dash", line_color="#ffd93d",
                         annotation_text="5% Threshold", annotation_font_color="#ffd93d")
    trend_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"), height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#2d3561", title="Month"),
        yaxis=dict(gridcolor="#2d3561", title="Probability of Default (%)"),
        title=dict(text="15-Month PD Trend Simulation", font=dict(color="#64ffda", size=13))
    )
    st.plotly_chart(trend_fig, width="stretch")
