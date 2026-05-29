"""
Module 6 – Value at Risk (15 marks)
Historical, Parametric, Monte Carlo VaR + CVaR + Kupiec backtesting.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats


def kupiec_test(returns, var_95, n_days=252):
    """Kupiec Proportion of Failures (POF) test."""
    losses = returns.tail(n_days)
    exceptions = int((losses < -abs(var_95)).sum())
    alpha = 0.05
    expected = n_days * alpha
    T = n_days
    x = exceptions

    if x == 0 or x == T:
        p_value = 1.0
    else:
        pi_hat = x / T
        pi_0 = alpha
        try:
            lr = -2 * (
                x * np.log(pi_0) + (T - x) * np.log(1 - pi_0)
                - x * np.log(pi_hat) - (T - x) * np.log(1 - pi_hat)
            )
            p_value = float(1 - stats.chi2.cdf(lr, df=1))
        except Exception:
            p_value = 0.5

    verdict = "✅ Valid" if p_value > 0.05 else "❌ Invalid"
    return exceptions, expected, p_value, verdict


def render_var(df, ticker):
    st.markdown("""
    <div class='module-header'>
        <p class='module-title'>MODULE 6 — VALUE AT RISK (VaR)</p>
        <p class='module-marks'>15 marks | 3 VaR Methods · CVaR · Kupiec Backtesting</p>
    </div>
    """, unsafe_allow_html=True)

    returns = df["Daily_Return"].dropna()
    current_price = float(df["Close"].iloc[-1])
    portfolio_value = current_price * 1000  # assume 1000 shares

    # ── Method 1: Historical Simulation ───────────────────
    var_hist_95 = float(np.percentile(returns, 5))
    var_hist_99 = float(np.percentile(returns, 1))

    # ── Method 2: Parametric Normal ───────────────────────
    mu_r = float(returns.mean())
    sigma_r = float(returns.std())
    var_param_95 = float(mu_r + stats.norm.ppf(0.05) * sigma_r)
    var_param_99 = float(mu_r + stats.norm.ppf(0.01) * sigma_r)

    # ── Method 3: Monte Carlo ─────────────────────────────
    np.random.seed(42)
    mc_returns = np.random.normal(mu_r, sigma_r, 10000)
    var_mc_95 = float(np.percentile(mc_returns, 5))
    var_mc_99 = float(np.percentile(mc_returns, 1))

    # ── CVaR (ES) at 95% ──────────────────────────────────
    cvar_95 = float(returns[returns < var_hist_95].mean())

    # ── Kupiec Test ────────────────────────────────────────
    exceptions, expected_exc, p_value, verdict = kupiec_test(returns, var_hist_95)

    # ── VaR Comparison Table ───────────────────────────────
    st.markdown("#### 📋 VaR Comparison Table (1-Day Holding Period)")

    var_data = {
        "Method": ["Historical Simulation", "Parametric Normal", "Monte Carlo (10,000 paths)"],
        "VaR 95% (%)": [f"{var_hist_95*100:.4f}%", f"{var_param_95*100:.4f}%", f"{var_mc_95*100:.4f}%"],
        "VaR 99% (%)": [f"{var_hist_99*100:.4f}%", f"{var_param_99*100:.4f}%", f"{var_mc_99*100:.4f}%"],
        "VaR 95% (₹)": [f"₹{var_hist_95*portfolio_value:,.0f}", f"₹{var_param_95*portfolio_value:,.0f}", f"₹{var_mc_95*portfolio_value:,.0f}"],
    }

    tbl_fig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Method</b>", "<b>VaR 95% (%)</b>", "<b>VaR 99% (%)</b>", "<b>VaR 95% (₹1000 shares)</b>"],
            fill_color="#16213e", font=dict(color="#64ffda", size=12),
            align="center", height=35
        ),
        cells=dict(
            values=[var_data[k] for k in var_data],
            fill_color=[["#0e1117", "#1a1f2e", "#0e1117"]],
            font=dict(color="#ccd6f6", size=12),
            align="center", height=32
        )
    )])
    tbl_fig.update_layout(paper_bgcolor="#0e1117", margin=dict(l=0, r=0, t=0, b=0), height=160)
    st.plotly_chart(tbl_fig, width="stretch")

    # CVaR info
    st.markdown(f"""
    <div class='insight-box'>
        <b style='color:#ff6b6b;'>Expected Shortfall (CVaR) at 95%: {cvar_95*100:.4f}%</b> (₹{cvar_95*portfolio_value:,.0f} for 1000 shares)<br>
        CVaR exceeds VaR in magnitude because it represents the <i>average</i> loss beyond the VaR threshold — 
        capturing the tail risk that VaR ignores. While VaR gives the worst-case loss at a given confidence level, 
        CVaR averages all losses worse than that level, providing a more conservative risk estimate.
    </div>
    """, unsafe_allow_html=True)

    # ── Loss Distribution Chart (6c) ──────────────────────
    st.markdown("#### 📊 Loss Distribution (Monte Carlo 1-Day P&L)")
    col_chart, col_kupiec = st.columns([2, 1])

    with col_chart:
        loss_fig = go.Figure()
        
        # Histogram
        loss_fig.add_trace(go.Histogram(
            x=mc_returns * 100, nbinsx=80,
            marker_color="#4fc3f7", opacity=0.7,
            name="P&L Distribution"
        ))

        # Shade tail
        tail_x = [v for v in mc_returns * 100 if v <= var_mc_95 * 100]
        if tail_x:
            tail_counts, tail_edges = np.histogram(tail_x, bins=20)
            for i in range(len(tail_counts)):
                if tail_counts[i] > 0:
                    loss_fig.add_trace(go.Bar(
                        x=[(tail_edges[i] + tail_edges[i+1]) / 2],
                        y=[tail_counts[i]],
                        width=tail_edges[i+1] - tail_edges[i],
                        marker_color="#ff6b6b", opacity=0.85,
                        showlegend=False
                    ))

        # VaR line
        loss_fig.add_vline(
            x=var_mc_95 * 100,
            line_dash="dash", line_color="#ff6b6b", line_width=2,
            annotation_text=f"VaR 95%: {var_mc_95*100:.2f}%",
            annotation_font_color="#ff6b6b"
        )

        loss_fig.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="#ccd6f6"), height=350,
            margin=dict(l=40, r=20, t=30, b=40),
            xaxis=dict(gridcolor="#2d3561", title="1-Day Return (%)"),
            yaxis=dict(gridcolor="#2d3561", title="Frequency"),
            showlegend=False,
            barmode="overlay",
            title=dict(text="Monte Carlo Loss Distribution (10,000 paths)", font=dict(color="#64ffda", size=13))
        )
        st.plotly_chart(loss_fig, width="stretch")

    # ── Kupiec Test (6d) ──────────────────────────────────
    with col_kupiec:
        st.markdown("#### 🧪 Kupiec POF Backtesting")
        
        verdict_color = "#64ffda" if "Valid" in verdict else "#ff6b6b"
        
        kupiec_rows = [
            ["Test Period", "252 trading days"],
            ["Exceptions", str(exceptions)],
            ["Expected Exceptions", f"{expected_exc:.1f}"],
            ["Significance Level", "5%"],
            ["p-value", f"{p_value:.4f}"],
            ["Model Verdict", verdict],
        ]

        k_fig = go.Figure(data=[go.Table(
            header=dict(
                values=["<b>Metric</b>", "<b>Value</b>"],
                fill_color="#16213e", font=dict(color="#64ffda", size=12),
                align="left", height=32
            ),
            cells=dict(
                values=[[r[0] for r in kupiec_rows], [r[1] for r in kupiec_rows]],
                fill_color=[
                    ["#0e1117" if i % 2 == 0 else "#1a1f2e" for i in range(len(kupiec_rows))]
                ],
                font=dict(
                    color=["#ccd6f6"] * (len(kupiec_rows) - 1) + [verdict_color],
                    size=12
                ),
                align="left", height=28
            )
        )])
        k_fig.update_layout(
            paper_bgcolor="#0e1117", margin=dict(l=0, r=0, t=0, b=0), height=260
        )
        st.plotly_chart(k_fig, width="stretch")

        st.markdown(f"""
        <div class='insight-box' style='border-color:{verdict_color};'>
            <b>Result: {verdict}</b><br>
            p-value = {p_value:.4f} {'> 0.05 → Model accurately captures risk' if p_value > 0.05 else '< 0.05 → Model underestimates risk'}
        </div>
        """, unsafe_allow_html=True)

    # ── VaR Metrics Row ────────────────────────────────────
    st.markdown("#### 📐 Key VaR Metrics Summary")
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val in zip(
        [m1, m2, m3, m4],
        ["VaR 95% (Historical)", "VaR 99% (Historical)", "CVaR 95%", "Model Verdict"],
        [f"{var_hist_95*100:.3f}%", f"{var_hist_99*100:.3f}%", f"{cvar_95*100:.3f}%", verdict]
    ):
        with col:
            color = "negative" if "❌" not in val else "positive"
            st.markdown(f"""
            <div class='metric-row'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value negative'>{val}</div>
            </div>""", unsafe_allow_html=True)
