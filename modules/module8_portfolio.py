"""
Module 8 – Portfolio Optimization (12 marks)
Efficient frontier, max-Sharpe portfolio, optimal allocation, interactive toggles.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize


RISK_FREE_RATE = 0.065  # India 10Y Gsec


def compute_portfolio_metrics(weights, mean_returns, cov_matrix):
    ret = float(np.dot(weights, mean_returns) * 252)
    vol = float(np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights))))
    sharpe = (ret - RISK_FREE_RATE) / vol if vol > 0 else 0
    return ret, vol, sharpe


def efficient_frontier(mean_returns, cov_matrix, n_portfolios=5000):
    n_assets = len(mean_returns)
    results = np.zeros((3, n_portfolios))
    weights_record = []

    for i in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        ret, vol, sharpe = compute_portfolio_metrics(w, mean_returns, cov_matrix)
        results[0, i] = vol
        results[1, i] = ret
        results[2, i] = sharpe
        weights_record.append(w)

    return results, weights_record


def max_sharpe_portfolio(mean_returns, cov_matrix):
    n = len(mean_returns)

    def neg_sharpe(w):
        ret, vol, sharpe = compute_portfolio_metrics(w, mean_returns, cov_matrix)
        return -sharpe

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    bounds = [(0, 1)] * n
    x0 = np.ones(n) / n

    result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
    return result.x


def render_portfolio(all_returns_df, selected_ticker):
    st.markdown("""
    <div class='module-header'>
        <p class='module-title'>MODULE 8 — PORTFOLIO OPTIMIZATION</p>
        <p class='module-marks'>12 marks | Efficient Frontier · Max-Sharpe Portfolio · Asset Toggle</p>
    </div>
    """, unsafe_allow_html=True)

    all_assets = list(all_returns_df.columns)

    # ── Asset Toggle (8d) ─────────────────────────────────
    st.markdown("#### 🔧 Asset Selection (Toggle to Re-Run Optimization)")
    selected_assets = st.multiselect(
        "Select assets for portfolio optimization:",
        all_assets,
        default=all_assets,
        key="portfolio_assets"
    )

    if len(selected_assets) < 2:
        st.warning("Please select at least 2 assets for portfolio optimization.")
        return

    returns_df = all_returns_df[selected_assets].dropna()

    if len(returns_df) < 30:
        st.warning("Insufficient data for selected assets.")
        return

    with st.spinner("Computing efficient frontier with 5,000 random portfolios..."):
        mean_returns = returns_df.mean().values
        cov_matrix = returns_df.cov().values

        # Efficient frontier
        results, weights_record = efficient_frontier(mean_returns, cov_matrix, 5000)

        # Max Sharpe portfolio
        opt_weights = max_sharpe_portfolio(mean_returns, cov_matrix)
        opt_ret, opt_vol, opt_sharpe = compute_portfolio_metrics(opt_weights, mean_returns, cov_matrix)

    # ── Efficient Frontier Plot (8b) ───────────────────────
    st.markdown("#### 📊 Efficient Frontier")

    sharpe_vals = results[2, :]
    vols = results[0, :] * 100
    rets = results[1, :] * 100

    ef_fig = go.Figure()
    ef_fig.add_trace(go.Scatter(
        x=vols, y=rets,
        mode="markers",
        marker=dict(
            color=sharpe_vals,
            colorscale="Viridis",
            size=4, opacity=0.6,
            colorbar=dict(title=dict(text="Sharpe Ratio", font=dict(color="#ccd6f6")), tickfont=dict(color="#ccd6f6"))
        ),
        name="Random Portfolios",
        text=[f"Sharpe: {s:.2f}" for s in sharpe_vals],
        hovertemplate="Vol: %{x:.2f}%<br>Return: %{y:.2f}%<br>%{text}"
    ))

    # Mark max Sharpe
    ef_fig.add_trace(go.Scatter(
        x=[opt_vol * 100], y=[opt_ret * 100],
        mode="markers+text",
        marker=dict(color="#ff6b6b", size=18, symbol="star", line=dict(color="white", width=2)),
        text=[f"★ Max Sharpe<br>{opt_sharpe:.2f}"],
        textposition="top right",
        textfont=dict(color="white", size=11),
        name=f"Max Sharpe Portfolio ({opt_sharpe:.2f})"
    ))

    ef_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"), height=450,
        margin=dict(l=50, r=20, t=40, b=50),
        xaxis=dict(gridcolor="#2d3561", title="Portfolio Volatility (%)"),
        yaxis=dict(gridcolor="#2d3561", title="Expected Return (%)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"Efficient Frontier — {len(selected_assets)} Assets, 5,000 Portfolios",
                   font=dict(color="#64ffda", size=14))
    )
    st.plotly_chart(ef_fig, width="stretch")

    # ── Optimal Allocation (8c) ────────────────────────────
    st.markdown("#### 🥧 Optimal Portfolio Allocation")
    pie_col, metrics_col = st.columns([1, 1])

    with pie_col:
        pie_fig = go.Figure(go.Pie(
            labels=selected_assets,
            values=opt_weights * 100,
            hole=0.35,
            marker=dict(colors=px_colors(len(selected_assets))),
            textinfo="label+percent",
            textfont=dict(color="white", size=11)
        ))
        pie_fig.update_layout(
            paper_bgcolor="#0e1117", font=dict(color="#ccd6f6"),
            height=350, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(pie_fig, width="stretch")

    with metrics_col:
        st.markdown(f"""
        <div class='metric-row' style='margin-bottom:10px;'>
            <div class='metric-label'>Expected Return (Optimized)</div>
            <div class='metric-value positive'>{opt_ret*100:.2f}%</div>
        </div>
        <div class='metric-row' style='margin-bottom:10px;'>
            <div class='metric-label'>Portfolio Volatility (Optimized)</div>
            <div class='metric-value neutral'>{opt_vol*100:.2f}%</div>
        </div>
        <div class='metric-row' style='margin-bottom:10px;'>
            <div class='metric-label'>Sharpe Ratio (Optimized)</div>
            <div class='metric-value positive'>{opt_sharpe:.4f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("**Optimal Weights:**")
        for asset, weight in zip(selected_assets, opt_weights):
            st.markdown(f"- **{asset}**: `{weight*100:.2f}%`")

    # ── Weight Bar Chart ───────────────────────────────────
    st.markdown("#### 📊 Optimal Portfolio Weights")
    w_fig = go.Figure(go.Bar(
        x=selected_assets, y=opt_weights * 100,
        marker_color=px_colors(len(selected_assets)),
        text=[f"{w*100:.1f}%" for w in opt_weights],
        textposition="outside"
    ))
    w_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"), height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#2d3561"),
        yaxis=dict(gridcolor="#2d3561", title="Weight (%)"),
        showlegend=False
    )
    st.plotly_chart(w_fig, width="stretch")


def px_colors(n):
    palette = ["#64ffda", "#4fc3f7", "#ff9800", "#ab47bc", "#ef5350",
               "#26a69a", "#ec407a", "#7e57c2"]
    return [palette[i % len(palette)] for i in range(n)]
