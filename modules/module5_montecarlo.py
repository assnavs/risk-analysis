"""
Module 5 – Monte Carlo Simulation (13 marks)
GBM simulation, path visualization, probability summary, interactive controls.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import time


def render_monte_carlo(df, ticker):
    st.markdown("""
    <div class='module-header'>
        <p class='module-title'>MODULE 5 — MONTE CARLO SIMULATION</p>
        <p class='module-marks'>13 marks | GBM Paths · Probability Summary · Interactive Controls</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Controls (5d) ─────────────────────────────────────
    st.markdown("#### ⚙️ Simulation Parameters")
    c1, c2 = st.columns(2)
    with c1:
        n_sims = st.slider("Number of Simulations", 500, 10000, 1000, 500)
    with c2:
        horizon = st.slider("Time Horizon (Trading Days)", 63, 252, 252, 21)

    run_btn = st.button("🚀 Run Simulation", type="primary")

    if "mc_results" not in st.session_state or run_btn:
        with st.spinner(f"Running {n_sims:,} Monte Carlo paths over {horizon} days..."):
            t_start = time.time()

            returns = df["Daily_Return"].dropna()
            mu = float(returns.mean())
            sigma = float(returns.std())
            S0 = float(df["Close"].iloc[-1])
            dt = 1.0

            # GBM: S(t+1) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
            np.random.seed(42)
            Z = np.random.standard_normal((horizon, n_sims))
            log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
            paths = S0 * np.exp(np.cumsum(log_returns, axis=0))
            paths = np.vstack([np.full(n_sims, S0), paths])  # prepend S0

            t_elapsed = time.time() - t_start

            st.session_state["mc_results"] = {
                "paths": paths, "S0": S0, "mu": mu, "sigma": sigma,
                "horizon": horizon, "n_sims": n_sims, "elapsed": t_elapsed
            }

    if "mc_results" not in st.session_state:
        st.info("Click 'Run Simulation' to start.")
        return

    res = st.session_state["mc_results"]
    paths = res["paths"]
    S0 = res["S0"]
    horizon = res["horizon"]
    n_sims = res["n_sims"]

    final_prices = paths[-1, :]

    # ── Probability Summary (5c) ───────────────────────────
    st.markdown("#### 📊 Simulation Summary (1 Year)")
    s1, s2, s3, s4, s5, s6 = st.columns(6)

    expected_price = float(np.mean(final_prices))
    median_price = float(np.median(final_prices))
    best_case = float(np.percentile(final_prices, 95))
    worst_case = float(np.percentile(final_prices, 5))
    prob_up_10 = float(np.mean(final_prices > S0 * 1.10) * 100)
    prob_down_10 = float(np.mean(final_prices < S0 * 0.90) * 100)

    metrics = [
        ("Expected Price", f"₹{expected_price:,.0f}", s1, "positive" if expected_price > S0 else "negative"),
        ("Median Price", f"₹{median_price:,.0f}", s2, "positive" if median_price > S0 else "negative"),
        ("Best Case (P95)", f"₹{best_case:,.0f}", s3, "positive"),
        ("Worst Case (P5)", f"₹{worst_case:,.0f}", s4, "negative"),
        ("P(Price > +10%)", f"{prob_up_10:.1f}%", s5, "positive"),
        ("P(Price < -10%)", f"{prob_down_10:.1f}%", s6, "negative"),
    ]
    for label, val, col, color in metrics:
        with col:
            st.markdown(f"""
            <div class='metric-row'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value {color}'>{val}</div>
            </div>""", unsafe_allow_html=True)

    st.caption(f"⏱ Computation time: {res['elapsed']:.3f}s | μ={res['mu']*100:.4f}% | σ={res['sigma']*100:.4f}% daily")

    # ── Path Visualization (5b) ────────────────────────────
    st.markdown("#### 📈 Simulation Paths")

    # Percentile thresholds at each step
    p95 = np.percentile(paths, 95, axis=1)
    p05 = np.percentile(paths, 5, axis=1)
    x_days = list(range(horizon + 1))

    fig = go.Figure()

    # Identify top/bottom 5% paths by final price
    sorted_idx = np.argsort(final_prices)
    n5 = max(1, int(n_sims * 0.05))
    bottom_idx = sorted_idx[:n5]
    top_idx = sorted_idx[-n5:]
    mid_idx = sorted_idx[n5:-n5]

    # Sample middle paths to avoid clutter
    sample_mid = mid_idx[::max(1, len(mid_idx) // 100)]

    for i in sample_mid[:80]:
        fig.add_trace(go.Scatter(
            x=x_days, y=paths[:, i],
            mode="lines", opacity=0.15,
            line=dict(color="#4fc3f7", width=0.5),
            showlegend=False
        ))

    for i in bottom_idx[:20]:
        fig.add_trace(go.Scatter(
            x=x_days, y=paths[:, i],
            mode="lines", opacity=0.6,
            line=dict(color="#ff6b6b", width=0.8),
            showlegend=False
        ))

    for i in top_idx[:20]:
        fig.add_trace(go.Scatter(
            x=x_days, y=paths[:, i],
            mode="lines", opacity=0.6,
            line=dict(color="#64ffda", width=0.8),
            showlegend=False
        ))

    # Mean and confidence bands
    mean_path = np.mean(paths, axis=1)
    fig.add_trace(go.Scatter(
        x=x_days, y=mean_path,
        mode="lines", name="Mean Path",
        line=dict(color="white", width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=x_days + x_days[::-1],
        y=list(p95) + list(p05[::-1]),
        fill="toself", fillcolor="rgba(100,255,218,0.07)",
        line=dict(color="rgba(0,0,0,0)"),
        name="P5-P95 Band"
    ))

    # Legend traces
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
        line=dict(color="#64ffda", width=2), name=f"Top 5% ({n5} paths)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
        line=dict(color="#ff6b6b", width=2), name=f"Bottom 5% ({n5} paths)"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
        line=dict(color="#4fc3f7", width=1), name="Remaining paths"))

    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"), height=480,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor="#2d3561", title="Trading Days"),
        yaxis=dict(gridcolor="#2d3561", title="Price (INR)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title=dict(
            text=f"Monte Carlo GBM — {n_sims:,} Paths over {horizon} Trading Days",
            font=dict(color="#64ffda", size=14)
        )
    )
    st.plotly_chart(fig, width="stretch")

    # ── Terminal Distribution ──────────────────────────────
    st.markdown("#### 📊 Terminal Price Distribution")
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=final_prices, nbinsx=60,
        marker_color="#4fc3f7", opacity=0.75,
        name="Final Price Distribution"
    ))
    hist_fig.add_vline(x=S0, line_color="white", line_dash="dash", line_width=2,
                        annotation_text=f"Current: ₹{S0:,.0f}")
    hist_fig.add_vline(x=expected_price, line_color="#64ffda", line_dash="dot", line_width=2,
                        annotation_text=f"Expected: ₹{expected_price:,.0f}")
    hist_fig.add_vline(x=worst_case, line_color="#ff6b6b", line_dash="dash", line_width=1.5,
                        annotation_text=f"P5: ₹{worst_case:,.0f}")
    hist_fig.add_vline(x=best_case, line_color="#64ffda", line_dash="dash", line_width=1.5,
                        annotation_text=f"P95: ₹{best_case:,.0f}")

    hist_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"), height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#2d3561", title="Final Price (INR)"),
        yaxis=dict(gridcolor="#2d3561", title="Frequency"),
        showlegend=False
    )
    st.plotly_chart(hist_fig, width="stretch")
