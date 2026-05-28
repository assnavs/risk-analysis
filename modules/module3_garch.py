"""
Module 3 – GARCH Volatility Modeling (12 marks)
GARCH(1,1) conditional volatility, regime detection, spike annotation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from arch import arch_model


def render_garch(df, ticker):
    st.markdown("""
    <div class='module-header'>
        <p class='module-title'>MODULE 3 — GARCH VOLATILITY MODELING</p>
        <p class='module-marks'>12 marks | GARCH(1,1) · Regime Detection · Volatility Spike Annotation</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Fitting GARCH(1,1) model..."):
        try:
            log_ret = df["Log_Return"].dropna() * 100  # scale for GARCH

            # Fit GARCH(1,1)
            garch = arch_model(log_ret, vol="Garch", p=1, q=1, dist="Normal")
            res = garch.fit(disp="off", show_warning=False)

            # Conditional volatility (annualised)
            cond_vol = res.conditional_volatility * np.sqrt(252)

            # 20-day rolling volatility
            roll_vol = df["Log_Return"].rolling(20).std() * np.sqrt(252) * 100

            # Align index
            cond_vol.index = df.index[df.index.isin(cond_vol.index)] if len(cond_vol) == len(df) else df.index[-len(cond_vol):]
            roll_vol = roll_vol.dropna()

            # Spike date
            spike_idx = cond_vol.idxmax()
            spike_val = cond_vol.max()

            # Summary stats
            current_vol = float(cond_vol.iloc[-1])
            lt_avg = float(cond_vol.mean())

            p25 = np.percentile(cond_vol, 25)
            p75 = np.percentile(cond_vol, 75)

            if current_vol > p75:
                regime = "HIGH"
                regime_color = "#ff6b6b"
            elif current_vol < p25:
                regime = "LOW"
                regime_color = "#64ffda"
            else:
                regime = "MODERATE"
                regime_color = "#ffd93d"

        except Exception as e:
            st.error(f"GARCH fitting error: {e}")
            return

    # ── Plot ──────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=cond_vol.index, y=cond_vol.values,
        mode="lines", name="Conditional Volatility (GARCH)",
        line=dict(color="#ff9800", width=2)
    ))

    common_idx = cond_vol.index.intersection(roll_vol.index)
    fig.add_trace(go.Scatter(
        x=common_idx, y=roll_vol.loc[common_idx].values,
        mode="lines", name="20-Day Rolling Volatility",
        line=dict(color="white", width=1.5, dash="dot")
    ))

    # Spike annotation
    fig.add_vline(
        x=spike_idx, line_dash="dash", line_color="#ff6b6b", line_width=1.5
    )
    fig.add_annotation(
        x=spike_idx, y=spike_val,
        text=f"⚠ Spike: {spike_val:.1f}%<br>{spike_idx.strftime('%d %b %Y')}",
        showarrow=True, arrowhead=2,
        font=dict(color="#ff6b6b", size=11),
        bgcolor="#1a1f2e", bordercolor="#ff6b6b"
    )

    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"), height=420,
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor="#2d3561", title="Date"),
        yaxis=dict(gridcolor="#2d3561", title="Annualised Volatility (%)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"GARCH(1,1) Conditional Volatility — {ticker}", font=dict(color="#64ffda", size=14))
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Summary Stats ─────────────────────────────────────
    st.markdown("#### 📋 Volatility Summary")
    c1, c2, c3, c4 = st.columns(4)

    stats = [
        ("Current Volatility", f"{current_vol:.2f}%", c1),
        ("Long-Term Average", f"{lt_avg:.2f}%", c2),
        ("Last Spike Date", spike_idx.strftime("%d %b %Y"), c4),
    ]
    for label, val, col in stats:
        with col:
            st.markdown(f"""
            <div class='metric-row'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
            </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='metric-row' style='border-color:{regime_color};'>
            <div class='metric-label'>Volatility Regime</div>
            <div class='metric-value' style='color:{regime_color};'>{regime}</div>
        </div>""", unsafe_allow_html=True)

    # ── GARCH Parameters ──────────────────────────────────
    st.markdown("#### ⚙️ GARCH(1,1) Parameters")
    params = res.params
    param_names = res.params.index.tolist()
    
    p_col1, p_col2 = st.columns(2)
    with p_col1:
        st.markdown("**Model Parameters:**")
        for name in param_names:
            st.markdown(f"- **{name}**: `{params[name]:.6f}`")
    with p_col2:
        st.info(f"""
**Model:** GARCH(1,1)  
**AIC:** {res.aic:.2f}  
**BIC:** {res.bic:.2f}  
**Log-Likelihood:** {res.loglikelihood:.2f}  
**Observations:** {len(log_ret)}
        """)

    # ── Regime Distribution ───────────────────────────────
    st.markdown("#### 📊 Volatility Distribution")
    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=cond_vol.values, nbinsx=40,
        marker_color="#ff9800", opacity=0.7,
        name="Volatility Distribution"
    ))
    hist_fig.add_vline(x=p25, line_dash="dash", line_color="#64ffda",
                        annotation_text=f"P25: {p25:.1f}%", line_width=1.5)
    hist_fig.add_vline(x=p75, line_dash="dash", line_color="#ff6b6b",
                        annotation_text=f"P75: {p75:.1f}%", line_width=1.5)
    hist_fig.add_vline(x=current_vol, line_dash="solid", line_color="white",
                        annotation_text=f"Current: {current_vol:.1f}%", line_width=2)

    hist_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"), height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#2d3561", title="Volatility (%)"),
        yaxis=dict(gridcolor="#2d3561", title="Frequency"),
        showlegend=False
    )
    st.plotly_chart(hist_fig, use_container_width=True)
