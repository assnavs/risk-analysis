"""
Module 10 – Correlation Heatmap (8 marks)
Correlation matrix, colour-coded heatmap, annotations, diversification insights.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def render_heatmap(all_returns_df, selected_ticker):
    st.markdown("""
    <div class='module-header'>
        <p class='module-title'>MODULE 10 — CORRELATION HEATMAP</p>
        <p class='module-marks'>8 marks | Correlation Matrix · Annotated Heatmap · Diversification Insights</p>
    </div>
    """, unsafe_allow_html=True)

    returns_df = all_returns_df.dropna()

    if len(returns_df) < 10:
        st.error("Insufficient data for correlation analysis.")
        return

    # Compute correlation matrix (10a)
    corr_matrix = returns_df.corr()
    assets = list(corr_matrix.columns)
    n = len(assets)

    # ── Heatmap (10b, 10c) ────────────────────────────────
    st.markdown("#### 🔥 Asset Correlation Heatmap")

    z = corr_matrix.values
    text = [[f"{z[i][j]:.2f}" for j in range(n)] for i in range(n)]

    # Custom colorscale: blue (neg) → white (zero) → red (pos)
    colorscale = [
        [0.0, "#1a3a6b"],
        [0.25, "#4a6fa5"],
        [0.5, "#f5f5f5"],
        [0.75, "#e05252"],
        [1.0, "#8b0000"],
    ]

    fig = go.Figure(go.Heatmap(
        z=z,
        x=assets,
        y=assets,
        colorscale=colorscale,
        zmin=-1, zmax=1,
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=13, color="black"),
        colorbar=dict(
            title=dict(text="Correlation", font=dict(color="#ccd6f6")),
            tickfont=dict(color="#ccd6f6"),
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0<br>(Inverse)", "-0.5", "0<br>(None)", "0.5", "+1.0<br>(Perfect)"]
        )
    ))

    # Annotate high correlations with warning (10c)
    annotations = []
    for i in range(n):
        for j in range(n):
            if i != j and abs(z[i][j]) > 0.70:
                annotations.append(dict(
                    x=assets[j], y=assets[i],
                    text="⚠",
                    showarrow=False,
                    font=dict(size=14, color="#ffd93d"),
                    xshift=18, yshift=12
                ))

    fig.update_layout(
        paper_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"),
        height=500,
        margin=dict(l=20, r=20, t=50, b=20),
        annotations=annotations,
        title=dict(text="Asset Correlation Matrix (Daily Log Returns)", font=dict(color="#64ffda", size=14)),
        xaxis=dict(side="bottom"),
    )
    st.plotly_chart(fig, width="stretch")

    # Warning legend
    st.caption("⚠ = |correlation| > 0.70 — high co-movement reduces diversification benefit")

    # ── Diversification Insights (10d) ────────────────────
    st.markdown("#### 💡 Dynamic Diversification Insights")

    # Find most diversifying pair (lowest correlation)
    min_corr = 1.0
    min_pair = ("", "")
    max_corr = -1.0
    max_pair = ("", "")

    for i in range(n):
        for j in range(i + 1, n):
            c = z[i][j]
            if c < min_corr:
                min_corr = c
                min_pair = (assets[i], assets[j])
            if c > max_corr:
                max_corr = c
                max_pair = (assets[i], assets[j])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class='insight-box' style='border-left: 4px solid #64ffda;'>
            <b style='color:#64ffda;'>🌿 Most Diversifying Pair</b><br>
            <b>{min_pair[0]}</b> ↔ <b>{min_pair[1]}</b><br>
            Correlation: <b style='color:#64ffda;'>{min_corr:.4f}</b><br>
            These assets move independently — combining them reduces portfolio variance most effectively.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        redundant_color = "#ff6b6b" if max_corr > 0.7 else "#ffd93d"
        st.markdown(f"""
        <div class='insight-box' style='border-left: 4px solid {redundant_color};'>
            <b style='color:{redundant_color};'>🔗 Most Redundant Pair</b><br>
            <b>{max_pair[0]}</b> ↔ <b>{max_pair[1]}</b><br>
            Correlation: <b style='color:{redundant_color};'>{max_corr:.4f}</b><br>
            {'High co-movement detected ⚠ — holding both may not improve diversification.' if max_corr > 0.7 else 'Moderate co-movement — some redundancy exists.'}
        </div>
        """, unsafe_allow_html=True)

    # ── Correlation Distribution ──────────────────────────
    st.markdown("#### 📊 Pairwise Correlation Distribution")
    flat_corrs = []
    pair_labels = []
    for i in range(n):
        for j in range(i + 1, n):
            flat_corrs.append(z[i][j])
            pair_labels.append(f"{assets[i]} / {assets[j]}")

    hist_fig = go.Figure()
    hist_fig.add_trace(go.Histogram(
        x=flat_corrs, nbinsx=20,
        marker_color="#4fc3f7", opacity=0.75,
        name="Pairwise Correlations"
    ))
    hist_fig.add_vline(x=0.7, line_dash="dash", line_color="#ff6b6b", line_width=1.5,
                        annotation_text="0.70 threshold", annotation_font_color="#ff6b6b")
    hist_fig.add_vline(x=-0.7, line_dash="dash", line_color="#64ffda", line_width=1.5,
                        annotation_text="-0.70 threshold", annotation_font_color="#64ffda")
    hist_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"), height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#2d3561", title="Correlation Coefficient", range=[-1, 1]),
        yaxis=dict(gridcolor="#2d3561", title="Count"),
        showlegend=False
    )
    st.plotly_chart(hist_fig, width="stretch")

    # Full Correlation Table
    st.markdown("#### 📋 Full Correlation Table")
    styled_df = corr_matrix.round(4)
    st.dataframe(
        styled_df.style.background_gradient(cmap="RdBu_r", vmin=-1, vmax=1)
                        .format("{:.4f}"),
        width="stretch",
        height=min(400, n * 40 + 60)
    )
