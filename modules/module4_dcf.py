"""
Module 4 – DCF Valuation (13 marks)
DCF model with Gordon Growth, waterfall chart, margin of safety.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf


def render_dcf(df, ticker):
    st.markdown("""
    <div class='module-header'>
        <p class='module-title'>MODULE 4 — DCF VALUATION</p>
        <p class='module-marks'>13 marks | Gordon Growth Model · Waterfall Chart · Margin of Safety</p>
    </div>
    """, unsafe_allow_html=True)

    # ── User Inputs ────────────────────────────────────────
    st.markdown("#### ⚙️ DCF Parameters")
    c1, c2, c3 = st.columns(3)
    with c1:
        forecast_years = st.slider("Forecast Period (Years)", 1, 10, 5)
    with c2:
        wacc = st.slider("WACC (%)", 5.0, 25.0, 10.0, 0.5) / 100
    with c3:
        terminal_g = st.slider("Terminal Growth Rate (%)", 1.0, 5.0, 3.0, 0.5) / 100

    # ── Fetch cash flow data ───────────────────────────────
    with st.spinner("Fetching financial data..."):
        try:
            tkr = yf.Ticker(ticker)
            cf = tkr.cashflow
            info = tkr.info or {}

            # Try to get operating cash flow
            if cf is not None and not cf.empty:
                if "Operating Cash Flow" in cf.index:
                    ocf_row = cf.loc["Operating Cash Flow"]
                elif "Total Cash From Operating Activities" in cf.index:
                    ocf_row = cf.loc["Total Cash From Operating Activities"]
                else:
                    ocf_row = cf.iloc[0]
                
                # Take the most recent non-null value
                valid_vals = [v for v in ocf_row.values if pd.notna(v) and v != 0]
                if valid_vals:
                    base_fcf = abs(float(valid_vals[0]))
                else:
                    base_fcf = None
            else:
                base_fcf = None

            if not base_fcf:
                # Fallback: EPS * shares
                eps = info.get("trailingEps", None)
                shares = info.get("sharesOutstanding", 1e9)
                if eps:
                    base_fcf = abs(eps) * shares * 0.7
                else:
                    base_fcf = float(df["Close"].iloc[-1]) * 1e7 * 0.05

            # Growth rate (use 5Y avg revenue growth or 8% default)
            growth_rate = info.get("revenueGrowth", 0.08) or 0.08
            growth_rate = max(0.03, min(growth_rate, 0.25))

            market_price = float(df["Close"].iloc[-1])
            shares_outstanding = info.get("sharesOutstanding", 1e9) or 1e9

        except Exception as e:
            st.warning(f"Using estimated cash flow data: {e}")
            market_price = float(df["Close"].iloc[-1])
            base_fcf = market_price * 1e7 * 0.05
            growth_rate = 0.08
            shares_outstanding = 1e9

    # ── DCF Calculation ────────────────────────────────────
    yearly_fcf = []
    yearly_pv = []

    for yr in range(1, forecast_years + 1):
        fcf_yr = base_fcf * (1 + growth_rate) ** yr
        pv = fcf_yr / (1 + wacc) ** yr
        yearly_fcf.append(fcf_yr)
        yearly_pv.append(pv)

    # Terminal value (Gordon Growth Model)
    fcf_n = yearly_fcf[-1]
    if wacc > terminal_g:
        terminal_value = fcf_n * (1 + terminal_g) / (wacc - terminal_g)
    else:
        terminal_value = fcf_n * 15  # fallback EV/FCF multiple

    pv_terminal = terminal_value / (1 + wacc) ** forecast_years
    total_pv = sum(yearly_pv)
    enterprise_value = total_pv + pv_terminal
    intrinsic_value_per_share = enterprise_value / shares_outstanding

    # Margin of Safety
    mos = (intrinsic_value_per_share - market_price) / intrinsic_value_per_share * 100 if intrinsic_value_per_share > 0 else 0

    if mos > 15:
        valuation_status = "✅ Undervalued"
        mos_color = "#64ffda"
    elif mos >= 0:
        valuation_status = "🟡 Fairly Valued"
        mos_color = "#ffd93d"
    else:
        valuation_status = "🔴 Overvalued"
        mos_color = "#ff6b6b"

    # ── Waterfall Chart ────────────────────────────────────
    st.markdown("#### 📊 DCF Waterfall Chart")

    bar_labels = [f"Year {i}" for i in range(1, forecast_years + 1)] + ["Terminal Value", "Enterprise Value"]
    bar_values = [pv / 1e9 for pv in yearly_pv] + [pv_terminal / 1e9, enterprise_value / 1e9]
    bar_colors = ["#64ffda"] * forecast_years + ["#4fc3f7", "#ab47bc"]

    wf_fig = go.Figure()
    wf_fig.add_trace(go.Bar(
        x=bar_labels, y=bar_values,
        marker_color=bar_colors,
        text=[f"₹{v:.1f}B" for v in bar_values],
        textposition="outside",
        textfont=dict(color="#ccd6f6", size=11)
    ))
    wf_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"), height=380,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#2d3561"),
        yaxis=dict(gridcolor="#2d3561", title="Present Value (₹ Billions)"),
        title=dict(text="DCF Valuation — Present Value Components", font=dict(color="#64ffda", size=14))
    )
    st.plotly_chart(wf_fig, use_container_width=True)

    # ── DCF Summary Table ──────────────────────────────────
    st.markdown("#### 📋 DCF Summary Table")
    col_table, col_mos = st.columns([2, 1])

    with col_table:
        dcf_rows = []
        for i, (fcf, pv) in enumerate(zip(yearly_fcf, yearly_pv), 1):
            dcf_rows.append([f"Year {i} FCF", f"₹{fcf/1e9:.2f}B"])
            dcf_rows.append([f"Year {i} PV", f"₹{pv/1e9:.2f}B"])
        dcf_rows += [
            ["Terminal Value (Gordon Growth)", f"₹{terminal_value/1e9:.2f}B"],
            ["PV of Terminal Value", f"₹{pv_terminal/1e9:.2f}B"],
            ["Total PV of Cash Flows", f"₹{total_pv/1e9:.2f}B"],
            ["Enterprise Value", f"₹{enterprise_value/1e9:.2f}B"],
            ["Intrinsic Value / Share", f"₹{intrinsic_value_per_share:,.2f}"],
            ["Market Price / Share", f"₹{market_price:,.2f}"],
            ["Margin of Safety", f"{mos:.2f}%"],
            ["Valuation", valuation_status],
        ]

        tbl_fig = go.Figure(data=[go.Table(
            header=dict(
                values=["<b>Particulars</b>", "<b>Value (INR)</b>"],
                fill_color="#16213e", font=dict(color="#64ffda", size=12),
                align="left", height=32
            ),
            cells=dict(
                values=[[r[0] for r in dcf_rows], [r[1] for r in dcf_rows]],
                fill_color=[["#0e1117" if i % 2 == 0 else "#1a1f2e" for i in range(len(dcf_rows))]],
                font=dict(color="#ccd6f6", size=12),
                align="left", height=28
            )
        )])
        tbl_fig.update_layout(
            paper_bgcolor="#0e1117", margin=dict(l=0, r=0, t=0, b=0),
            height=max(250, len(dcf_rows) * 30)
        )
        st.plotly_chart(tbl_fig, use_container_width=True)

    with col_mos:
        # Gauge for Margin of Safety
        gauge_fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=mos,
            delta={"reference": 15, "valueformat": ".1f"},
            title={"text": "Margin of Safety (%)", "font": {"color": "#ccd6f6", "size": 13}},
            number={"suffix": "%", "font": {"color": mos_color, "size": 28}},
            gauge={
                "axis": {"range": [-50, 50], "tickcolor": "#8892b0"},
                "bar": {"color": mos_color},
                "bgcolor": "#1a1f2e",
                "steps": [
                    {"range": [-50, 0], "color": "#4f0d0d"},
                    {"range": [0, 15], "color": "#4f3d0d"},
                    {"range": [15, 50], "color": "#0d4f3c"},
                ],
                "threshold": {"line": {"color": "white", "width": 2}, "value": 15}
            }
        ))
        gauge_fig.update_layout(
            paper_bgcolor="#0e1117", font=dict(color="#ccd6f6"),
            height=300, margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(gauge_fig, use_container_width=True)
        st.markdown(f"""
        <div class='metric-row' style='border-color:{mos_color};'>
            <div class='metric-label'>Valuation Status</div>
            <div class='metric-value' style='color:{mos_color}; font-size:14px;'>{valuation_status}</div>
        </div>""", unsafe_allow_html=True)

    # Key assumptions
    st.markdown("#### ℹ️ Key Assumptions")
    st.info(f"""
    **WACC:** {wacc*100:.1f}% | **Terminal Growth Rate:** {terminal_g*100:.1f}% | **Forecast Years:** {forecast_years}
    **Base FCF:** ₹{base_fcf/1e9:.2f}B | **FCF Growth Rate:** {growth_rate*100:.1f}%  
    *Note: Gordon Growth Model TV = FCF_n × (1 + g) / (WACC - g)*
    """)
