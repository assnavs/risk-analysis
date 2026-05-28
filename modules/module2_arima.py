"""
Module 2 – ARIMA Forecasting (15 marks)
auto_arima model selection, confidence interval chart, walk-forward validation.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error


def render_arima(df, ticker):
    st.markdown("""
    <div class='module-header'>
        <p class='module-title'>MODULE 2 — ARIMA FORECASTING</p>
        <p class='module-marks'>15 marks | Auto-ARIMA · 90-Day Forecast · Walk-Forward Validation</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Fitting ARIMA model (auto_arima)..."):
        try:
            import pmdarima as pm
            from statsmodels.tsa.arima.model import ARIMA as StatsARIMA

            close = df["Close"].dropna()
            series = close.values

            # auto_arima
            model = pm.auto_arima(
                series,
                start_p=1, start_q=1, max_p=4, max_q=4,
                stepwise=True, seasonal=False,
                information_criterion="aic",
                error_action="ignore", suppress_warnings=True
            )

            p, d, q = model.order
            aic = model.aic()
            bic = model.bic()

            # 90-day forecast
            n_forecast = 90
            forecast, conf_int = model.predict(n_periods=n_forecast, return_conf_int=True)

            last_date = close.index[-1]
            future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_forecast)

            forecast_series = pd.Series(forecast, index=future_dates)
            lower = pd.Series(conf_int[:, 0], index=future_dates)
            upper = pd.Series(conf_int[:, 1], index=future_dates)

            # RMSE on training
            fitted = model.predict_in_sample()
            rmse = float(np.sqrt(mean_squared_error(series[d:], fitted[d:])))
            mae = float(mean_absolute_error(series[d:], fitted[d:]))
            mape = float(np.mean(np.abs((series[d:] - fitted[d:]) / (series[d:] + 1e-9))) * 100)

            direction = "📈 Uptrend" if forecast[-1] > series[-1] else "📉 Downtrend"

            # Walk-forward validation (80/20)
            n = len(series)
            split = int(n * 0.8)
            train, test = series[:split], series[split:]
            
            wf_preds = []
            history = list(train)
            for obs in test:
                try:
                    wf_model = pm.ARIMA(order=(p, d, q))
                    wf_model.fit(history)
                    pred = wf_model.predict(n_periods=1)[0]
                    wf_preds.append(pred)
                except Exception:
                    wf_preds.append(history[-1])
                history.append(obs)

            in_sample_rmse = rmse
            oos_rmse = float(np.sqrt(mean_squared_error(test, wf_preds)))

        except Exception as e:
            st.error(f"ARIMA fitting error: {e}")
            return

    # ── Plot ──────────────────────────────────────────────
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=close.index, y=close.values,
        mode="lines", name="Actual Price",
        line=dict(color="white", width=2)
    ))
    fig.add_trace(go.Scatter(
        x=forecast_series.index, y=forecast_series.values,
        mode="lines", name="ARIMA Forecast",
        line=dict(color="#4fc3f7", width=2, dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=list(upper.index) + list(lower.index[::-1]),
        y=list(upper.values) + list(lower.values[::-1]),
        fill="toself", fillcolor="rgba(79,195,247,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% Confidence Interval"
    ))

    fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"), height=420,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#2d3561", title="Date"),
        yaxis=dict(gridcolor="#2d3561", title="Price (INR)"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"ARIMA({p},{d},{q}) Forecast — {ticker}", font=dict(color="#64ffda", size=14))
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Model Summary Card ────────────────────────────────
    st.markdown("#### 📋 Model Summary")
    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, label, val in zip(
        [sc1, sc2, sc3, sc4],
        ["Model Order", "AIC", "BIC", "Direction"],
        [f"ARIMA({p},{d},{q})", f"{aic:.2f}", f"{bic:.2f}", direction]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-row'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
            </div>""", unsafe_allow_html=True)

    # ── Metrics Row ───────────────────────────────────────
    st.markdown("#### 📐 Model Metrics")
    m1, m2, m3, m4 = st.columns(4)
    for col, label, val in zip(
        [m1, m2, m3, m4],
        ["RMSE (₹)", "MAE (₹)", "MAPE (%)", "90-Day Direction"],
        [f"{rmse:.2f}", f"{mae:.2f}", f"{mape:.2f}%", direction]
    ):
        with col:
            st.markdown(f"""
            <div class='metric-row'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
            </div>""", unsafe_allow_html=True)

    # ── Walk-Forward Validation ────────────────────────────
    st.markdown("#### 🔄 Walk-Forward Validation (80/20 Split)")
    wf_df = pd.DataFrame({
        "Metric": ["In-Sample RMSE", "Out-of-Sample RMSE", "Validation Period"],
        "Value": [f"₹{in_sample_rmse:.2f}", f"₹{oos_rmse:.2f}", f"{len(test)} trading days"]
    })

    wfig = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>"],
            fill_color="#16213e", font=dict(color="#64ffda", size=13),
            align="left", height=35
        ),
        cells=dict(
            values=[wf_df["Metric"].tolist(), wf_df["Value"].tolist()],
            fill_color="#0e1117", font=dict(color="#ccd6f6", size=13),
            align="left", height=30
        )
    )])
    wfig.update_layout(
        paper_bgcolor="#0e1117", margin=dict(l=0, r=0, t=0, b=0), height=160
    )
    st.plotly_chart(wfig, use_container_width=True)

    # Walk-forward chart
    wf_dates = close.index[split:]
    wf_fig2 = go.Figure()
    wf_fig2.add_trace(go.Scatter(x=wf_dates, y=test, mode="lines", name="Actual", line=dict(color="white")))
    wf_fig2.add_trace(go.Scatter(x=wf_dates, y=wf_preds, mode="lines", name="Predicted",
                                  line=dict(color="#4fc3f7", dash="dash")))
    wf_fig2.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"), height=280,
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis=dict(gridcolor="#2d3561"), yaxis=dict(gridcolor="#2d3561"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title=dict(text="Walk-Forward: Actual vs Predicted (Out-of-Sample 20%)", font=dict(color="#64ffda", size=13))
    )
    st.plotly_chart(wf_fig2, use_container_width=True)
