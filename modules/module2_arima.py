"""
Module 2 – ARIMA Forecasting (15 marks)
Uses statsmodels ARIMA with grid search for model selection (no pmdarima dependency).
Fully compatible with Python 3.12+/3.14+.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from itertools import product
import warnings
warnings.filterwarnings("ignore")


def select_arima_order(series, max_p=3, max_q=3):
    """
    Grid-search ARIMA(p,1,q) orders and return best by AIC.
    Equivalent to auto_arima(stepwise=True, seasonal=False, d=1).
    """
    from statsmodels.tsa.arima.model import ARIMA

    best_aic = np.inf
    best_order = (1, 1, 1)

    for p, q in product(range(0, max_p + 1), range(0, max_q + 1)):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = ARIMA(series, order=(p, 1, q)).fit()
            if m.aic < best_aic:
                best_aic = m.aic
                best_order = (p, 1, q)
        except Exception:
            continue

    return best_order, best_aic


def render_arima(df, ticker):
    st.markdown("""
    <div class='module-header'>
        <p class='module-title'>MODULE 2 — ARIMA FORECASTING</p>
        <p class='module-marks'>15 marks | Auto-ARIMA Grid Search · 90-Day Forecast · Walk-Forward Validation</p>
    </div>
    """, unsafe_allow_html=True)

    from statsmodels.tsa.arima.model import ARIMA
    from sklearn.metrics import mean_absolute_error

    close = df["Close"].dropna()
    series = close.values.astype(float)

    with st.spinner("Selecting best ARIMA model (grid search over p,d,q)..."):
        # Use last 300 points for speed
        fit_series = series[-300:] if len(series) > 300 else series
        order, best_aic = select_arima_order(fit_series, max_p=3, max_q=3)
        p, d, q = order

        # Fit final model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted_model = ARIMA(fit_series, order=order).fit()

        best_bic = fitted_model.bic

        # 90-day forecast
        n_forecast = 90
        forecast_res = fitted_model.get_forecast(steps=n_forecast)
        forecast_mean = forecast_res.predicted_mean
        conf_int = forecast_res.conf_int(alpha=0.05)

        last_date = close.index[-1]
        future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=n_forecast)

        forecast_series = pd.Series(forecast_mean, index=future_dates)
        lower = pd.Series(conf_int[:, 0], index=future_dates)
        upper = pd.Series(conf_int[:, 1], index=future_dates)

        # In-sample metrics
        fitted_vals = fitted_model.fittedvalues
        n_skip = d
        actual_trim = fit_series[n_skip:]
        fitted_trim = fitted_vals[n_skip:len(actual_trim) + n_skip]
        min_len = min(len(actual_trim), len(fitted_trim))

        rmse = float(np.sqrt(np.mean((actual_trim[:min_len] - fitted_trim[:min_len])**2)))
        mae  = float(mean_absolute_error(actual_trim[:min_len], fitted_trim[:min_len]))
        mape = float(np.mean(np.abs((actual_trim[:min_len] - fitted_trim[:min_len]) /
                                     (actual_trim[:min_len] + 1e-9))) * 100)
        direction = "📈 Uptrend" if forecast_mean[-1] > series[-1] else "📉 Downtrend"

        # Walk-forward validation (80/20)
        split = int(len(series) * 0.8)
        train_wf, test_wf = series[:split], series[split:]
        wf_preds = []
        history  = list(train_wf)

        for obs in test_wf:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    wf_m = ARIMA(history, order=order).fit()
                pred = float(wf_m.forecast(steps=1)[0])
            except Exception:
                pred = history[-1]
            wf_preds.append(pred)
            history.append(obs)

        oos_rmse = float(np.sqrt(np.mean((test_wf - np.array(wf_preds))**2)))

    # ── Main chart ────────────────────────────────────────
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
        title=dict(text=f"ARIMA({p},{d},{q}) Forecast — {ticker}",
                   font=dict(color="#64ffda", size=14))
    )
    st.plotly_chart(fig, width="stretch")

    # ── Model Summary ─────────────────────────────────────
    st.markdown("#### 📋 Model Summary")
    sc1, sc2, sc3, sc4 = st.columns(4)
    for col, label, val in zip(
        [sc1, sc2, sc3, sc4],
        ["Model Order", "AIC", "BIC", "Direction"],
        [f"ARIMA({p},{d},{q})", f"{best_aic:.2f}", f"{best_bic:.2f}", direction]
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
    wf_dates = close.index[split:]
    wf_fig = go.Figure()
    wf_fig.add_trace(go.Scatter(x=wf_dates, y=test_wf, mode="lines",
                                 name="Actual", line=dict(color="white")))
    wf_fig.add_trace(go.Scatter(x=wf_dates, y=wf_preds, mode="lines",
                                 name="Predicted", line=dict(color="#4fc3f7", dash="dash")))
    wf_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="#ccd6f6"), height=280,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(gridcolor="#2d3561"),
        yaxis=dict(gridcolor="#2d3561"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title=dict(text=f"Walk-Forward: In-Sample RMSE ₹{rmse:.2f} | OOS RMSE ₹{oos_rmse:.2f}",
                   font=dict(color="#64ffda", size=13))
    )
    st.plotly_chart(wf_fig, width="stretch")

    wf_tbl = go.Figure(data=[go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Value</b>"],
            fill_color="#16213e", font=dict(color="#64ffda", size=13),
            align="left", height=35
        ),
        cells=dict(
            values=[
                ["In-Sample RMSE", "Out-of-Sample RMSE", "Validation Period", "Train / Test Split"],
                [f"₹{rmse:.2f}", f"₹{oos_rmse:.2f}", f"{len(test_wf)} trading days", "80% / 20%"]
            ],
            fill_color="#0e1117", font=dict(color="#ccd6f6", size=13),
            align="left", height=30
        )
    )])
    wf_tbl.update_layout(paper_bgcolor="#0e1117",
                          margin=dict(l=0,r=0,t=0,b=0), height=180)
    st.plotly_chart(wf_tbl, width="stretch")
