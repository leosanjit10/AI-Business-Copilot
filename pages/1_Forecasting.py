"""Prophet sales forecast page."""

import pandas as pd
import plotly.express as px
import streamlit as st
from prophet import Prophet

from utils.shared import apply_compact_page_spacing, require_dashboard_data

apply_compact_page_spacing()
st.title("Sales forecasting")
st.caption("Forecast revenue for the next 30 days from the uploaded sales history.")

if not require_dashboard_data():
    st.stop()

daily_sales = st.session_state["daily_sales"].copy()
if not {"Date", "Revenue"}.issubset(daily_sales.columns):
    st.error("Dashboard sales data is missing the Date or Revenue column.")
    st.stop()

if daily_sales["Date"].nunique() < 5:
    st.session_state["future_revenue"] = None
    st.session_state["forecast_data"] = None
    st.warning("Upload at least five distinct days of sales data to create a forecast.")
    st.stop()


@st.cache_data(show_spinner=False, max_entries=5)
def build_forecast(data: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """Fit Prophet to daily revenue and return only future predictions."""
    training_data = data.rename(columns={"Date": "ds", "Revenue": "y"})
    model = Prophet(interval_width=0.8)
    model.fit(training_data)
    future_dates = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future_dates)
    predictions = forecast.loc[
        forecast["ds"] > training_data["ds"].max(),
        ["ds", "yhat", "yhat_lower", "yhat_upper"],
    ].copy()
    predictions["yhat"] = predictions["yhat"].clip(lower=0)
    return predictions


with st.spinner("Building the revenue forecast..."):
    forecast = build_forecast(daily_sales, periods=30)

if forecast.empty:
    st.session_state["future_revenue"] = None
    st.session_state["forecast_data"] = None
    st.warning("Prophet did not return any future predictions for this dataset.")
    st.stop()

# Preserve the existing meaning of future_revenue: average daily forecast revenue.
future_revenue = float(forecast["yhat"].mean())
st.session_state["future_revenue"] = future_revenue
st.session_state["forecast_data"] = forecast.copy()

recent_daily_average = float(daily_sales.tail(14)["Revenue"].mean())
expected_growth = (
    (future_revenue / recent_daily_average - 1) * 100
    if recent_daily_average > 0
    else None
)
highest_forecast_date = forecast.loc[forecast["yhat"].idxmax(), "ds"]
lowest_forecast_date = forecast.loc[forecast["yhat"].idxmin(), "ds"]

st.markdown(
    """
    <style>
    .forecast-hero {
        background: radial-gradient(ellipse at 90% 15%, rgba(66,214,197,.14), transparent 38%),
            linear-gradient(115deg, #172941 0%, #102033 58%, #12343d 100%);
        border: 1px solid rgba(120,169,255,.22); border-radius: 22px;
        padding: 1.45rem 1.8rem; margin: .15rem 0 1rem;
        box-shadow: 0 18px 46px rgba(0,0,0,.22);
    }
    .forecast-hero span {
        color: #42d6c5; font-size: .75rem; font-weight: 750;
        letter-spacing: .14em; text-transform: uppercase;
    }
    .forecast-hero h1 {
        color: #f5f8fc; font-size: clamp(1.8rem, 3vw, 2.5rem);
        margin: .35rem 0 .3rem; letter-spacing: -.035em;
    }
    .forecast-hero p { color: #aebdd0; margin: 0; }
    .forecast-section {
        margin: 1.35rem 0 .55rem; color: #f1f5fb;
        font-size: 1.3rem; font-weight: 700; letter-spacing: -.02em;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(150deg, rgba(22,36,58,.96), rgba(17,29,47,.96));
        border: 1px solid rgba(154,170,192,.17); border-radius: 15px;
        padding: .8rem .9rem; min-height: 106px;
        box-shadow: 0 10px 26px rgba(0,0,0,.13);
    }
    [data-testid="stMetricLabel"] { color: #b9c8da !important; font-weight: 600; }
    [data-testid="stMetricValue"] {
        color: #f5f8fc !important; font-size: clamp(1.25rem, 2vw, 1.85rem) !important;
        font-weight: 750 !important; letter-spacing: -.035em;
    }
    @media (max-width: 700px) {
        .forecast-hero { padding: 1.1rem; border-radius: 16px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="forecast-hero">
      <span>Planning outlook · 30-day view</span>
      <h1>Revenue Forecast</h1>
      <p>Projected daily revenue and decision-ready interpretation based on your sales history.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="forecast-section">Forecast Summary</div>', unsafe_allow_html=True)

if expected_growth is None or abs(expected_growth) < 0.05:
    revenue_outlook = "Stable"
    recommendation = (
        "Maintain current operating plans and monitor actual sales against the "
        "forecast. Reassess allocation when a sustained demand signal emerges."
    )
    outlook_tone = "info"
elif expected_growth > 0:
    revenue_outlook = "Positive"
    recommendation = (
        "Review inventory and staffing ahead of forecasted demand peaks, and "
        "schedule marketing activity around the projected growth period. "
        "Validate the plan against realized sales before making major commitments."
    )
    outlook_tone = "success"
else:
    revenue_outlook = "Cautious"
    recommendation = (
        "Plan inventory and spend conservatively, investigate the projected "
        "softer period, and monitor actual results closely before adjusting "
        "operating commitments."
    )
    outlook_tone = "warning"

summary_columns = st.columns(5)
summary_columns[0].metric(
    "📈 Expected Growth",
    f"{expected_growth:+.1f}%" if expected_growth is not None else "N/A",
    help=(
        "Average daily forecast compared with the average revenue across the "
        "latest 14 recorded sales dates."
    ),
    border=True,
)
summary_columns[1].metric(
    "📅 Highest Forecast Date",
    pd.Timestamp(highest_forecast_date).strftime("%b %d, %Y"),
    border=True,
)
summary_columns[2].metric(
    "📅 Lowest Forecast Date",
    pd.Timestamp(lowest_forecast_date).strftime("%b %d, %Y"),
    border=True,
)
summary_columns[3].metric(
    "💰 Expected Average Daily Revenue",
    f"${future_revenue:,.2f}",
    border=True,
)
summary_columns[4].metric("🎯 Revenue Outlook", revenue_outlook, border=True)
st.caption(
    "Growth compares forecast average daily revenue to the latest 14 recorded "
    "sales dates; it is a model-based planning signal, not a guarantee."
)
st.markdown("**Business Recommendation**")
if outlook_tone == "success":
    st.success(recommendation)
elif outlook_tone == "warning":
    st.warning(recommendation)
else:
    st.info(recommendation)

st.markdown('<div class="forecast-section">Daily Revenue Projection</div>', unsafe_allow_html=True)
st.plotly_chart(
    px.line(
        forecast,
        x="ds",
        y="yhat",
        markers=True,
        title="Projected daily revenue · next 30 days",
        labels={"ds": "Forecast date", "yhat": "Forecast revenue"},
    ),
    width="stretch",
)
with st.expander("View daily forecast values"):
    st.dataframe(forecast, hide_index=True, width="stretch")
