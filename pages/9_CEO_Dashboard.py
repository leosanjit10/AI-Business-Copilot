"""Flagship executive dashboard combining health, forecast, risk, and AI briefing."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import APIError

from utils.anomaly_detection import SEVERITY_LEVELS, detect_sales_anomalies
from utils.business_health import calculate_business_health_score
from utils.ceo_dashboard import (
    build_forecast_summary,
    build_ceo_ai_context,
    build_ceo_scorecard,
    build_executive_priorities,
)
from utils.churn_prediction import analyze_churn
from utils.customer_segmentation import build_customer_segments
from utils.shared import (
    apply_compact_page_spacing,
    get_groq_client,
    get_groq_model,
    require_dashboard_data,
)

st.set_page_config(
    page_title="CEO Dashboard | AI Business Copilot",
    page_icon=":material/space_dashboard:",
    layout="wide",
)
apply_compact_page_spacing()
st.title("CEO dashboard")
st.caption("A decision-ready overview of performance, outlook, customer risk, and priorities.")

if not require_dashboard_data():
    st.stop()

SESSION_DEFAULTS = {
    "churn_results": None,
    "churn_analysis": None,
    "customer_segments": None,
    "anomaly_results": None,
    "ceo_briefing": None,
    "ceo_scorecard": None,
}
for key, value in SESSION_DEFAULTS.items():
    st.session_state.setdefault(key, value)

sales = st.session_state.get("df")
daily_sales = st.session_state.get("daily_sales")
product_sales = st.session_state.get("product_sales")
if (
    not isinstance(sales, pd.DataFrame)
    or sales.empty
    or not isinstance(daily_sales, pd.DataFrame)
    or daily_sales.empty
    or not isinstance(product_sales, pd.DataFrame)
    or product_sales.empty
):
    st.warning("Upload a valid sales dataset on the Dashboard before opening the CEO view.")
    st.stop()

# Keep this executive page independently useful even if the user has not visited
# the analytics pages yet; all derived results are cached in the current session.
health = calculate_business_health_score(sales, st.session_state.get("forecast_data"))
forecast = build_forecast_summary(
    daily_sales,
    st.session_state.get("forecast_data"),
)
if not isinstance(st.session_state.get("churn_results"), dict):
    st.session_state["churn_results"] = analyze_churn(sales)
churn = st.session_state["churn_results"]
if not isinstance(st.session_state.get("customer_segments"), dict):
    st.session_state["customer_segments"] = build_customer_segments(sales)
segmentation = st.session_state["customer_segments"]
if not isinstance(st.session_state.get("anomaly_results"), pd.DataFrame):
    st.session_state["anomaly_results"] = detect_sales_anomalies(
        sales,
        st.session_state.get("raw_df"),
    )
anomalies = st.session_state["anomaly_results"]
scorecard = build_ceo_scorecard(health, forecast, churn, segmentation, anomalies)
risks, opportunities = build_executive_priorities(
    health,
    forecast,
    churn,
    segmentation,
    anomalies,
)
st.session_state["ceo_scorecard"] = scorecard

st.markdown("### AI executive brief")
st.caption("Boardroom-ready context based on the current dataset and measured signals.")
if st.button("Generate CEO Executive Briefing", type="primary"):
    context = build_ceo_ai_context(
        sales,
        health,
        forecast,
        churn,
        segmentation,
        anomalies,
        scorecard,
        risks,
        opportunities,
    )
    prompt = f"""Write a concise, boardroom-ready CEO briefing based only on the
evidence below. Use all the exact Markdown headings listed here:
## Current Business Health
## Revenue Outlook
## Churn Risk
## Customer Insights
## Anomaly Findings
## Strategic Risks
## Growth Opportunities
## Executive Recommendations

State unavailable evidence plainly. Distinguish modeled outlooks and heuristics
from realized results. Do not invent causes, margins, customer motivations, or
market facts. Keep recommendations specific, proportionate, and measurable.

{context}
"""
    try:
        with st.spinner("Preparing the CEO briefing..."):
            response = get_groq_client().chat.completions.create(
                model=get_groq_model(),
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an executive intelligence advisor preparing "
                            "a concise boardroom report. Be precise, calm, and "
                            "transparent about evidence limits."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
            )
        briefing = response.choices[0].message.content
        if not briefing or not briefing.strip():
            st.error("Groq returned an empty CEO briefing. Please try again.")
        else:
            st.session_state["ceo_briefing"] = briefing.strip()
            st.rerun()
    except APIError as error:
        st.error(f"Groq could not generate the CEO briefing: {error}")

briefing = st.session_state.get("ceo_briefing")
with st.container(border=True):
    if isinstance(briefing, str) and briefing.strip():
        current_health_section = briefing.split("## Current Business Health", 1)
        brief_summary = (
            current_health_section[1].split("\n## ", 1)[0].strip()
            if len(current_health_section) > 1
            else briefing.strip()
        )
        st.markdown(brief_summary or briefing.strip())
        with st.expander("Read full CEO briefing"):
            st.markdown(briefing)
    else:
        st.info(
            "Generate an AI executive brief to see a concise interpretation of "
            "business health, outlook, risks, and recommended actions."
        )

risk_columns = st.columns(2)
for column, title, items, empty_message in (
    (
        risk_columns[0],
        "Top risk",
        risks,
        "No material risks were identified from available data.",
    ),
    (
        risk_columns[1],
        "Top opportunity",
        opportunities,
        "No material opportunities were identified from available data.",
    ),
):
    with column:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.markdown(items[0] if items else empty_message)

st.markdown("### Executive scorecard")
score_summary = st.columns(2)
score_summary[0].metric(
    "Business health",
    f"{health['overall_score']}/100",
    delta=str(health["category"]),
    border=True,
)
score_summary[1].metric(
    "CEO score",
    f"{scorecard['score']}/100",
    delta=str(scorecard["category"]),
    border=True,
)

st.markdown("### Key performance indicators")
kpis = st.columns(4)
kpis[0].metric("Revenue", f"{float(st.session_state['total_revenue']):,.2f}", border=True)
kpis[1].metric("Orders / rows", f"{int(st.session_state['total_orders']):,}", border=True)
kpis[2].metric(
    "Average order value",
    f"{float(st.session_state['average_order_value']):,.2f}",
    border=True,
    help="Revenue divided by transaction rows; unique order IDs are not available.",
)
kpis[3].metric("Top product", str(st.session_state["best_product"]), border=True)

with st.expander("CEO score details and methodology"):
    score_columns = st.columns([1, 1.25], vertical_alignment="center")
    with score_columns[0]:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=scorecard["score"],
                number={"suffix": "/100"},
                title={"text": scorecard["category"]},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#42D6C5"},
                    "steps": [
                        {"range": [0, 60], "color": "#572E3A"},
                        {"range": [60, 75], "color": "#66532D"},
                        {"range": [75, 90], "color": "#24495A"},
                        {"range": [90, 100], "color": "#20564A"},
                    ],
                },
            )
        )
        gauge.update_layout(
            height=250,
            margin={"t": 40, "b": 5, "l": 15, "r": 15},
            paper_bgcolor="rgba(0,0,0,0)",
            font={"color": "#DCE6F1"},
        )
        st.plotly_chart(gauge, width="stretch")
    with score_columns[1]:
        st.markdown("**Score components**")
        for component, value in scorecard["components"].items():
            st.progress(int(value), text=f"{component}: {value:.0f}/100")
        st.caption(
            "Weighted score: business health 40%, forecast outlook 25%, customer "
            "stability 20%, and risk exposure 15%. Unavailable forecast/customer "
            "evidence uses a neutral score and is identified in the component notes."
        )
        st.caption(
            f"Customer stability basis: {scorecard['customer_basis']}. "
            f"Risk exposure basis: {scorecard['exposure_basis']}."
        )

with st.expander("Forecast analysis", expanded=False):
    st.markdown("### Forecast summary")
    if forecast["available"]:
        growth = forecast["growth_pct"]
        forecast_columns = st.columns(4)
        forecast_columns[0].metric(
            "30-day forecast revenue",
            f"{forecast['total_revenue']:,.2f}",
            help="Sum of daily forecast values for the forecast horizon.",
        )
        forecast_columns[1].metric(
            "Average daily forecast",
            f"{forecast['average_daily_revenue']:,.2f}",
        )
        forecast_columns[2].metric(
            "Forecast growth",
            f"{growth:+.1f}%" if growth is not None else "Unavailable",
            help="Forecast daily average compared with the latest recorded-date average.",
        )
        forecast_columns[3].metric(
            "Trend / outlook", f"{forecast['trend']} · {forecast['outlook']}"
        )

        forecast_plot = forecast["data"].copy()
        actual = daily_sales.loc[:, ["Date", "Revenue"]].copy()
        actual["Date"] = pd.to_datetime(actual["Date"], errors="coerce")
        actual["Revenue"] = pd.to_numeric(actual["Revenue"], errors="coerce")
        chart = go.Figure()
        chart.add_trace(
            go.Scatter(
                x=actual["Date"],
                y=actual["Revenue"],
                mode="lines",
                name="Recorded revenue",
                line={"color": "#4387D6", "width": 2.5},
            )
        )
        chart.add_trace(
            go.Scatter(
                x=forecast_plot["ds"],
                y=forecast_plot["yhat"],
                mode="lines",
                name="Forecast",
                line={"color": "#42D6C5", "width": 2.5, "dash": "dash"},
            )
        )
        if {"yhat_lower", "yhat_upper"}.issubset(
            st.session_state.get("forecast_data", pd.DataFrame()).columns
        ):
            original_forecast = st.session_state["forecast_data"].copy()
            original_forecast["ds"] = pd.to_datetime(
                original_forecast["ds"], errors="coerce"
            )
            chart.add_trace(
                go.Scatter(
                    x=original_forecast["ds"],
                    y=original_forecast["yhat_upper"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            chart.add_trace(
                go.Scatter(
                    x=original_forecast["ds"],
                    y=original_forecast["yhat_lower"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(66,214,197,.12)",
                    name="Forecast interval",
                    hoverinfo="skip",
                )
            )
        chart.update_layout(
            height=340,
            margin={"t": 15, "b": 10, "l": 8, "r": 8},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#C6D3E2"},
            hovermode="x unified",
            yaxis={"title": "Revenue", "gridcolor": "rgba(154,170,192,.12)"},
            xaxis={"title": None, "showgrid": False},
        )
        st.plotly_chart(chart, width="stretch")
    else:
        st.info(
            "Forecast summary is unavailable. Generate a forecast on the Forecasting "
            "page to add an outlook and forecast trend."
        )

with st.expander("Customer analysis", expanded=False):
    st.markdown("### Customer churn summary")
    if churn.get("available"):
        churn_columns = st.columns(3)
        churn_columns[0].metric("High Risk", f"{churn['high_risk_count']:,}")
        churn_columns[1].metric("Medium Risk", f"{churn['medium_risk_count']:,}")
        churn_columns[2].metric("Low Risk", f"{churn['low_risk_count']:,}")
        risk_distribution = (
            churn["customers"]["Risk Level"]
            .value_counts()
            .reindex(["High Risk", "Medium Risk", "Low Risk"], fill_value=0)
        )
        risk_chart = px.pie(
            names=list(risk_distribution.index),
            values=[int(value) for value in risk_distribution.values],
            hole=0.58,
            color=list(risk_distribution.index),
            color_discrete_map={
                "High Risk": "#EF6461",
                "Medium Risk": "#F5B942",
                "Low Risk": "#42B883",
            },
            title="Customer risk distribution",
        )
        risk_chart.update_layout(
            height=320, margin={"t": 50, "b": 8, "l": 8, "r": 8}
        )
        st.plotly_chart(risk_chart, width="stretch")
        st.caption(churn["message"])
    else:
        st.info(
            "Customer churn summary is unavailable without a stable customer ID. "
            "Add customer-linked transaction history to enable retention analysis."
        )

    st.markdown("### Customer segmentation")
    if segmentation.get("available"):
        segment_summary = segmentation["summary"].copy()
        segment_counts: dict[str, int] = {}
        for name in (
            "VIP Customers",
            "Loyal Customers",
            "Regular Customers",
            "At-Risk Customers",
        ):
            segment_counts[name] = (
                int(
                    segment_summary.loc[
                        segment_summary["Segment"].eq(name), "Customers"
                    ].sum()
                )
                if segment_summary["Segment"].eq(name).any()
                else 0
            )
        for column, (name, count) in zip(
            st.columns(4),
            segment_counts.items(),
        ):
            column.metric(name, f"{count:,}")
        segment_chart = px.bar(
            segment_summary,
            x="Segment",
            y="Customers",
            color="Segment",
            title="Identified customers by segment",
            labels={"Customers": "Customers"},
        )
        st.plotly_chart(segment_chart, width="stretch")
    else:
        st.info(
            "Customer segments are unavailable for this upload because it has no "
            "usable customer identifiers and profiles. No segment counts are inferred."
        )

with st.expander("Anomaly analysis", expanded=False):
    st.markdown("### Anomaly summary")
    if anomalies.empty:
        anomaly_counts = pd.Series(0, index=SEVERITY_LEVELS)
        st.success("No anomaly findings are currently recorded.")
    else:
        anomaly_counts = (
            anomalies["Severity"].value_counts().reindex(SEVERITY_LEVELS, fill_value=0)
        )
    anomaly_columns = st.columns(4)
    for column, severity in zip(
        anomaly_columns, ("Critical", "High", "Medium", "Low")
    ):
        column.metric(severity, f"{int(anomaly_counts[severity]):,}")
    if not anomalies.empty:
        severity_rank = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        ranked = anomalies.assign(
            _rank=anomalies["Severity"].map(severity_rank).fillna(4)
        ).sort_values(["_rank", "Deviation %"], ascending=[True, False])
        top_anomaly = ranked.iloc[0]
        st.markdown(
            f"**Top anomaly:** {top_anomaly['Category']} · "
            f"{top_anomaly['Severity']} · {top_anomaly['Metric']} "
            f"(observed {top_anomaly['Actual Value']}, expected "
            f"{top_anomaly['Expected Value']})."
        )
        significant_date = next(
            (
                str(row["Date"])
                for _, row in ranked.iterrows()
                if str(row["Date"]) != "Source file"
            ),
            "Source file",
        )
        st.markdown(f"**Most significant date:** {significant_date}")
        impact_text = (
            "This finding may affect revenue interpretation or operating decisions. "
            "It is a review signal, not a confirmed financial loss; verify the source transaction."
            if top_anomaly["Category"] != "Data quality anomaly"
            else "Source-data exceptions can distort comparisons. Confirm and correct source records before acting."
        )
        st.markdown(f"**Business impact:** {impact_text}")

with st.expander("All risks and opportunities"):
    priority_columns = st.columns(2)
    with priority_columns[0]:
        st.markdown("**Risks**")
        for item in risks:
            st.markdown(f"- {item}")
    with priority_columns[1]:
        st.markdown("**Opportunities**")
        for item in opportunities:
            st.markdown(f"- {item}")
