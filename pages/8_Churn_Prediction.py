"""Customer churn-risk analysis with supervised and rule-based modes."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from groq import APIError

from utils.churn_prediction import FEATURE_COLUMNS, RISK_LEVELS, analyze_churn
from utils.shared import (
    apply_compact_page_spacing,
    get_groq_client,
    get_groq_model,
    require_dashboard_data,
)

st.set_page_config(
    page_title="Churn Prediction | AI Business Copilot",
    page_icon=":material/person_alert:",
    layout="wide",
)
apply_compact_page_spacing()
st.title("Churn prediction")
st.caption("Prioritize customer retention using observed purchase behavior.")

if not require_dashboard_data():
    st.stop()

st.session_state.setdefault("churn_results", None)
st.session_state.setdefault("churn_analysis", None)
st.session_state.setdefault("include_churn_in_report", True)

sales = st.session_state.get("df")
if not isinstance(sales, pd.DataFrame) or sales.empty:
    st.warning("Upload a non-empty sales dataset on the Dashboard first.")
    st.stop()

if not isinstance(st.session_state.get("churn_results"), dict):
    st.session_state["churn_results"] = analyze_churn(sales)
churn = st.session_state["churn_results"]

if not churn["available"]:
    st.markdown(
        """
        <style>
        .churn-empty-hero {
            padding: 1.35rem 1.55rem;
            border: 1px solid rgba(120,169,255,.2);
            border-radius: 20px;
            background: radial-gradient(ellipse at 88% 10%, rgba(248,113,113,.12), transparent 38%),
                linear-gradient(115deg, #172941 0%, #102033 60%, #25303f 100%);
            margin: .25rem 0 1rem;
        }
        .churn-empty-hero span {
            color: #f6c667;
            font-size: .72rem;
            font-weight: 750;
            letter-spacing: .13em;
            text-transform: uppercase;
        }
        .churn-empty-hero h2 {
            color: #f5f8fc;
            font-size: clamp(1.4rem, 2.6vw, 1.95rem);
            margin: .35rem 0;
        }
        .churn-empty-hero p { color: #afbed0; margin: 0; max-width: 850px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="churn-empty-hero">
          <span>Retention intelligence · Data needed</span>
          <h2>Connect transactions to individual customers</h2>
          <p>
            Churn risk is a customer-level measure. A stable customer identifier
            is needed to combine repeat purchases, calculate time since last
            purchase, and distinguish active customers from customers who may
            need re-engagement.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        st.markdown("#### Required transaction data")
        st.markdown(
            "- **CustomerID** — stable identifier reused across purchases (also recognized: Customer_ID, Customer, ClientID, Client, Customer Name).\n"
            "- **Date** — transaction date, used to calculate days since last purchase.\n"
            "- **Revenue** — numeric transaction value.\n"
            "- **Product** and **Quantity** — remain required by the Dashboard upload."
        )
        st.caption(
            "An optional binary churn field named Churn, Churned, IsChurn, "
            "ChurnStatus, or Attrition enables supervised Random Forest training."
        )
    st.markdown("#### Example schema")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "CustomerID": "CUST-001",
                    "Date": "2026-05-03",
                    "Product": "Product A",
                    "Quantity": 2,
                    "Revenue": 120.00,
                    "Churn": 0,
                },
                {
                    "CustomerID": "CUST-001",
                    "Date": "2026-05-19",
                    "Product": "Product B",
                    "Quantity": 1,
                    "Revenue": 75.00,
                    "Churn": 0,
                },
                {
                    "CustomerID": "CUST-002",
                    "Date": "2026-02-04",
                    "Product": "Product A",
                    "Quantity": 1,
                    "Revenue": 60.00,
                    "Churn": 1,
                },
            ]
        ),
        hide_index=True,
        width="stretch",
        alt="Example customer transaction data with optional churn label",
    )
    st.markdown("#### Insights this data unlocks")
    insights = st.columns(3)
    with insights[0]:
        with st.container(border=True):
            st.markdown("**Early risk signals**")
            st.write("Find customers with increasing inactivity or lower purchase frequency.")
    with insights[1]:
        with st.container(border=True):
            st.markdown("**Retention prioritization**")
            st.write("Compare high-, medium-, and low-risk customer counts and probabilities.")
    with insights[2]:
        with st.container(border=True):
            st.markdown("**Targeted actions**")
            st.write("Use customer-level patterns to shape measurable re-engagement tests.")
    st.info(churn["message"])
    st.stop()

customers = churn["customers"]
st.success(churn["message"])
st.caption(
    f"Recency is measured against the latest transaction in this dataset "
    f"({pd.Timestamp(churn['snapshot_date']):%Y-%m-%d}). Purchase count represents "
    "transaction rows because no unique order ID is required."
)

metric_columns = st.columns(4)
metric_columns[0].metric("Customers analyzed", f"{len(customers):,}")
metric_columns[1].metric("High risk", f"{churn['high_risk_count']:,}")
metric_columns[2].metric("Medium risk", f"{churn['medium_risk_count']:,}")
metric_columns[3].metric("Low risk", f"{churn['low_risk_count']:,}")

if churn["metrics"]:
    st.markdown("#### Held-out model performance")
    scores = st.columns(4)
    for column, metric_name in zip(
        scores,
        ("Accuracy", "Precision", "Recall", "F1 Score"),
    ):
        column.metric(metric_name, f"{churn['metrics'][metric_name]:.1%}")
else:
    if churn.get("mode") == "supervised":
        st.warning(
            "A Random Forest was trained from the available churn labels, but "
            "the labeled sample is too small for reliable held-out metrics. "
            "Treat the probabilities as provisional until more labeled customer "
            "outcomes are available."
        )
    elif churn.get("churn_label_column"):
        st.warning(
            f"The `{churn['churn_label_column']}` field was found, but it does not "
            "provide two usable outcome classes. The page is showing rule-based "
            "risk indicators instead."
        )
    else:
        st.info(
            "No churn outcome label is available. Customer risk uses transparent "
            "recency and purchase-frequency rules; the displayed risk probability "
            "is a heuristic, not a calibrated prediction."
        )

st.markdown("#### Customer risk overview")
risk_columns = st.columns([1, 1.2])
with risk_columns[0]:
    risk_counts = (
        customers["Risk Level"].value_counts().reindex(RISK_LEVELS, fill_value=0)
    )
    donut = go.Figure(
        go.Pie(
            labels=list(RISK_LEVELS),
            values=[int(risk_counts[level]) for level in RISK_LEVELS],
            hole=0.62,
            sort=False,
            marker={
                "colors": ["#EF6461", "#F5B942", "#42B883"],
                "line": {"color": "#142033", "width": 3},
            },
            textinfo="label+value",
            hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        )
    )
    donut.update_layout(
        showlegend=False,
        height=330,
        margin={"t": 12, "b": 12, "l": 12, "r": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#DCE6F1"},
        annotations=[
            {
                "text": f"<b>{len(customers)}</b><br>customers",
                "x": 0.5,
                "y": 0.5,
                "showarrow": False,
                "font": {"size": 17, "color": "#F5F8FC"},
            }
        ],
    )
    st.plotly_chart(donut, width="stretch")
with risk_columns[1]:
    probability_chart = px.histogram(
        customers,
        x="Churn Probability",
        color="Risk Level",
        nbins=15,
        barmode="overlay",
        opacity=0.82,
        color_discrete_map={
            "High Risk": "#EF6461",
            "Medium Risk": "#F5B942",
            "Low Risk": "#42B883",
        },
        title="Customer churn probability",
        labels={"Churn Probability": "Estimated churn probability"},
    )
    probability_chart.update_layout(
        height=330,
        margin={"t": 48, "b": 12, "l": 8, "r": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#C6D3E2"},
        xaxis={"tickformat": ".0%", "gridcolor": "rgba(154,170,192,.12)"},
        yaxis={"title": "Customers", "gridcolor": "rgba(154,170,192,.12)"},
    )
    st.plotly_chart(probability_chart, width="stretch")

st.markdown("#### Customers to review")
high_risk = customers.loc[customers["Risk Level"].eq("High Risk")].copy()
if high_risk.empty:
    st.success("No customers are currently classified as High Risk.")
else:
    st.dataframe(
        high_risk.head(25),
        hide_index=True,
        width="stretch",
        column_config={
            "Churn Probability": st.column_config.ProgressColumn(
                "Churn Probability",
                min_value=0,
                max_value=1,
                format="percent",
            ),
            "Monetary Value": st.column_config.NumberColumn(format="%.2f"),
            "Average Order Value": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    if len(high_risk) > 25:
        st.caption("Showing the 25 highest-risk customers.")

st.markdown("#### Customer risk trends")
trend_data = customers.copy()
trend_data["Last Purchase Month"] = pd.to_datetime(
    trend_data["Last Purchase"], errors="coerce"
).dt.to_period("M").astype(str)
trend = (
    trend_data.dropna(subset=["Last Purchase Month"])
    .groupby(["Last Purchase Month", "Risk Level"], as_index=False)
    .size()
    .rename(columns={"size": "Customers"})
)
if trend.empty:
    st.info("There is not enough dated customer history to display a risk trend.")
else:
    trend_chart = px.bar(
        trend,
        x="Last Purchase Month",
        y="Customers",
        color="Risk Level",
        barmode="stack",
        category_orders={"Risk Level": list(RISK_LEVELS)},
        color_discrete_map={
            "High Risk": "#EF6461",
            "Medium Risk": "#F5B942",
            "Low Risk": "#42B883",
        },
        title="Customer risk by last-purchase month",
        labels={"Last Purchase Month": "Last purchase month"},
    )
    st.plotly_chart(trend_chart, width="stretch")

if not churn["feature_importance"].empty:
    st.markdown("#### Model drivers")
    importance_chart = px.bar(
        churn["feature_importance"].sort_values("Importance"),
        x="Importance",
        y="Feature",
        orientation="h",
        title="Random Forest feature importance",
        labels={"Importance": "Relative importance"},
        color="Importance",
        color_continuous_scale=[[0, "#326A85"], [1, "#42D6C5"]],
    )
    st.plotly_chart(importance_chart, width="stretch")

with st.expander("Customer feature definitions"):
    st.markdown(
        "- **Recency / Days Since Last Purchase:** days since the customer’s most recent purchase, measured at the dataset’s latest date.\n"
        "- **Frequency / Purchase Count:** transaction-row count; a unique order count needs an order identifier.\n"
        "- **Monetary Value:** sum of customer transaction revenue.\n"
        "- **Average Order Value:** average revenue per transaction row."
    )
    st.caption(f"Model inputs: {', '.join(FEATURE_COLUMNS)}.")

with st.expander("AI churn consultant", expanded=False):
    st.caption(
        "Generate an executive-ready interpretation and retention plan from "
        "aggregate risk evidence. Customer IDs are not sent to Groq."
    )
    st.checkbox(
        "Include AI churn analysis in the next executive PDF",
        key="include_churn_in_report",
    )
    if st.button("Analyze Churn Risk", type="primary"):
        distribution = (
            customers.groupby("Risk Level", as_index=False)
            .agg(
                Customers=("Customer", "nunique"),
                Average_Probability=("Churn Probability", "mean"),
                Average_Recency=("Days Since Last Purchase", "mean"),
                Average_Frequency=("Frequency", "mean"),
                Average_Monetary=("Monetary Value", "mean"),
            )
            .to_dict(orient="records")
        )
        prompt = f"""Write a concise board-ready churn report using only the
aggregate evidence. Use these exact Markdown sections:
## Executive Summary
## Churn Drivers
## Business Impact
## Customer Retention Risks
## Recommended Retention Strategy
## Immediate Actions

State that associations and rule-based scores do not prove customer intent.
Do not invent churn causes, margins, lifetime value, or recovery rates. Recommend
measurable retention tests and identify where a labeled outcome or more data is
needed. No customer identifiers are included.

Analysis method: {churn['mode']}
Status: {churn['message']}
Held-out metrics: {churn['metrics']}
Customer risk distribution: {distribution}
Feature importance: {churn['feature_importance'].to_dict(orient='records')}
"""
        try:
            with st.spinner("Preparing churn risk recommendations..."):
                response = get_groq_client().chat.completions.create(
                    model=get_groq_model(),
                    temperature=0.2,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a customer retention advisor. Ground all "
                                "claims in the supplied aggregate evidence and "
                                "distinguish model indicators from causal findings."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                )
            answer = response.choices[0].message.content
            if not answer or not answer.strip():
                st.error("Groq returned an empty churn analysis. Please try again.")
            else:
                st.session_state["churn_analysis"] = answer.strip()
                st.rerun()
        except APIError as error:
            st.error(f"Groq could not generate the churn analysis: {error}")

churn_analysis = st.session_state.get("churn_analysis")
if isinstance(churn_analysis, str) and churn_analysis.strip():
    with st.expander("Churn analysis report", expanded=True):
        st.markdown(churn_analysis)
