"""RFM customer segmentation using KMeans and optional Groq interpretation."""

import pandas as pd
import plotly.express as px
import streamlit as st

from utils.customer_segmentation import build_customer_segments
from utils.shared import (
    GroqCompletionError,
    apply_compact_page_spacing,
    generate_groq_text,
    require_dashboard_data,
)

st.set_page_config(
    page_title="Customer Segmentation | AI Business Copilot",
    page_icon=":material/groups:",
    layout="wide",
)
apply_compact_page_spacing()
st.title("Customer segmentation")
st.caption(
    "Group identified customers using spend, frequency, order value, and recency."
)

if not require_dashboard_data():
    st.stop()

st.session_state.setdefault("customer_segments", None)
st.session_state.setdefault("segment_ai_insights", None)
st.session_state.setdefault("include_segment_insights_in_report", True)

sales = st.session_state.get("df")
if not isinstance(sales, pd.DataFrame) or sales.empty:
    st.warning("Upload a non-empty sales dataset on the Dashboard first.")
    st.stop()

if not isinstance(st.session_state.get("customer_segments"), dict):
    st.session_state["customer_segments"] = build_customer_segments(sales)
segmentation = st.session_state["customer_segments"]

if not segmentation["available"]:
    missing_customer_id = segmentation.get("customer_column") is None
    st.markdown(
        """
        <style>
        .segment-empty-hero {
            padding: 1.4rem 1.6rem;
            border: 1px solid rgba(120,169,255,.22);
            border-radius: 20px;
            background: radial-gradient(ellipse at 88% 12%, rgba(246,198,103,.12), transparent 38%),
                linear-gradient(115deg, #172941 0%, #102033 60%, #18313a 100%);
            margin: .25rem 0 1rem;
        }
        .segment-empty-eyebrow {
            color: #f6c667;
            font-size: .72rem;
            font-weight: 750;
            letter-spacing: .13em;
            text-transform: uppercase;
        }
        .segment-empty-hero h2 {
            color: #f5f8fc;
            font-size: clamp(1.45rem, 2.6vw, 2rem);
            margin: .35rem 0 .35rem;
            letter-spacing: -.03em;
        }
        .segment-empty-hero p {
            color: #b4c2d3;
            max-width: 850px;
            margin: 0;
            line-height: 1.55;
        }
        .segment-next-step {
            padding: 1rem 1.15rem;
            border-left: 4px solid #42d6c5;
            border-radius: 12px;
            background: rgba(66,214,197,.08);
            margin: .6rem 0 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if missing_customer_id:
        st.markdown(
            """
            <div class="segment-empty-hero">
              <div class="segment-empty-eyebrow">Customer intelligence · Data needed</div>
              <h2>Give every buyer a consistent identity</h2>
              <p>
                This sales file has transaction and product data, but no customer
                identifier. Without a stable ID, transactions cannot be grouped
                into individual customer histories, so spend, purchase frequency,
                and recency cannot be calculated reliably.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.container(border=True):
            st.markdown("#### Why customer ID matters")
            st.write(
                "Segmentation compares customer-level purchasing patterns. A "
                "product name or transaction row is not a customer identity, and "
                "using either would create misleading segments."
            )
        st.markdown("#### Data to include")
        data_columns = st.columns(2)
        with data_columns[0]:
            with st.container(border=True):
                st.markdown("**Required for customer segmentation**")
                st.markdown(
                    "- **CustomerID** — stable ID reused for each purchase by the same customer.\n"
                    "- **Date** — transaction date, used to calculate recency.\n"
                    "- **Revenue** — transaction value, used to calculate spend and order value."
                )
                st.caption(
                    "CustomerID may also be named Customer, ClientID, or Client. "
                    "Use the same ID consistently across rows."
                )
        with data_columns[1]:
            with st.container(border=True):
                st.markdown("**Keep the sales upload fields**")
                st.markdown(
                    "- **Date**, **Product**, **Quantity**, and **Revenue** are still required by the Dashboard upload.\n"
                    "- **Quantity** improves purchase-volume context; it does not replace CustomerID.\n"
                    "- Optional fields such as region or channel can support future analysis."
                )

        st.markdown("#### Example customer transaction schema")
        sample_schema = pd.DataFrame(
            [
                {
                    "CustomerID": "CUST-001",
                    "Date": "2026-05-03",
                    "Product": "Product A",
                    "Quantity": 2,
                    "Revenue": 120.00,
                },
                {
                    "CustomerID": "CUST-001",
                    "Date": "2026-05-19",
                    "Product": "Product B",
                    "Quantity": 1,
                    "Revenue": 75.00,
                },
                {
                    "CustomerID": "CUST-002",
                    "Date": "2026-05-12",
                    "Product": "Product A",
                    "Quantity": 1,
                    "Revenue": 60.00,
                },
            ]
        )
        st.dataframe(
            sample_schema,
            hide_index=True,
            width="stretch",
            alt="Example sales transactions with stable customer identifiers",
        )
        st.caption(
            "Illustrative rows only. Add the CustomerID column to your source CSV "
            "and retain the existing Dashboard-required sales columns."
        )

        st.markdown("#### Insights this will unlock")
        insight_columns = st.columns(3)
        with insight_columns[0]:
            with st.container(border=True):
                st.markdown("**Customer value**")
                st.write("Compare total spend, purchase frequency, and average order value.")
        with insight_columns[1]:
            with st.container(border=True):
                st.markdown("**Engagement signals**")
                st.write("See recency patterns and a recent-purchase indicator by segment.")
        with insight_columns[2]:
            with st.container(border=True):
                st.markdown("**Growth opportunities**")
                st.write("Identify high-value groups, potential at-risk customers, and tailored marketing ideas.")

        st.markdown(
            """
            <div class="segment-next-step">
              <strong>Next step</strong><br>
              Export transaction-level sales with a persistent CustomerID, then
              upload the CSV again on the Dashboard. Segments and AI insights will
              be calculated from that customer-linked history.
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="segment-empty-hero">
              <div class="segment-empty-eyebrow">Customer intelligence · Not ready</div>
              <h2>More usable customer history is needed</h2>
              <p>
                Customer segmentation is unavailable for this upload. Review the
                data requirement below, then add valid customer-linked purchase
                history and upload the file again.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.warning(segmentation["message"])
        st.markdown("#### Required columns")
        st.markdown(
            "- **CustomerID** — a stable identifier shared by repeat purchases.\n"
            "- **Date** — a valid transaction date.\n"
            "- **Revenue** — a valid numeric transaction amount.\n"
            "- **Product** and **Quantity** — also required by the Dashboard upload."
        )
        st.markdown("#### Expected customer transaction shape")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "CustomerID": "CUST-001",
                        "Date": "2026-05-03",
                        "Product": "Product A",
                        "Quantity": 2,
                        "Revenue": 120.00,
                    },
                    {
                        "CustomerID": "CUST-001",
                        "Date": "2026-05-19",
                        "Product": "Product B",
                        "Quantity": 1,
                        "Revenue": 75.00,
                    },
                    {
                        "CustomerID": "CUST-002",
                        "Date": "2026-05-12",
                        "Product": "Product A",
                        "Quantity": 1,
                        "Revenue": 60.00,
                    },
                ]
            ),
            hide_index=True,
            width="stretch",
            alt="Example customer transaction dataset schema",
        )
        st.info(
            "With enough distinct customer profiles, this page can compare customer "
            "value, repeat purchase frequency, recency signals, revenue contribution, "
            "and segment-level marketing opportunities."
        )
    st.stop()

customers = segmentation["customers"]
summary = segmentation["summary"]
st.success(segmentation["message"])
st.caption(
    f"Customer recency is measured relative to the latest transaction date "
    f"({pd.Timestamp(segmentation['snapshot_date']):%Y-%m-%d}); a customer is "
    "marked recent when the last purchase is within 30 days."
)

metrics = st.columns(4)
metrics[0].metric("Distinct customers", f"{len(customers):,}")
metrics[1].metric("Revenue represented", f"${customers['Total Spend'].sum():,.2f}")
metrics[2].metric(
    "Average customer value",
    f"${customers['Total Spend'].mean():,.2f}",
)
metrics[3].metric("KMeans clusters", segmentation["cluster_count"])

st.subheader("Customer segmentation dashboard")
left, right = st.columns(2)
with left:
    size_chart = px.bar(
        summary.sort_values("Customers", ascending=False),
        x="Segment",
        y="Customers",
        color="Segment",
        title="Customers by segment",
        labels={"Customers": "Customer count"},
    )
    st.plotly_chart(size_chart, width="stretch")
with right:
    revenue_chart = px.bar(
        summary.sort_values("Revenue", ascending=False),
        x="Segment",
        y="Revenue",
        color="Segment",
        title="Revenue contribution by segment",
        labels={"Revenue": "Total revenue"},
    )
    st.plotly_chart(revenue_chart, width="stretch")

st.subheader("Customer value and recency")
scatter = px.scatter(
    customers,
    x="Total Spend",
    y="Recency Days",
    color="Segment",
    size="Purchase Frequency",
    hover_data={
        "Customer": True,
        "Average Order Value": ":,.2f",
        "Purchase Frequency": True,
        "Recency Status": True,
    },
    title="Customer spend vs. days since last purchase",
    labels={"Total Spend": "Total spend", "Recency Days": "Recency (days)"},
)
st.plotly_chart(scatter, width="stretch")

st.subheader("Segment comparison")
comparison = px.bar(
    summary,
    x="Segment",
    y=["Average Customer Value", "Average Order Value"],
    barmode="group",
    title="Average customer value vs. average order value",
    labels={"value": "Revenue", "variable": "Measure"},
)
st.plotly_chart(comparison, width="stretch")
retention = px.bar(
    summary,
    x="Segment",
    y="Retention Indicator",
    color="Segment",
    title="Recent-purchase indicator by segment",
    labels={"Retention Indicator": "Customers with purchase in last 30 days (%)"},
)
st.plotly_chart(retention, width="stretch")
st.dataframe(
    summary,
    hide_index=True,
    width="stretch",
    column_config={
        "Revenue": st.column_config.NumberColumn(format="$%.2f"),
        "Average Customer Value": st.column_config.NumberColumn(format="$%.2f"),
        "Average Order Value": st.column_config.NumberColumn(format="$%.2f"),
        "Revenue Contribution %": st.column_config.NumberColumn(format="%.1f%%"),
        "Retention Indicator": st.column_config.NumberColumn(format="%.1f%%"),
    },
)

st.subheader("AI segment insights")
st.checkbox(
    "Include segment insights in the next executive PDF",
    key="include_segment_insights_in_report",
)
if st.button("Generate AI segment insights", type="primary"):
    # Exclude customer identifiers from the external prompt; aggregate profiles
    # are sufficient for business analysis.
    prompt = f"""Analyze these customer segment aggregates and create a concise
business report with the exact Markdown sections below:
## Segment Summary
## Revenue Contribution
## Risks
## Growth Opportunities
## Marketing Recommendations

Ground every observation in these aggregates. Do not claim profitability,
retention, causes, or customer motivations that the features do not establish.
The retention indicator means only whether the last recorded purchase is within
30 days of the dataset's latest date; it is not a proven retention rate.

Cluster count: {segmentation['cluster_count']}
Silhouette score: {segmentation['silhouette_score']:.3f}
Segment profile table:
{summary.to_string(index=False)}
"""
    try:
        with st.spinner("Preparing segment insights..."):
            answer = generate_groq_text(
                [
                    {
                        "role": "system",
                        "content": (
                            "You are a careful customer analytics advisor. "
                            "Use only the supplied segment aggregates and "
                            "distinguish measured indicators from assumptions."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.25,
            )
        st.session_state["segment_ai_insights"] = answer
    except GroqCompletionError as error:
        st.error(f"Groq could not generate segment insights: {error}")

segment_insights = st.session_state.get("segment_ai_insights")
if isinstance(segment_insights, str) and segment_insights.strip():
    st.markdown(segment_insights)
