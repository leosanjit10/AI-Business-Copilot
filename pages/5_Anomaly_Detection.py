"""Detect and explain unusual sales patterns in the uploaded dataset."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from groq import APIError

from utils.anomaly_detection import SEVERITY_LEVELS, detect_sales_anomalies
from utils.shared import (
    apply_compact_page_spacing,
    get_groq_client,
    get_groq_model,
    require_dashboard_data,
)

st.set_page_config(
    page_title="Anomaly Detection | AI Business Copilot",
    page_icon=":material/troubleshoot:",
    layout="wide",
)
apply_compact_page_spacing()
st.title("Anomaly detection")
st.caption(
    "Find unusual revenue, order-volume, product-mix, and source-data patterns."
)

if not require_dashboard_data():
    st.stop()

st.session_state.setdefault("anomaly_results", None)
st.session_state.setdefault("anomaly_analysis", None)
st.session_state.setdefault("include_anomaly_in_report", True)

sales = st.session_state.get("df")
raw_sales = st.session_state.get("raw_df")
if not isinstance(sales, pd.DataFrame) or sales.empty:
    st.warning("Upload a non-empty sales dataset on the Dashboard first.")
    st.stop()

if not isinstance(st.session_state.get("anomaly_results"), pd.DataFrame):
    st.session_state["anomaly_results"] = detect_sales_anomalies(sales, raw_sales)
anomalies = st.session_state["anomaly_results"]

severity_counts = (
    anomalies["Severity"].value_counts().reindex(SEVERITY_LEVELS, fill_value=0)
    if not anomalies.empty
    else pd.Series(0, index=SEVERITY_LEVELS, dtype=int)
)
SEVERITY_COLORS = {
   "Low": ("#60A5FA", "rgba(59, 130, 246, .12)"),
   "Medium": ("#FBBF24", "rgba(245, 158, 11, .12)"),
   "High": ("#FB923C", "rgba(249, 115, 22, .14)"),
   "Critical": ("#F87171", "rgba(239, 68, 68, .15)"),
}

st.markdown(
   """
   <style>
   .anomaly-hero {
       padding: 1.25rem 1.5rem;
       border: 1px solid rgba(120,169,255,.2);
       border-radius: 18px;
       background: radial-gradient(ellipse at 88% 10%, rgba(248,113,113,.13), transparent 36%),
           linear-gradient(115deg, #172941 0%, #102033 60%, #26303f 100%);
       margin: .2rem 0 .8rem;
   }
   .anomaly-hero-eyebrow {
       color: #9fb2c9;
       font-size: .72rem;
       font-weight: 750;
       letter-spacing: .13em;
       text-transform: uppercase;
   }
   .anomaly-hero h2 {
       color: #f5f8fc;
       font-size: clamp(1.4rem, 2.4vw, 1.9rem);
       margin: .35rem 0 .25rem;
       letter-spacing: -.025em;
   }
   .anomaly-hero p { color: #afbed0; margin: 0; font-size: .92rem; }
   .severity-card {
       min-height: 102px;
       padding: .85rem 1rem;
       border: 1px solid var(--card-border);
       border-left: 4px solid var(--card-accent);
       border-radius: 14px;
       background: var(--card-bg);
   }
   .severity-card-label {
       color: #bdc9d8;
       font-size: .77rem;
       font-weight: 700;
       text-transform: uppercase;
       letter-spacing: .08em;
   }
   .severity-card-value {
       color: var(--card-accent);
       font-size: 1.85rem;
       font-weight: 800;
       line-height: 1.3;
   }
   .anomaly-section-note { color: #97a9bd; font-size: .86rem; margin-top: -.35rem; }
   </style>
   """,
   unsafe_allow_html=True,
)

high_priority = int(severity_counts["Critical"] + severity_counts["High"])
if anomalies.empty:
   summary_title = "No anomalies detected"
   summary_detail = (
       "The uploaded sales data shows no patterns flagged by the current checks."
   )
   summary_status = "Clear"
elif high_priority:
   summary_title = f"{high_priority:,} high-priority finding(s) need review"
   summary_detail = (
       f"{len(anomalies):,} finding(s) detected across the dataset. "
       "Start with Critical and High items, then check the source transactions."
   )
   summary_status = "Action recommended"
else:
   summary_title = f"{len(anomalies):,} finding(s) for review"
   summary_detail = (
       "No Critical or High findings. Review the highlighted dates and "
       "source-quality items as part of routine monitoring."
   )
   summary_status = "Monitor"

st.markdown(
   f"""
   <div class="anomaly-hero">
     <div class="anomaly-hero-eyebrow">Executive summary · {summary_status}</div>
     <h2>{summary_title}</h2>
     <p>{summary_detail}</p>
   </div>
   """,
   unsafe_allow_html=True,
)

st.markdown("#### Findings by severity")
severity_columns = st.columns(4)
for column, severity in zip(severity_columns, ("Critical", "High", "Medium", "Low")):
   accent, background = SEVERITY_COLORS[severity]
   with column:
       st.markdown(
           f"""
           <div class="severity-card" style="--card-accent:{accent};
               --card-bg:{background};--card-border:{accent}55">
             <div class="severity-card-label">{severity}</div>
             <div class="severity-card-value">{int(severity_counts[severity])}</div>
           </div>
           """,
           unsafe_allow_html=True,
       )

selected_severities = st.multiselect(
    "Highlight severities in the chart and table",
    options=list(SEVERITY_LEVELS),
    default=list(SEVERITY_LEVELS),
    key="anomaly_severity_filter",
)
filtered = anomalies.loc[anomalies["Severity"].isin(selected_severities)].copy()

st.markdown("#### Recommended response")
if anomalies.empty:
   with st.container(border=True):
       st.success("No immediate action required")
       st.caption(
           "Continue routine monitoring as new sales data is added. A clean "
           "result does not rule out issues that are not represented in this file."
       )
else:
   if high_priority:
       with st.container(border=True):
           st.error("Review the highest-severity findings first")
           st.caption(
               f"{int(severity_counts['Critical'])} Critical and "
               f"{int(severity_counts['High'])} High finding(s). Verify the "
               "source transactions before changing business plans."
           )
   data_quality_count = int(
       anomalies["Category"].eq("Data quality anomaly").sum()
   )
   if data_quality_count:
       with st.container(border=True):
           st.warning("Resolve source-data exceptions")
           st.caption(
               f"{data_quality_count} data-quality finding(s) may affect "
               "revenue comparisons and model results. Check duplicates and blanks."
           )
   statistical_count = len(anomalies) - data_quality_count
   if statistical_count:
       with st.container(border=True):
           st.info("Validate unusual sales activity")
           st.caption(
               f"{statistical_count} revenue, order, or product finding(s). "
               "Compare the flagged date or product with the underlying source "
               "before treating it as a business event."
           )

st.markdown("#### Severity distribution")
chart_column, detail_column = st.columns([1.15, 1], vertical_alignment="center")
with chart_column:
   donut_values = [int(severity_counts[level]) for level in SEVERITY_LEVELS]
   if sum(donut_values):
       donut = go.Figure(
           go.Pie(
               labels=list(SEVERITY_LEVELS),
               values=donut_values,
               hole=0.66,
               sort=False,
               marker={
                   "colors": [SEVERITY_COLORS[level][0] for level in SEVERITY_LEVELS],
                   "line": {"color": "#142033", "width": 3},
               },
               textinfo="label+value",
               hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
           )
       )
       donut.update_layout(
           showlegend=False,
           height=300,
           margin={"t": 8, "b": 8, "l": 8, "r": 8},
           paper_bgcolor="rgba(0,0,0,0)",
           font={"color": "#DCE6F1"},
           annotations=[
               {
                   "text": f"<b>{len(anomalies)}</b><br>findings",
                   "x": 0.5,
                   "y": 0.5,
                   "showarrow": False,
                   "font": {"size": 18, "color": "#F5F8FC"},
               }
           ],
       )
       st.plotly_chart(donut, width="stretch")
   else:
       with st.container(border=True):
           st.caption("The severity distribution will appear when findings exist.")
with detail_column:
   st.metric("Total findings", f"{len(anomalies):,}")
   if high_priority:
       st.metric("Critical / High", f"{high_priority:,}")
   st.caption(
       "Severity is assigned from the magnitude of the observed deviation; "
       "it is a review priority, not a confirmed financial impact."
   )

st.markdown("#### Revenue trend")
st.markdown(
   '<div class="anomaly-section-note">Red markers are revenue dates flagged by the selected severity filter.</div>',
   unsafe_allow_html=True,
)
daily = (
   sales.assign(
       Date=pd.to_datetime(sales["Date"], errors="coerce"),
       Revenue=pd.to_numeric(sales["Revenue"], errors="coerce"),
   )
   .dropna(subset=["Date", "Revenue"])
   .groupby("Date", as_index=False)["Revenue"]
   .sum()
   .sort_values("Date")
)
anomaly_dates = set(
   filtered.loc[
       filtered["Category"].isin(["Revenue spike", "Revenue drop"]),
       "Date",
   ].astype(str)
)
daily["Status"] = daily["Date"].dt.strftime("%Y-%m-%d").map(
   lambda date: "Anomaly" if date in anomaly_dates else "Normal"
)
if daily.empty:
   st.info("There are no usable dated revenue values to plot.")
else:
   chart = go.Figure()
   chart.add_trace(
       go.Scatter(
           x=daily["Date"],
           y=daily["Revenue"],
           mode="lines",
           name="Revenue trend",
           line={"color": "#4387D6", "width": 2.5},
           hovertemplate="%{x|%b %d, %Y}<br>Revenue: %{y:,.2f}<extra></extra>",
       )
   )
   for status, color in (("Normal", "#4387D6"), ("Anomaly", "#E5484D")):
       points = daily.loc[daily["Status"].eq(status)]
       if points.empty:
           continue
       chart.add_trace(
           go.Scatter(
               x=points["Date"],
               y=points["Revenue"],
               mode="markers",
               name=status,
               marker={
                   "color": color,
                   "size": 10 if status == "Anomaly" else 6,
                   "line": {"width": 1, "color": "#F5F8FC"},
               },
               customdata=points["Status"],
               hovertemplate=(
                   "%{x|%b %d, %Y}<br>Revenue: %{y:,.2f}"
                   "<br>Status: %{customdata}<extra></extra>"
               ),
           )
       )
   chart.update_layout(
       height=360,
       margin={"t": 12, "b": 12, "l": 8, "r": 8},
       paper_bgcolor="rgba(0,0,0,0)",
       plot_bgcolor="rgba(0,0,0,0)",
       font={"color": "#C6D3E2"},
       hovermode="x unified",
       legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
       xaxis={"title": None, "showgrid": False},
       yaxis={"title": "Revenue", "gridcolor": "rgba(154,170,192,.15)"},
   )
   st.plotly_chart(chart, width="stretch")

if daily["Date"].nunique() < 3:
   st.info(
       "At least three distinct dated sales totals are needed for statistical "
       "anomaly detection. Source-file duplicate and missing-value checks still run."
   )

with st.expander("Anomaly summary table", expanded=False):
   if filtered.empty:
       st.info("No findings match the selected severities.")
   else:
       st.dataframe(
           filtered,
           hide_index=True,
           width="stretch",
           column_config={
               "Deviation %": st.column_config.NumberColumn(format="%.1f%%"),
               "Expected Value": st.column_config.NumberColumn(format="%.2f"),
               "Actual Value": st.column_config.NumberColumn(format="%.2f"),
           },
       )

with st.expander("About this analysis"):
   st.caption(
       "Isolation Forest is used for datasets with at least eight distinct "
       "recorded dates. Shorter histories use a robust Z-score where possible; "
       "source-quality checks run independently."
   )

with st.expander(
   "AI anomaly report",
   expanded=bool(
       isinstance(st.session_state.get("anomaly_analysis"), str)
       and st.session_state["anomaly_analysis"].strip()
   ),
):
   st.caption(
       "Generate a concise explanation of observed anomalies, potential causes, "
       "business impact, risk, and recommended actions."
   )
   st.checkbox(
       "Include AI analysis in the next executive PDF",
       key="include_anomaly_in_report",
   )
   if st.button("Explain detected anomalies", type="primary"):
       evidence = (
           anomalies.head(40).to_dict(orient="records")
           if not anomalies.empty
           else []
       )
       prompt = f"""Review the sales anomaly evidence below. Do not claim a cause as
fact unless the data proves it. Explicitly distinguish observations from
hypotheses, and state when more information is required.

Return a professional Markdown report with these exact sections:
## Executive Summary
## Possible Causes
## Business Impact
## Risk Assessment
## Recommended Actions

For every possible cause, label it as a hypothesis. Tailor each recommended
action to the listed anomaly or explain that no anomaly was detected.

Anomaly count: {len(anomalies)}
Severity counts: {severity_counts.to_dict()}
Evidence (at most 40 rows): {evidence}
"""
       try:
           with st.spinner("Preparing the anomaly analysis..."):
               response = get_groq_client().chat.completions.create(
                   model=get_groq_model(),
                   temperature=0.2,
                   messages=[
                       {
                           "role": "system",
                           "content": (
                               "You are a careful business intelligence analyst. "
                               "Explain observed evidence without inventing business "
                               "causes, benchmarks, margins, or customer behavior."
                           ),
                       },
                       {"role": "user", "content": prompt},
                   ],
               )
           answer = response.choices[0].message.content
           if not answer or not answer.strip():
               st.error("Groq returned an empty explanation. Please try again.")
           else:
               st.session_state["anomaly_analysis"] = answer.strip()
               st.rerun()
       except APIError as error:
           st.error(f"Groq could not generate the anomaly explanation: {error}")

   analysis = st.session_state.get("anomaly_analysis")
   if isinstance(analysis, str) and analysis.strip():
       st.markdown(analysis)
