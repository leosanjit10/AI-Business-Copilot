"""AI Business Copilot dashboard and shared sales-data initializer."""

import hashlib
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
from pandas.errors import EmptyDataError, ParserError

from utils.executive_dashboard import (
    render_data_quality_report,
    render_executive_dashboard,
)
from utils.business_health import calculate_business_health_score
from utils.shared import apply_compact_page_spacing

st.set_page_config(
    page_title="AI Business Copilot",
    page_icon=":material/analytics:",
    layout="wide",
)

REQUIRED_COLUMNS = {"Date", "Product", "Quantity", "Revenue"}
SESSION_DEFAULTS = {
    "df": None,
    "raw_df": None,
    "daily_sales": None,
    "product_sales": None,
    "total_revenue": None,
    "total_orders": None,
    "best_product": None,
    "average_order_value": None,
    "future_revenue": None,
    "ai_report": None,
    "ai_pdf_bytes": None,
    "consultant_report": None,
    "consultant_report_pdf": None,
    "forecast_data": None,
    "strategy_report": None,
    "anomaly_results": None,
    "anomaly_analysis": None,
    "customer_segments": None,
    "segment_ai_insights": None,
    "churn_results": None,
    "churn_analysis": None,
    "ceo_briefing": None,
    "ceo_scorecard": None,
    "ask_data_messages": [],
    "uploaded_file_hash": None,
}

for key, value in SESSION_DEFAULTS.items():
    st.session_state.setdefault(key, value)

apply_compact_page_spacing()


def validate_sales_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate the uploaded CSV and normalize fields used by the dashboard."""
    if data.empty:
        raise ValueError("The uploaded CSV contains no sales rows.")
    if not data.columns.is_unique:
        raise ValueError("The CSV contains duplicate column names.")

    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required CSV columns: {missing}.")

    # Retain optional source columns, notably customer identifiers, so pages
    # can offer customer analytics without inventing identifiers.
    clean = data.copy()
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
    clean["Product"] = clean["Product"].astype("string").str.strip()
    clean["Quantity"] = pd.to_numeric(clean["Quantity"], errors="coerce")
    clean["Revenue"] = pd.to_numeric(clean["Revenue"], errors="coerce")

    invalid_date = clean["Date"].isna()
    invalid_product = clean["Product"].isna() | clean["Product"].eq("")
    invalid_quantity = (
        clean["Quantity"].isna()
        | ~np.isfinite(clean["Quantity"])
        | clean["Quantity"].le(0)
    )
    invalid_revenue = (
        clean["Revenue"].isna()
        | ~np.isfinite(clean["Revenue"])
        | clean["Revenue"].lt(0)
    )
    invalid_rows = (
        invalid_date | invalid_product | invalid_quantity | invalid_revenue
    )
    if invalid_rows.any():
        raise ValueError(
            f"The CSV has {int(invalid_rows.sum())} invalid row(s). "
            "Check dates, product names, positive quantities, and non-negative "
            "numeric revenue."
        )

    return clean.sort_values("Date").reset_index(drop=True)


def save_sales_data(data: pd.DataFrame, raw_data: pd.DataFrame | None = None) -> None:
    """Compute and persist all dashboard metrics for use on other pages."""
    daily_sales = (
        data.groupby("Date", as_index=False)["Revenue"]
        .sum()
        .sort_values("Date")
        .reset_index(drop=True)
    )
    product_sales = (
        data.groupby("Product", as_index=False)["Revenue"]
        .sum()
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )
    total_revenue = float(data["Revenue"].sum())
    total_orders = int(len(data))

    st.session_state.update(
        {
            "df": data,
            "raw_df": raw_data.copy() if raw_data is not None else data.copy(),
            "daily_sales": daily_sales,
            "product_sales": product_sales,
            "total_revenue": total_revenue,
            "total_orders": total_orders,
            "best_product": str(product_sales.iloc[0]["Product"]),
            "average_order_value": total_revenue / total_orders,
            # A new upload invalidates a forecast from the previous dataset.
            "future_revenue": None,
            "forecast_data": None,
            # Reports must always correspond to the currently uploaded data.
            "ai_report": None,
            "ai_pdf_bytes": None,
            "consultant_report": None,
            "consultant_report_pdf": None,
            "strategy_report": None,
            "anomaly_results": None,
            "anomaly_analysis": None,
            "customer_segments": None,
            "segment_ai_insights": None,
            "churn_results": None,
            "churn_analysis": None,
            "ceo_briefing": None,
            "ceo_scorecard": None,
            "ask_data_messages": [],
        }
    )


def _format_change(current: float, previous: float) -> str:
    """Return a readable percentage or absolute change from a previous value."""
    if previous > 0:
        return f"{(current - previous) / previous * 100:+.1f}%"
    return f"${current - previous:+,.2f}"


def build_business_insights(
    total_revenue: float,
    total_orders: int,
    average_order_value: float,
    product_sales: pd.DataFrame,
    daily_sales: pd.DataFrame,
    future_revenue: float | None,
) -> list[dict[str, str]]:
    """Calculate evidence-based business insights from dashboard aggregates."""
    insights: list[dict[str, str]] = []

    # Compare equal groups of recorded sales dates, avoiding unsupported claims
    # about dates that are absent from the uploaded data.
    daily = daily_sales.loc[:, ["Date", "Revenue"]].copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    daily["Revenue"] = pd.to_numeric(daily["Revenue"], errors="coerce")
    daily = (
        daily.dropna(subset=["Date", "Revenue"])
        .sort_values("Date")
        .drop_duplicates(subset=["Date"], keep="last")
        .reset_index(drop=True)
    )
    if len(daily) >= 4:
        period_size = min(14, len(daily) // 2)
        prior_revenue = float(daily.iloc[-2 * period_size : -period_size]["Revenue"].sum())
        recent_revenue = float(daily.iloc[-period_size:]["Revenue"].sum())
        trend_change = _format_change(recent_revenue, prior_revenue)
        trend_direction = (
            "increased"
            if recent_revenue > prior_revenue
            else "decreased"
            if recent_revenue < prior_revenue
            else "was unchanged"
        )
        insights.append(
            {
                "title": "Revenue trend insight",
                "metric": trend_change,
                "detail": (
                    f"Revenue {trend_direction} across the latest {period_size} "
                    f"recorded sales dates compared with the preceding "
                    f"{period_size} recorded dates "
                    f"(${recent_revenue:,.2f} vs. ${prior_revenue:,.2f}). "
                    "Use the direction as a monitoring signal; the comparison "
                    "does not establish why revenue changed."
                ),
                "tone": "positive" if recent_revenue >= prior_revenue else "caution",
            }
        )
    else:
        insights.append(
            {
                "title": "Revenue trend insight",
                "metric": "More history needed",
                "detail": (
                    "At least four distinct recorded sales dates are needed to "
                    "compare recent revenue periods reliably."
                ),
                "tone": "neutral",
            }
        )

    products = product_sales.loc[:, ["Product", "Revenue"]].copy()
    products["Revenue"] = pd.to_numeric(products["Revenue"], errors="coerce")
    products = products.dropna(subset=["Product", "Revenue"])
    products = products.loc[products["Revenue"] >= 0].sort_values(
        "Revenue", ascending=False
    )
    product_revenue = float(products["Revenue"].sum())

    if products.empty or product_revenue <= 0:
        insights.extend(
            [
                {
                    "title": "Top product contribution insight",
                    "metric": "Unavailable",
                    "detail": "Product revenue is unavailable or totals zero.",
                    "tone": "neutral",
                },
                {
                    "title": "Product concentration risk analysis",
                    "metric": "Unavailable",
                    "detail": "Product revenue is unavailable for concentration analysis.",
                    "tone": "neutral",
                },
                {
                    "title": "Product opportunity insight",
                    "metric": "Unavailable",
                    "detail": "Product revenue is unavailable for opportunity analysis.",
                    "tone": "neutral",
                },
            ]
        )
        top_share = 0.0
        concentration_label = "unavailable"
        opportunity_product = None
    else:
        products["Share"] = products["Revenue"] / product_revenue
        top_product = str(products.iloc[0]["Product"])
        top_product_revenue = float(products.iloc[0]["Revenue"])
        top_share = float(products.iloc[0]["Share"])
        insights.append(
            {
                "title": "Top product contribution insight",
                "metric": f"{top_share:.1%} of product revenue",
                "detail": (
                    f"{top_product} is the largest product by revenue, "
                    f"contributing ${top_product_revenue:,.2f} of "
                    f"${product_revenue:,.2f} across the listed products."
                ),
                "tone": "positive",
            }
        )

        hhi = float((products["Share"].pow(2)).sum() * 10_000)
        concentration_label = (
            "high"
            if top_share >= 0.60 or hhi >= 2_500
            else "moderate"
            if top_share >= 0.40 or hhi >= 1_500
            else "broadly distributed"
        )
        risk_detail = (
            f"The leading product accounts for {top_share:.1%} of product revenue; "
            f"the product-revenue Herfindahl index is {hhi:,.0f} "
            "(share-squared scale of 0-10,000). "
        )
        if concentration_label == "high":
            risk_detail += (
                "This indicates material reliance on a narrow product mix. "
                "Monitor the leading product and test demand across other products."
            )
        elif concentration_label == "moderate":
            risk_detail += (
                "There is some reliance on leading products. Track their share "
                "and build a measured contribution from the rest of the range."
            )
        else:
            risk_detail += (
                "Revenue is comparatively spread across products, reducing "
                "single-product dependence in this dataset."
            )
        insights.append(
            {
                "title": "Product concentration risk analysis",
                "metric": f"{concentration_label.title()} concentration",
                "detail": risk_detail,
                "tone": "caution" if concentration_label == "high" else "neutral",
            }
        )

        opportunity_candidates = products.loc[products["Share"] < (1 / len(products))]
        if len(products) > 1 and not opportunity_candidates.empty:
            opportunity_product = str(
                opportunity_candidates.sort_values("Revenue").iloc[0]["Product"]
            )
            opportunity_row = products.loc[
                products["Product"].astype(str) == opportunity_product
            ].iloc[0]
            opportunity_share = float(opportunity_row["Share"])
            opportunity_revenue = float(opportunity_row["Revenue"])
            opportunity_detail = (
                f"{opportunity_product} contributes {opportunity_share:.1%} "
                f"(${opportunity_revenue:,.2f}) of listed product revenue, "
                "below an equal-share reference of "
                f"{1 / len(products):.1%}. Consider a small, measurable "
                "merchandising or promotion test before expanding investment; "
                "lower revenue alone does not prove unmet demand."
            )
        elif len(products) > 1:
            opportunity_product = str(products.iloc[-1]["Product"])
            opportunity_detail = (
                "No product falls below the equal-share reference in this data. "
                "Compare product-level margins, stock availability, and customer "
                "demand before choosing an expansion opportunity."
            )
        else:
            opportunity_product = None
            opportunity_detail = (
                "Only one product is represented. Test additional products or "
                "bundles with a limited pilot, and validate demand before scaling."
            )
        insights.append(
            {
                "title": "Product opportunity insight",
                "metric": (
                    f"Candidate: {opportunity_product}"
                    if opportunity_product
                    else "Validate product range"
                ),
                "detail": opportunity_detail,
                "tone": "neutral",
            }
        )

    order_count = max(int(total_orders), 0)
    aov = float(average_order_value)
    five_percent_uplift = aov * 0.05 * order_count
    ten_percent_uplift = aov * 0.10 * order_count
    insights.append(
        {
            "title": "Average order value analysis",
            "metric": f"${aov:,.2f} per order",
            "detail": (
                f"At the current {order_count:,} orders, a hypothetical 5% "
                f"increase in average order value would correspond to "
                f"${five_percent_uplift:,.2f} additional revenue; 10% would "
                f"correspond to ${ten_percent_uplift:,.2f}. These are scenario "
                "calculations, not forecasts; test bundles or complementary "
                "offers and monitor conversion alongside order value."
            ),
            "tone": "neutral",
        }
    )

    recent_average = (
        float(daily.iloc[-min(14, len(daily)) :]["Revenue"].mean())
        if not daily.empty
        else None
    )
    if future_revenue is not None and np.isfinite(float(future_revenue)):
        forecast_value = float(future_revenue)
        if recent_average is not None and recent_average > 0:
            forecast_change = _format_change(forecast_value, recent_average)
            outlook = (
                f"The stored forecast averages ${forecast_value:,.2f} per day, "
                f"{forecast_change} versus the latest recorded-date average of "
                f"${recent_average:,.2f}. Treat this model estimate as a planning "
                "signal and revisit it as new actual sales arrive."
            )
        else:
            outlook = (
                f"The stored forecast averages ${forecast_value:,.2f} per day. "
                "Compare it with actual results as new sales are recorded."
            )
        insights.append(
            {
                "title": "Forecast insight",
                "metric": f"${forecast_value:,.2f} average daily",
                "detail": outlook,
                "tone": (
                    "positive"
                    if recent_average is not None and forecast_value >= recent_average
                    else "caution"
                ),
            }
        )
    else:
        insights.append(
            {
                "title": "Forecast insight",
                "metric": "No forecast available",
                "detail": (
                    "Generate a forecast on the Forecasting page to add a forward "
                    "revenue estimate to this dashboard."
                ),
                "tone": "neutral",
            }
        )

    if concentration_label == "high":
        recommendation = (
            "Reduce product dependence risk: track the leading product weekly "
            "and run a controlled cross-sell test for another product. Compare "
            "incremental revenue and conversion before scaling."
        )
    elif recent_average is not None and len(daily) >= 4:
        latest = float(daily.iloc[-min(14, len(daily)) :]["Revenue"].sum())
        period_size = min(14, len(daily) // 2)
        prior = float(daily.iloc[-2 * period_size : -period_size]["Revenue"].sum())
        if latest < prior:
            recommendation = (
                "Investigate the recent revenue decline by checking sales activity "
                "and the leading product's contribution. Test one targeted sales "
                "or bundling action, then compare the next equivalent set of "
                "recorded dates before investing further."
            )
        else:
            recommendation = (
                "Protect current momentum: repeat the strongest observed product "
                "and sales-period patterns, then test one complementary offer. "
                "Review revenue and average order value after the next comparable "
                "sales period."
            )
    else:
        recommendation = (
            "Build a stronger baseline by collecting more sales dates and "
            "reviewing product revenue regularly. Then run one small, measurable "
            "offer test and compare revenue and average order value with the "
            "current baseline."
        )
    insights.append(
        {
            "title": "Actionable recommendation",
            "metric": "Next step",
            "detail": recommendation,
            "tone": "positive",
        }
    )

    # Keep total_revenue in view as a validation check against the itemized sales mix.
    if product_revenue > 0 and not np.isclose(product_revenue, float(total_revenue)):
        insights.insert(
            0,
            {
                "title": "Revenue reconciliation",
                "metric": f"Difference: ${float(total_revenue) - product_revenue:,.2f}",
                "detail": (
                    "The dashboard total and product-level revenue totals differ. "
                    "Review the uploaded records or grouping before relying on "
                    "product contribution and concentration findings."
                ),
                "tone": "caution",
            },
        )

    return insights


def render_dashboard() -> None:
    """Render the dashboard from the current session's validated sales data."""
    data = st.session_state["df"]
    daily_sales = st.session_state["daily_sales"]
    product_sales = st.session_state["product_sales"]
    insights = build_business_insights(
        total_revenue=float(st.session_state["total_revenue"]),
        total_orders=int(st.session_state["total_orders"]),
        average_order_value=float(st.session_state["average_order_value"]),
        product_sales=product_sales,
        daily_sales=daily_sales,
        future_revenue=st.session_state.get("future_revenue"),
    )
    render_executive_dashboard(data, daily_sales, product_sales, insights)
    health = calculate_business_health_score(
        data,
        st.session_state.get("forecast_data"),
    )
    st.subheader("Business health score")
    health_columns = st.columns([1, 1.6], vertical_alignment="center")
    with health_columns[0]:
        import plotly.graph_objects as go

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=health["overall_score"],
                number={"suffix": "/100"},
                title={"text": str(health["category"])},
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
        gauge.update_layout(height=260, margin={"t": 45, "b": 10, "l": 20, "r": 20})
        st.plotly_chart(gauge, width="stretch")
    with health_columns[1]:
        st.metric("Business Health Score", f"{health['overall_score']}/100")
        st.caption(f"Health status: {health['category']}")
        st.markdown("✅ **Strengths**")
        for item in health["strengths"]:
            st.markdown(f"- {item}")
        st.markdown("⚠ **Risks**")
        for item in health["risks"]:
            st.markdown(f"- {item}")
        st.markdown("📈 **Recommendations**")
        for item in health["recommendations"]:
            st.markdown(f"- {item}")
        with st.expander("Score breakdown"):
            for component, score in health["component_scores"].items():
                st.write(f"**{component}:** {score}/25")


st.title("AI Business Copilot")
st.caption("Upload sales data to explore performance and business insights.")

uploaded_file = st.file_uploader(
    "Upload a sales CSV",
    type=["csv"],
    help="Required columns: Date, Product, Quantity, Revenue.",
    key="sales_csv_upload",
)

if uploaded_file is not None:
    file_contents = uploaded_file.getvalue()
    file_hash = hashlib.sha256(file_contents).hexdigest()

    if file_hash != st.session_state["uploaded_file_hash"]:
        uploaded_data = None
        try:
            uploaded_data = pd.read_csv(BytesIO(file_contents))
            validated_data = validate_sales_data(uploaded_data)
        except (EmptyDataError, ParserError, UnicodeDecodeError, ValueError) as error:
            st.error(f"Unable to use this sales CSV: {error}")
            if isinstance(uploaded_data, pd.DataFrame):
                render_data_quality_report(uploaded_data)
            st.stop()

        # Replace saved data only after validation succeeds.
        save_sales_data(validated_data, uploaded_data)
        st.session_state["uploaded_file_hash"] = file_hash

if st.session_state["df"] is None:
    st.info(
        "Upload a CSV with Date, Product, Quantity, and Revenue columns to "
        "populate the dashboard and unlock the other pages."
    )
else:
    render_dashboard()
