"""Generate and download AI-authored executive business reports."""

import re
from difflib import SequenceMatcher
from datetime import datetime
from html import escape
from io import BytesIO
from typing import Any

import pandas as pd
import streamlit as st
from groq import APIError
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    ListFlowable,
    ListItem,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from utils.shared import (
    apply_compact_page_spacing,
    get_groq_client,
    get_groq_model,
    require_dashboard_data,
    render_forecast_prompt_card,
)
from utils.anomaly_detection import SEVERITY_LEVELS, detect_sales_anomalies
from utils.business_health import calculate_business_health_score
from utils.ceo_dashboard import (
    build_forecast_summary,
    build_ceo_scorecard,
    build_executive_priorities,
)
from utils.churn_prediction import analyze_churn
from utils.customer_segmentation import build_customer_segments

st.title("Executive reports")
st.caption("Create a consulting-grade report from your uploaded sales data.")

apply_compact_page_spacing()

if not require_dashboard_data():
    st.stop()

st.session_state.setdefault("ai_report", None)
st.session_state.setdefault("ai_pdf_bytes", None)

REQUIRED_REPORT_SECTIONS = (
    "Executive Summary",
    "Key Findings",
    "Business Risks",
    "Growth Opportunities",
    "Strategic Recommendations",
    "90-Day Action Plan",
)


def _required_report_value(key: str) -> Any:
    """Read one required session value, stopping with a clear message if absent."""
    value = st.session_state.get(key)
    if value is None:
        st.error(f"Required dashboard value '{key}' is unavailable. Re-upload the CSV.")
        st.stop()
    return value


def _build_report_prompt() -> str:
    """Build a bounded, data-grounded prompt from the shared dashboard state."""
    total_revenue = float(_required_report_value("total_revenue"))
    total_orders = int(_required_report_value("total_orders"))
    best_product = str(_required_report_value("best_product"))
    average_order_value = float(_required_report_value("average_order_value"))
    product_sales = _required_report_value("product_sales")
    daily_sales = _required_report_value("daily_sales")

    if not isinstance(product_sales, pd.DataFrame) or product_sales.empty:
        st.error("Product revenue data is unavailable. Re-upload the sales CSV.")
        st.stop()
    if not isinstance(daily_sales, pd.DataFrame) or daily_sales.empty:
        st.error("Daily revenue data is unavailable. Re-upload the sales CSV.")
        st.stop()

    forecast_value = st.session_state.get("future_revenue")
    if forecast_value is None:
        forecast_summary = (
            "No forecast has been generated yet. State that forecast data is "
            "unavailable; do not invent future revenue values."
        )
    else:
        forecast_summary = (
            "Prophet average forecast daily revenue for the next 30 days: "
            f"{float(forecast_value):,.2f} revenue units"
        )

    daily_data = daily_sales.copy()
    daily_data["Date"] = pd.to_datetime(daily_data["Date"], errors="coerce")
    daily_data = daily_data.dropna(subset=["Date"]).sort_values("Date")
    daily_window = daily_data.tail(120)
    history_note = (
        f"Showing {len(daily_window)} most recent daily totals out of "
        f"{len(daily_data)} total daily records."
    )

    return f"""Write a decision-ready executive briefing in the style of a leading
management consulting firm. Be specific, commercially practical, concise, and
analytical. Ground factual statements in the dashboard metrics and tables below.
Write as an analyst examining this dataset, not as a generic business coach.

OUTPUT CONTRACT — follow exactly:
- Output all six Markdown headings below, spelled exactly and in the stated order.
- Key Findings, Business Risks, and Growth Opportunities must each contain
  exactly 3 substantive items.
- Every item in those three sections must use all four labels, in this order:
  Observation, Business Impact, Recommendation, Expected Outcome. Put each
  field on its own line beneath one Markdown bullet; do not add sub-bullets.
- Observation must state a specific fact from the supplied metrics or tables.
  Business Impact must explain a financial or operational consequence, not
  merely say the finding is important.
- Recommendation must name the relevant product, metric, period, or observed
  pattern and prescribe a concrete action that directly addresses the
  observation. Expected Outcome must explain the plausible improvement and
  how it would be visible in revenue, order value, product mix, or units sold.
- Executive Summary must contain exactly 3 concise bullets.
- Strategic Recommendations must be exactly one each for Immediate,
  Medium-Term, and Long-Term.
- Strategic Recommendations must focus on dataset-supported decisions about
  revenue growth, product mix, inventory planning, pricing tests, or marketing.
- The 90-Day Action Plan must contain exactly three month groups (Month 1,
  Month 2, Month 3), each with exactly 3 concise action bullets. Do not create
  extra phases, milestone groups, recommendations, or bullets.
- Do not number recommendations. Use Markdown bullets only.
- Never omit a section, leave one blank, combine sections, or replace a required
  heading with a synonym. Do not add an introduction or conclusion outside these
  sections.
- Prefer distinct insights with reasoning and business impact. Avoid repeating
  the same observation, action, or caveat across bullets; remove filler.
- Do not use generic advice such as "validate the finding," "assign an owner,"
  "monitor outcomes," "review performance," "review metrics," or "diversify
  revenue" without a specific dataset-linked commercial tactic. Never substitute
  process language for a business action.
- If the data does not support a precise conclusion, state the limitation in
  Observation, explain its decision consequence in Business Impact, and
  recommend a small, named test using the available products or metrics.

Required sections and analysis:
# Executive Summary
Use three bullets covering overall recorded performance, the most decision-
relevant product/revenue signal, and one priority action.
# Key Findings
Give exactly three non-overlapping evidence-led findings. Select the strongest
signals from revenue, orders, average order value, product mix, quantity, daily
history, and forecast only when available. For each, follow this exact shape:
- **Observation:** [specific dataset fact]
  **Business Impact:** [specific financial or operational consequence]
  **Recommendation:** [specific action tied to the fact]
  **Expected Outcome:** [plausible improvement and how to measure it]
# Business Risks
Use exactly three items in this order: revenue risk, product concentration risk,
and demand volatility risk. Use the four-field shape specified above for every item.
When the data does not prove a risk, frame it as a specific exposure to monitor
and explain what evidence would confirm or reduce it. Do not assert unsupported
causes, margins, customer behavior, or market conditions.
# Growth Opportunities
Use exactly three distinct opportunities in this order: upselling using order
value/quantity, product opportunities grounded in the listed product mix, and revenue expansion
through a measurable test. Use the four-field shape specified above for every
item. Name the observed product(s), order value, quantity, or sales period that
supports the proposed action.
# Strategic Recommendations
Use exactly three bullets, one each labeled Immediate, Medium-Term, and
Long-Term. Tie each to an observed signal or data gap, with a proportionate
business impact and measurable success check. State an actual action involving
revenue growth, product optimization, inventory planning, pricing, or marketing;
avoid procedural tasks such as assigning owners or reviewing metrics.
# 90-Day Action Plan
Use only the subheadings **Month 1**, **Month 2**, and **Month 3**, in that
order, with exactly three practical action bullets beneath each. Sequence
baseline, focused execution, and evidence-led review. No other milestone groups.

Data integrity rules:
- Use the supplied figures and preserve their values. Do not invent benchmarks,
  margins, costs, customer counts, retention, competitors, market size, or causes.
- Revenue and order value are in the dataset's units; do not assume a currency
  if not provided.
- If no forecast is available, do not show forecast numbers. Use historical
  revenue and daily patterns to offer a cautious outlook and state the limitation.
- Limited data is not a reason to omit a section. Make a cautious, business-
  oriented observation from available revenue, product, quantity, and daily
  history, then recommend a low-risk validation step.

Dashboard metrics:
- Total revenue: {total_revenue:,.2f} revenue units
- Total orders: {total_orders:,}
- Best product by revenue: {best_product}
- Average order value: {average_order_value:,.2f} revenue units per order
- {forecast_summary}

Revenue by product:
{product_sales.to_string(index=False, max_rows=100)}

Daily revenue history:
{history_note}
{daily_window.to_string(index=False, max_rows=120)}
"""


def generate_ai_report() -> str:
    """Generate and complete the required consulting report sections."""
    response = get_groq_client().chat.completions.create(
        model=get_groq_model(),
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior strategy and finance consultant. Be "
                    "analytical, commercially practical, explicit about uncertainty, "
                    "and faithful to the provided figures. Follow the user's output "
                    "contract exactly; completeness and evidence integrity are mandatory."
                ),
            },
            {"role": "user", "content": _build_report_prompt()},
        ],
    )
    report = response.choices[0].message.content
    if not report or not report.strip():
        return ""
    return _complete_report_sections(report.strip())


INSIGHT_FIELD_ORDER = (
    "Observation",
    "Business Impact",
    "Recommendation",
    "Expected Outcome",
)


def _format_report_insight(
    observation: str,
    business_impact: str,
    recommendation: str,
    expected_outcome: str,
) -> str:
    """Format one evidence-led item consistently for Markdown and card rendering."""
    values = (observation, business_impact, recommendation, expected_outcome)
    return "\n  ".join(
        f"**{label}:** {value}"
        for label, value in zip(INSIGHT_FIELD_ORDER, values)
    )


def _parse_report_insight_items(body: str) -> list[dict[str, str]]:
    """Parse four-field insight items from generated or normalized Markdown."""
    label_pattern = re.compile(
        r"\**(Observation|Business Impact|Recommendation|Expected Outcome)"
        r"\**\s*:\s*",
        re.IGNORECASE,
    )
    field_names = {name.casefold(): name for name in INSIGHT_FIELD_ORDER}
    items: list[dict[str, str]] = []
    fields: dict[str, str] = {}
    active_field: str | None = None

    def finish_item() -> None:
        nonlocal fields, active_field
        if all(fields.get(name) for name in INSIGHT_FIELD_ORDER):
            items.append(fields.copy())
        fields = {}
        active_field = None

    for raw_line in body.splitlines():
        text = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", raw_line.strip())
        if not text:
            continue
        labels = list(label_pattern.finditer(text))
        if labels:
            for index, match in enumerate(labels):
                label = field_names[match.group(1).casefold()]
                if label == "Observation" and fields:
                    finish_item()
                value_end = (
                    labels[index + 1].start()
                    if index + 1 < len(labels)
                    else len(text)
                )
                value = text[match.end() : value_end].strip(" \t|")
                value = re.sub(r"\*\*", "", value).strip()
                if value:
                    fields[label] = " ".join(
                        part for part in (fields.get(label, ""), value) if part
                    )
                active_field = label
        elif fields and active_field:
            continuation = re.sub(r"\*\*", "", text).strip()
            if continuation:
                fields[active_field] = " ".join(
                    part
                    for part in (fields.get(active_field, ""), continuation)
                    if part
                )
    finish_item()
    return items


def _fallback_report_bullets() -> dict[str, list[str]]:
    """Build dataset-specific report insights when model output is incomplete."""
    revenue = float(_required_report_value("total_revenue"))
    orders = int(_required_report_value("total_orders"))
    best_product = str(_required_report_value("best_product"))
    average_order_value = float(_required_report_value("average_order_value"))
    product_sales = _required_report_value("product_sales")
    daily_sales = _required_report_value("daily_sales")
    sales = _required_report_value("df")
    forecast_value = st.session_state.get("future_revenue")

    product_data = product_sales.loc[:, ["Product", "Revenue"]].copy()
    product_data["Revenue"] = pd.to_numeric(product_data["Revenue"], errors="coerce")
    product_data = (
        product_data.dropna(subset=["Revenue"])
        .sort_values("Revenue", ascending=False)
        .reset_index(drop=True)
    )
    product_total = float(product_data["Revenue"].sum())
    top_product = (
        str(product_data.iloc[0]["Product"]) if not product_data.empty else best_product
    )
    second_product = (
        str(product_data.iloc[1]["Product"])
        if len(product_data) > 1
        else top_product
    )
    top_revenue = (
        float(product_data.iloc[0]["Revenue"]) if not product_data.empty else 0.0
    )
    second_revenue = (
        float(product_data.iloc[1]["Revenue"]) if len(product_data) > 1 else 0.0
    )
    top_share = top_revenue / product_total if product_total > 0 else 0.0
    second_share = second_revenue / product_total if product_total > 0 else 0.0

    top_units = 0.0
    if isinstance(sales, pd.DataFrame) and {"Product", "Quantity"}.issubset(
        sales.columns
    ):
        quantity_data = sales.loc[:, ["Product", "Quantity"]].copy()
        quantity_data["Quantity"] = pd.to_numeric(
            quantity_data["Quantity"], errors="coerce"
        )
        top_units = float(
            quantity_data.loc[
                quantity_data["Product"].astype(str) == top_product, "Quantity"
            ].sum()
        )

    forecast_note = (
        f"The stored 30-day forecast averages {float(forecast_value):,.2f} "
        "revenue units per day; treat it as a planning estimate, not a guarantee."
        if forecast_value is not None
        else "No forecast is available; use recorded sales periods for planning "
        "until enough actuals exist to compare performance."
    )
    daily_data = daily_sales.copy()
    if {"Date", "Revenue"}.issubset(daily_data.columns):
        daily_data["Date"] = pd.to_datetime(daily_data["Date"], errors="coerce")
        daily_data["Revenue"] = pd.to_numeric(daily_data["Revenue"], errors="coerce")
        daily_data = (
            daily_data.dropna(subset=["Date", "Revenue"])
            .sort_values("Date")
            .reset_index(drop=True)
        )
    else:
        daily_data = pd.DataFrame(columns=["Date", "Revenue"])

    recent_top_units: float | None = None
    if (
        isinstance(sales, pd.DataFrame)
        and {"Date", "Product", "Quantity"}.issubset(sales.columns)
        and not daily_data.empty
    ):
        recent_sales = sales.loc[:, ["Date", "Product", "Quantity"]].copy()
        recent_sales["Date"] = pd.to_datetime(recent_sales["Date"], errors="coerce")
        recent_sales["Quantity"] = pd.to_numeric(
            recent_sales["Quantity"], errors="coerce"
        )
        recent_dates = set(daily_data["Date"].tail(14))
        recent_top_units = float(
            recent_sales.loc[
                recent_sales["Date"].isin(recent_dates)
                & (recent_sales["Product"].astype(str) == top_product),
                "Quantity",
            ].sum()
        )

    if len(daily_data) >= 28:
        latest_period = daily_data.tail(14)
        prior_period = daily_data.iloc[-28:-14]
        latest_revenue = float(latest_period["Revenue"].sum())
        prior_revenue = float(prior_period["Revenue"].sum())
        period_change = (
            (latest_revenue - prior_revenue) / prior_revenue * 100
            if prior_revenue > 0
            else None
        )
        momentum_observation = (
            f"Revenue across the latest 14 recorded sales dates was "
            f"{latest_revenue:,.2f} units versus {prior_revenue:,.2f} units in "
            f"the preceding 14 recorded sales dates ({period_change:+.1f}%)."
            if period_change is not None
            else f"Revenue was {latest_revenue:,.2f} units across the latest 14 "
            "recorded sales dates; the preceding period had no positive "
            "revenue baseline for a percentage comparison."
        )
    else:
        momentum_observation = (
            f"The dataset contains {len(daily_data)} recorded revenue dates, "
            "which is insufficient for comparing two full 14-date periods."
        )

    if daily_data.empty:
        daily_pattern = "The uploaded data does not contain usable daily revenue values."
    else:
        peak_day = daily_data.loc[daily_data["Revenue"].idxmax()]
        low_day = daily_data.loc[daily_data["Revenue"].idxmin()]
        daily_pattern = (
            f"Recorded daily revenue ranged from {float(low_day['Revenue']):,.2f} "
            f"units on {low_day['Date']:%Y-%m-%d} to "
            f"{float(peak_day['Revenue']):,.2f} units on "
            f"{peak_day['Date']:%Y-%m-%d}."
        )

    bundle_action = (
        f"Test a bundle pairing {top_product} with {second_product}; compare "
        "orders, average order value, and each product's revenue with the "
        "preceding 14 recorded sales dates."
        if second_product != top_product
        else f"Test a complementary offer around {top_product}; compare orders "
        "and average order value with the preceding 14 recorded sales dates."
    )
    if recent_top_units is not None and recent_top_units > 0:
        inventory_action = (
            f"Plan initial replenishment for {top_product} against its observed "
            f"{recent_top_units:,.0f} units sold across the latest 14 recorded "
            "sales dates rather than the single-day revenue peak."
        )
    elif top_units > 0:
        inventory_action = (
            f"Plan initial replenishment for {top_product} against its observed "
            f"{top_units:,.0f} total units sold, then use the latest 14-date "
            "revenue pace rather than the single-day peak."
        )
    else:
        inventory_action = (
            f"Use recorded revenue for {top_product} and the observed daily "
            "high/low range to size a cautious replenishment test; do not treat "
            "a single peak day as a new run rate."
        )

    findings = [
        _format_report_insight(
            f"Recorded revenue is {revenue:,.2f} units across {orders:,} orders; "
            f"average order value is {average_order_value:,.2f} units.",
            "Revenue growth can come from more orders, a larger basket, or both; "
            "total revenue alone does not show which lever is changing.",
            bundle_action,
            "Comparing orders and average order value before and during the test "
            "will show whether it adds transactions, basket value, or both.",
        ),
        _format_report_insight(
            f"{top_product} contributes {top_share:.1%} of recorded product "
            f"revenue ({top_revenue:,.2f} of {product_total:,.2f} units); "
            f"{second_product} contributes {second_share:.1%}.",
            f"A decline in {top_product} could have an outsized effect on total "
            "sales if other products do not absorb the shortfall.",
            f"Feature {second_product} alongside {top_product} in a bounded "
            "cross-sell test and compare both products' revenue shares with "
            "their current mix.",
            "A higher contribution from the second product would broaden the "
            "revenue mix using products already present in the dataset.",
        ),
        _format_report_insight(
            momentum_observation,
            "Without a comparable date window, a short-term rise or fall can be "
            "mistaken for a sustained change and lead to poorly timed promotions "
            "or replenishment.",
            f"Compare product revenue and units across matched 14-date windows; "
            f"if {top_product} is driving a decline, shift the next offer toward "
            f"{top_product} or {second_product} and compare the same measures.",
            "A like-for-like comparison will distinguish a broad sales shift "
            "from a change concentrated in one product.",
        ),
    ]
    risks = [
        _format_report_insight(
            momentum_observation,
            "If the latest comparable period is weaker, sustained softness could "
            "reduce sales throughput and make revenue-dependent purchasing less "
            "reliable; the dataset does not establish the cause.",
            f"Compare {top_product} and {second_product} revenue over the same "
            "14-date windows, then direct the next time-bounded promotion to the "
            "product with the clearest recorded decline.",
            "A product-specific response can protect sales without applying a "
            "broad discount unsupported by the observed product pattern.",
        ),
        _format_report_insight(
            f"{top_product} accounts for {top_share:.1%} of product revenue; "
            f"{second_product} accounts for {second_share:.1%}.",
            f"If {top_product} demand weakens, the current mix leaves less "
            "revenue from other products to offset it.",
            f"Pair {top_product} with {second_product} in a controlled cross-sell "
            "offer and track the second product's revenue share and units.",
            "A measurable lift in the second product's share would reduce "
            "dependence on the leader while adding sales to the basket.",
        ),
        _format_report_insight(
            daily_pattern,
            "Planning to an average alone can overstate replenishment on weak "
            "dates or leave too little stock for stronger recorded periods.",
            inventory_action,
            "Using actual product units and the observed daily range should make "
            "replenishment more proportional to recorded sales.",
        ),
    ]
    opportunities = [
        _format_report_insight(
            f"Average order value is {average_order_value:,.2f} revenue units, "
            f"with {top_product} leading product revenue at {top_share:.1%}.",
            "A higher-value basket can increase revenue per order without "
            "requiring the same proportional increase in order count.",
            bundle_action,
            "If the offer lifts average order value without reducing order "
            "volume, it can create incremental revenue from the existing mix.",
        ),
        _format_report_insight(
            f"{second_product} contributes {second_share:.1%} of product revenue "
            f"versus {top_product} at {top_share:.1%}.",
            f"If {second_product} gains share, its sales can add growth beyond "
            "the current product leader.",
            f"Place {second_product} in the same promotion or product placement "
            f"as {top_product}; compare its revenue and units with the current "
            "product mix before extending the offer.",
            f"A sustained increase in {second_product}'s sales share would add "
            "growth from a product already present in the uploaded data.",
        ),
        _format_report_insight(
            momentum_observation,
            "A campaign based on an unsupported calendar assumption can waste "
            "marketing effort without lifting orders or revenue.",
            f"Run a time-bounded campaign featuring {top_product} and compare "
            "revenue, orders, and units against the preceding 14 recorded sales "
            "dates; if fewer than 28 dates exist, establish that baseline first.",
            "The matched-period result will show whether the offer adds "
            "measurable sales before broader marketing spend is considered.",
        ),
    ]
    recommendations = [
        f"Immediate — Launch a bounded {top_product}/{second_product} cross-sell "
        f"test; compare order count and average order value "
        f"({average_order_value:,.2f} units) with the preceding 14 recorded "
        "sales dates.",
        f"Medium-Term — {inventory_action}",
        f"Long-Term — Test a price or offer variant for {top_product}; compare "
        "revenue per order and units sold with its pre-test baseline. The dataset "
        "has no cost data, so do not infer margin improvement.",
    ]

    return {
        "Executive Summary": [
            f"Recorded revenue is {revenue:,.2f} units across {orders:,} orders, "
            f"with average order value of {average_order_value:,.2f} units; "
            "separate order growth from basket-value growth when assessing offers.",
            f"{top_product} contributes {top_share:.1%} of product revenue; "
            f"test a basket that also features {second_product} to grow sales "
            "outside the leading product.",
            forecast_note,
        ],
        "Key Findings": findings,
        "Business Risks": risks,
        "Growth Opportunities": opportunities,
        "Strategic Recommendations": recommendations,
        "90-Day Action Plan": [
            f"Month 1 — Establish the latest 14-date baseline for {top_product}: "
            "revenue, units, orders, and average order value.",
            f"Month 1 — Select a {top_product}/{second_product} bundle and "
            "compare it with the baseline over a matched sales period.",
            f"Month 1 — Set replenishment for {top_product} from recorded units "
            "and the daily high/low range, not a single peak day.",
            f"Month 2 — Run the {top_product}/{second_product} cross-sell offer "
            "for one matched 14-recorded-date period.",
            "Month 2 — Compare order count, average order value, and both "
            "products' revenue shares with the baseline.",
            f"Month 2 — If the bundle misses the baseline, test a separate "
            f"offer on {top_product} rather than discounting the full range.",
            "Month 3 — Repeat only the offer that improved revenue or basket "
            "value without weakening recorded unit sales.",
            f"Month 3 — Trial one price or offer variant for {top_product}; "
            "compare revenue per order and units with its pre-test period.",
            "Month 3 — Use matched-period results to set next-quarter product, "
            "replenishment, and marketing priorities.",
        ],
    }


def _complete_report_sections(report: str) -> str:
    """Normalize the report to the requested, concise executive structure."""
    parsed = _parse_report_sections(report)
    fallback = _fallback_report_bullets()
    aliases = {
        "Executive Summary": ("Executive Summary",),
        "Key Findings": ("Key Findings",),
        "Business Risks": ("Business Risks", "Risks"),
        "Growth Opportunities": ("Growth Opportunities", "Opportunities"),
        "Strategic Recommendations": (
            "Strategic Recommendations",
            "Recommended Actions",
        ),
        "90-Day Action Plan": ("90-Day Action Plan", "90 day action plan"),
    }
    completed: list[str] = []
    bullet_pattern = re.compile(r"^(?:[-*+]\s+|\d+[.)]\s+)(.+)$")

    def clean_bullets(body: str) -> list[str]:
        bullets: list[str] = []
        for raw_line in body.splitlines():
            line = raw_line.strip()
            if not line or re.fullmatch(r"#{1,3}\s*Month\s+[123]", line, re.I):
                continue
            match = bullet_pattern.match(line)
            line = match.group(1).strip() if match else line
            line = re.sub(r"^\*\*(.+?)\*\*\s*[:—-]\s*", r"\1: ", line)
            if line and not any(
                SequenceMatcher(None, line.casefold(), prior.casefold()).ratio() > 0.9
                for prior in bullets
            ):
                bullets.append(line)
        return bullets

    def is_low_quality_insight(item: dict[str, str]) -> bool:
        business_impact = item.get("Business Impact", "").casefold().strip()
        recommendation = item.get("Recommendation", "").casefold().strip()
        generic_impacts = (
            "could affect performance",
            "may affect results",
            "is important for the business",
            "could guide focused effort",
            "helps inform priorities",
        )
        if any(phrase in business_impact for phrase in generic_impacts) or not re.search(
            r"\b(?:revenue|sales|orders?|basket|units?|quantity|product|demand|"
            r"inventory|stock|replenish\w*|purchas\w*|discount|conversion|cash flow)\b",
            business_impact,
        ):
            return True
        generic_phrases = (
            "validate the finding",
            "validate findings",
            "assign an owner",
            "assign owner",
            "monitor outcomes",
            "review performance",
            "review metrics",
            "review the metrics",
            "define a measurable follow-up",
            "define measurable follow-up",
            "track outcomes",
            "diversify revenue",
        )
        if any(phrase in recommendation for phrase in generic_phrases):
            return True

        product_sales = st.session_state.get("product_sales")
        product_names = (
            product_sales["Product"].dropna().astype(str).tolist()
            if isinstance(product_sales, pd.DataFrame)
            and "Product" in product_sales
            else []
        )
        has_product_reference = any(
            len(name.strip()) >= 3 and name.casefold() in recommendation
            for name in product_names
        )
        has_metric_reference = bool(
            re.search(
                r"\d[\d,.]*\s*(?:%|units?)?\b"
                r"|\b(?:revenue|average order value|order value|orders|units|"
                r"quantity|recorded sales dates|daily revenue|product mix|forecast)\b",
                recommendation,
            )
        )
        has_business_action = bool(
            re.search(
                r"\b(?:bundle|pair|feature|promot\w*|campaign|replenish\w*|"
                r"price|pricing|offer|cross[\s-]?sell|upsell|test|launch|place|"
                r"set|reduce|increase|shift|allocate|stock|reorder|adjust|compare)\b",
                recommendation,
            )
        )
        return not (has_business_action and (has_product_reference or has_metric_reference))

    def has_generic_action(text: str) -> bool:
        recommendation = text.casefold()
        if any(
            phrase in recommendation
            for phrase in (
                "validate the finding",
                "assign an owner",
                "monitor outcomes",
                "review performance",
                "review metrics",
                "diversify revenue",
            )
        ):
            return True
        has_business_action = bool(
            re.search(
                r"\b(?:bundle|pair|feature|promot\w*|campaign|replenish\w*|"
                r"price|pricing|offer|cross[\s-]?sell|upsell|test|launch|place|"
                r"set|reduce|increase|shift|allocate|stock|reorder|adjust|compare)\b",
                recommendation,
            )
        )
        product_sales = st.session_state.get("product_sales")
        product_names = (
            product_sales["Product"].dropna().astype(str).tolist()
            if isinstance(product_sales, pd.DataFrame)
            and "Product" in product_sales
            else []
        )
        has_product_reference = any(
            len(name.strip()) >= 3 and name.casefold() in recommendation
            for name in product_names
        )
        has_metric_reference = bool(
            re.search(
                r"\d[\d,.]*\s*(?:%|units?)?\b"
                r"|\b(?:revenue|average order value|order value|orders|units|"
                r"quantity|recorded sales dates|daily revenue|product mix|forecast)\b",
                recommendation,
            )
        )
        return not (
            has_business_action and (has_product_reference or has_metric_reference)
        )

    def select_three(
        source: list[str],
        fallback_bullets: list[str],
        themes: tuple[tuple[str, ...], ...] = (),
    ) -> list[str]:
        selected: list[str] = []
        source_text = [item.casefold() for item in source]

        def is_duplicate(item: str) -> bool:
            return any(
                SequenceMatcher(None, item.casefold(), prior.casefold()).ratio() > 0.88
                for prior in selected
            )

        for theme_index, terms in enumerate(themes):
            candidate = next(
                (
                    item
                    for item, lowered in zip(source, source_text)
                    if any(term in lowered for term in terms)
                    and not is_duplicate(item)
                ),
                None,
            )
            fallback_item = fallback_bullets[theme_index]
            if candidate is not None:
                selected.append(candidate)
            elif not is_duplicate(fallback_item):
                selected.append(fallback_item)
            else:
                alternative = next(
                    (
                        item
                        for item in fallback_bullets
                        if not is_duplicate(item)
                    ),
                    next((item for item in source if not is_duplicate(item)), ""),
                )
                if alternative:
                    selected.append(alternative)
        for item in source:
            if len(selected) >= 3:
                break
            if not is_duplicate(item):
                selected.append(item)
        for item in fallback_bullets:
            if len(selected) >= 3:
                break
            if not is_duplicate(item):
                selected.append(item)
        return selected[:3]

    for section in REQUIRED_REPORT_SECTIONS:
        body = next((parsed[key] for key in aliases[section] if key in parsed), "")
        if section in {"Key Findings", "Business Risks", "Growth Opportunities"}:
            source = [
                _format_report_insight(
                    *(item[field] for field in INSIGHT_FIELD_ORDER)
                )
                for item in _parse_report_insight_items(body)
                if not is_low_quality_insight(item)
            ]
        elif section == "Strategic Recommendations":
            source = [
                item
                for item in clean_bullets(body)
                if not has_generic_action(item)
            ]
        else:
            source = clean_bullets(body)
        if section == "Business Risks":
            bullets = select_three(
                source,
                fallback[section],
                (
                    (
                        "revenue risk",
                        "revenue exposure",
                        "revenue decline",
                        "revenue",
                        "latest 14",
                    ),
                    (
                        "concentration risk",
                        "product concentration",
                        "product dependency",
                        "product mix",
                        "accounts for",
                    ),
                    (
                        "demand volatility",
                        "sales volatility",
                        "demand variability",
                        "daily revenue",
                        "recorded daily",
                    ),
                ),
            )
        elif section == "Growth Opportunities":
            bullets = select_three(
                source,
                fallback[section],
                (
                    ("upsell", "average order value", "basket", "bundle"),
                    (
                        "product opportunity",
                        "product mix",
                        "product test",
                        "contributes",
                    ),
                    (
                        "revenue expansion",
                        "expand revenue",
                        "revenue growth",
                        "campaign",
                        "marketing",
                        "time-bounded",
                    ),
                ),
            )
        elif section == "Strategic Recommendations":
            bullets = select_three(
                source,
                fallback[section],
                (
                    ("immediate",),
                    ("medium-term", "medium term"),
                    ("long-term", "long term"),
                ),
            )
        elif section == "90-Day Action Plan":
            month_sources: dict[int, list[str]] = {1: [], 2: [], 3: []}
            active_month: int | None = None
            for line in body.splitlines():
                text = line.strip()
                heading_match = re.fullmatch(
                    r"#{1,3}\s*Month\s+([123])", text, re.IGNORECASE
                )
                if heading_match:
                    active_month = int(heading_match.group(1))
                    continue
                prefix_match = re.match(
                    r"^(?:[-*+]\s*)?\**Month\s+([123])\**\s*(?:[-—:]\s*)?(.+)$",
                    text,
                    re.IGNORECASE,
                )
                if prefix_match:
                    active_month = int(prefix_match.group(1))
                    text = prefix_match.group(2).strip()
                else:
                    match = bullet_pattern.match(text)
                    text = match.group(1).strip() if match else text
                if active_month is not None and text:
                    month_sources[active_month].append(text)

            month_fallbacks = {
                1: fallback[section][0:3],
                2: fallback[section][3:6],
                3: fallback[section][6:9],
            }
            sections = [f"# {section}"]
            used_roadmap_items: list[str] = []
            for month in (1, 2, 3):
                month_source = [
                    item
                    for item in clean_bullets("\n".join(month_sources[month]))
                    if not any(
                        SequenceMatcher(
                            None,
                            item.casefold(),
                            prior.casefold(),
                        ).ratio()
                        > 0.88
                        for prior in used_roadmap_items
                    )
                ]
                month_fallback = [
                    item
                    for item in month_fallbacks[month]
                    if not any(
                        SequenceMatcher(
                            None,
                            item.casefold(),
                            prior.casefold(),
                        ).ratio()
                        > 0.88
                        for prior in used_roadmap_items
                    )
                ]
                month_items = select_three(
                    month_source,
                    month_fallback,
                )
                used_roadmap_items.extend(month_items)
                sections.append(
                    f"## Month {month}\n"
                    + "\n".join(f"- {item}" for item in month_items)
                )
            completed.append("\n\n".join(sections))
            continue
        else:
            bullets = select_three(source, fallback[section])
        completed.append(
            f"# {section}\n" + "\n".join(f"- {bullet}" for bullet in bullets)
        )
    return "\n\n".join(completed)


def _safe_pdf_text(text: str) -> str:
    """Normalize common Markdown punctuation for ReportLab's built-in fonts."""
    replacements = {
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
    for original, replacement in replacements.items():
        text = text.replace(original, replacement)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _markdown_inline_to_reportlab(text: str) -> str:
    """Escape model output for ReportLab XML while retaining simple emphasis."""
    safe_text = escape(_safe_pdf_text(text), quote=False)
    safe_text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe_text)
    safe_text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", safe_text)
    safe_text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", safe_text)
    return safe_text


def _markdown_to_flowables(report: str, styles: dict[str, Any]) -> list[Any]:
    """Convert report headings, paragraphs, and lists to page-flowing PDF elements."""
    lines = report.splitlines()
    flowables: list[Any] = []
    paragraph_lines: list[str] = []
    list_items: list[ListItem] = []

    def flush_paragraph() -> None:
        if paragraph_lines:
            paragraph_text = " ".join(line.strip() for line in paragraph_lines)
            flowables.append(
                Paragraph(
                    _markdown_inline_to_reportlab(paragraph_text),
                    styles["ReportBody"],
                )
            )
            flowables.append(Spacer(1, 7))
            paragraph_lines.clear()

    def flush_list() -> None:
        if list_items:
            flowables.append(
                ListFlowable(
                    list_items.copy(),
                    bulletType="bullet",
                    start="circle",
                    leftIndent=18,
                    bulletFontName="Helvetica",
                    bulletFontSize=7,
                    spaceAfter=8,
                )
            )
            list_items.clear()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            flush_list()
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            level = len(heading.group(1))
            style_name = "ReportHeading1" if level == 1 else "ReportHeading2"
            if flowables:
                flowables.append(Spacer(1, 8 if level == 1 else 4))
            flowables.append(
                Paragraph(
                    _markdown_inline_to_reportlab(heading.group(2)),
                    styles[style_name],
                )
            )
            flowables.append(Spacer(1, 5))
            continue

        bullet = re.match(r"^(?:[-*+]|\d+[.)])\s+(.+)$", line)
        if bullet:
            flush_paragraph()
            list_items.append(
                ListItem(
                    Paragraph(
                        _markdown_inline_to_reportlab(bullet.group(1)),
                        styles["ReportBody"],
                    ),
                    leftIndent=4,
                    bulletColor=colors.HexColor("#2563EB"),
                )
            )
            continue

        flush_list()
        paragraph_lines.append(line)

    flush_paragraph()
    flush_list()
    return flowables


def create_pdf_report(report: str) -> bytes:
    """Render the AI-generated Markdown report to a multi-page PDF in memory."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.7 * inch,
        title="AI Business Copilot Executive Report",
        author="AI Business Copilot",
    )
    base_styles = getSampleStyleSheet()
    styles = {
        "ReportHeading1": ParagraphStyle(
            "ReportHeading1",
            parent=base_styles["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=16,
            leading=20,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=12,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "ReportHeading2": ParagraphStyle(
            "ReportHeading2",
            parent=base_styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#245B83"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "ReportBody": ParagraphStyle(
            "ReportBody",
            parent=base_styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor("#202938"),
            spaceAfter=5,
            splitLongWords=True,
        ),
    }

    report_title = ParagraphStyle(
        "ReportTitle",
        parent=base_styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=21,
        leading=26,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17324D"),
        spaceAfter=5,
    )
    story: list[Any] = [
        Paragraph("AI Business Copilot", report_title),
        Paragraph("Executive Report", styles["ReportHeading2"]),
        Paragraph(
            f"Generated {datetime.now().astimezone().strftime('%B %d, %Y %H:%M %Z')}",
            styles["ReportBody"],
        ),
        Spacer(1, 14),
    ]
    story.extend(_markdown_to_flowables(report, styles))

    def draw_page_footer(pdf_canvas: Any, _document: Any) -> None:
        pdf_canvas.saveState()
        page_width, _ = letter
        pdf_canvas.setStrokeColor(colors.HexColor("#D8DEE8"))
        pdf_canvas.line(0.75 * inch, 0.55 * inch, page_width - 0.75 * inch, 0.55 * inch)
        pdf_canvas.setFont("Helvetica", 8)
        pdf_canvas.setFillColor(colors.HexColor("#667085"))
        pdf_canvas.drawString(0.75 * inch, 0.36 * inch, "Confidential - AI Business Copilot")
        pdf_canvas.drawRightString(page_width - 0.75 * inch, 0.36 * inch, f"Page {_document.page}")
        pdf_canvas.restoreState()

    document.build(story, onFirstPage=draw_page_footer, onLaterPages=draw_page_footer)
    return buffer.getvalue()


def _build_supplemental_appendix() -> str:
    """Compose a reproducible analytics appendix for PDF and text exports."""
    sales = st.session_state.get("df")
    if not isinstance(sales, pd.DataFrame) or sales.empty:
        return ""

    health = calculate_business_health_score(
        sales,
        st.session_state.get("forecast_data"),
    )
    lines = [
        "## Supplemental Analytics",
        "### Business Health Score",
        f"- **Business Health Score:** {health['overall_score']}/100",
        f"- **Health Status:** {health['category']}",
        f"- **Component scores:** {health['component_scores']}",
        "- **Strengths:**",
    ]
    lines.extend(f"  - {item}" for item in health["strengths"])
    lines.append("- **Risks:**")
    lines.extend(f"  - {item}" for item in health["risks"])
    lines.append("- **Recommendations:**")
    lines.extend(f"  - {item}" for item in health["recommendations"])

    anomalies = st.session_state.get("anomaly_results")
    if not isinstance(anomalies, pd.DataFrame):
        anomalies = detect_sales_anomalies(
            sales,
            st.session_state.get("raw_df"),
        )
        st.session_state["anomaly_results"] = anomalies
    lines.extend(["", "### Anomaly Summary", f"- **Detected anomalies:** {len(anomalies)}"])
    if anomalies.empty:
        lines.append("- No anomalies were detected in the available data.")
    else:
        severity_counts = (
            anomalies["Severity"]
            .value_counts()
            .reindex(SEVERITY_LEVELS, fill_value=0)
        )
        lines.append(
            "- **Severity counts:** "
            + ", ".join(f"{severity}: {int(count)}" for severity, count in severity_counts.items())
        )
        for _, item in anomalies.head(15).iterrows():
            lines.append(
                "- "
                f"{item['Date']} · {item['Category']} ({item['Severity']}) · "
                f"{item['Metric']}: expected {item['Expected Value']}, "
                f"actual {item['Actual Value']} "
                f"({item['Deviation %']}% deviation)."
            )
        if len(anomalies) > 15:
            lines.append(f"- {len(anomalies) - 15} additional anomaly rows are omitted from this summary.")

    anomaly_analysis = st.session_state.get("anomaly_analysis")
    if st.session_state.get("include_anomaly_in_report", True):
        lines.append("#### AI Anomaly Analysis")
        lines.append(
            anomaly_analysis.strip()
            if isinstance(anomaly_analysis, str) and anomaly_analysis.strip()
            else "No AI anomaly explanation has been generated yet."
        )

    segmentation = st.session_state.get("customer_segments")
    if not isinstance(segmentation, dict):
        segmentation = build_customer_segments(sales)
        st.session_state["customer_segments"] = segmentation
    lines.extend(["", "### Customer Segmentation Summary"])
    if not segmentation.get("available"):
        lines.append(f"- {segmentation.get('message', 'Customer segmentation is unavailable.')}")
    else:
        summary = segmentation["summary"]
        lines.append(
            f"- {segmentation['message']} "
            f"Customer identifier: {segmentation['customer_column']}."
        )
        for _, segment in summary.iterrows():
            lines.append(
                "- "
                f"{segment['Segment']}: {int(segment['Customers'])} customers; "
                f"revenue {float(segment['Revenue']):,.2f} "
                f"({float(segment['Revenue Contribution %']):.1f}%); "
                f"average customer value {float(segment['Average Customer Value']):,.2f}; "
                f"recent-purchase indicator {float(segment['Retention Indicator']):.1f}%."
            )

    if st.session_state.get("include_segment_insights_in_report", True):
        lines.append("#### AI Segment Insights")
        segment_insights = st.session_state.get("segment_ai_insights")
        lines.append(
            segment_insights.strip()
            if segmentation.get("available")
            and isinstance(segment_insights, str)
            and segment_insights.strip()
            else (
                "No AI segment insights have been generated yet."
                if segmentation.get("available")
                else "AI segment insights are unavailable because customer segmentation could not be calculated."
            )
        )

    churn = st.session_state.get("churn_results")
    if not isinstance(churn, dict):
        churn = analyze_churn(sales)
        st.session_state["churn_results"] = churn
    lines.extend(["", "### Churn Overview"])
    if not churn.get("available"):
        lines.append(
            "- Customer-level churn analysis is unavailable: "
            f"{churn.get('message', 'Required customer data is missing.')}"
        )
    else:
        lines.extend(
            [
                f"- **Analysis method:** {churn['mode'].replace('_', ' ').title()}",
                f"- **Risk counts:** High {churn['high_risk_count']}, "
                f"Medium {churn['medium_risk_count']}, Low {churn['low_risk_count']}.",
                f"- **Method notes:** {churn['message']}",
            ]
        )
        high_risk = churn["customers"].loc[
            churn["customers"]["Risk Level"].eq("High Risk")
        ].head(10)
        lines.append("#### High Risk Customers")
        if high_risk.empty:
            lines.append("- No customers are currently classified as High Risk.")
        else:
            for _, customer in high_risk.iterrows():
                lines.append(
                    "- "
                    f"{customer['Customer']}: estimated probability "
                    f"{float(customer['Churn Probability']):.1%}; "
                    f"{float(customer['Days Since Last Purchase']):.0f} days since "
                    "last purchase; transaction-row count "
                    f"{int(customer['Purchase Count'])}."
                )
        if churn.get("metrics"):
            lines.append(
                "- **Held-out Random Forest metrics:** "
                + ", ".join(
                    f"{metric} {value:.1%}"
                    for metric, value in churn["metrics"].items()
                )
            )
        lines.append("#### Retention Recommendations")
        if churn["high_risk_count"]:
            lines.append(
                "- Prioritize a small, personalized outreach test for High Risk customers; compare subsequent purchase activity with an appropriate baseline."
            )
        if churn["medium_risk_count"]:
            lines.append(
                "- Test a timely reminder or relevant offer for Medium Risk customers and measure response before expanding the campaign."
            )
        if not churn["high_risk_count"] and not churn["medium_risk_count"]:
            lines.append(
                "- Maintain a consistent purchase-history baseline and re-evaluate the risk mix as additional customer transactions are collected."
            )
        lines.append("#### AI Churn Analysis")
        churn_analysis = st.session_state.get("churn_analysis")
        if st.session_state.get("include_churn_in_report", True):
            lines.append(
                churn_analysis.strip()
                if isinstance(churn_analysis, str) and churn_analysis.strip()
                else "AI churn analysis has not been generated."
            )
        else:
            lines.append("AI churn analysis was excluded from this report.")

    forecast_summary = build_forecast_summary(
        st.session_state.get("daily_sales"),
        st.session_state.get("forecast_data"),
    )
    scorecard = build_ceo_scorecard(
        health,
        forecast_summary,
        churn,
        segmentation,
        anomalies,
    )
    ceo_risks, ceo_opportunities = build_executive_priorities(
        health,
        forecast_summary,
        churn,
        segmentation,
        anomalies,
    )
    lines.extend(
        [
            "",
            "### CEO Dashboard",
            "#### Executive KPI Snapshot",
            f"- **Revenue:** {float(st.session_state['total_revenue']):,.2f}",
            f"- **Orders / transaction rows:** {int(st.session_state['total_orders']):,}",
            f"- **Average revenue per row:** {float(st.session_state['average_order_value']):,.2f}",
            f"- **Top product:** {st.session_state['best_product']}",
            f"- **Business health:** {health['overall_score']}/100 ({health['category']})",
            f"- **Forecast outlook:** {forecast_summary['outlook']}; "
            + (
                f"{float(forecast_summary['growth_pct']):+.1f}% versus recent daily average."
                if forecast_summary.get("growth_pct") is not None
                else "forecast comparison unavailable."
            ),
            "#### CEO Score",
            f"- **Overall CEO Score:** {scorecard['score']}/100 ({scorecard['category']})",
            "- **Score components:** "
            + ", ".join(
                f"{component} {value:.0f}/100"
                for component, value in scorecard["components"].items()
            ),
            "#### Executive Recommendations",
        ]
    )
    lines.extend(f"- {item}" for item in ceo_risks)
    lines.append("#### Growth Opportunities")
    lines.extend(f"- {item}" for item in ceo_opportunities)
    ceo_briefing = st.session_state.get("ceo_briefing")
    lines.append("#### CEO Briefing")
    lines.append(
        ceo_briefing.strip()
        if isinstance(ceo_briefing, str) and ceo_briefing.strip()
        else "CEO briefing has not been generated."
    )

    return "\n".join(lines)


def _metrics_text_report() -> str:
    """Create the basic downloadable report when no AI report has been generated."""
    forecast_value = st.session_state.get("future_revenue")
    forecast_text = (
        f"${float(forecast_value):,.2f} average forecast daily revenue"
        if forecast_value is not None
        else "Not available; generate a forecast on the Forecasting page."
    )
    return (
        "AI Business Copilot - Business Report\n\n"
        f"Total revenue: ${float(st.session_state['total_revenue']):,.2f}\n"
        f"Total orders: {int(st.session_state['total_orders']):,}\n"
        f"Best product: {st.session_state['best_product']}\n"
        f"Average order value: ${float(st.session_state['average_order_value']):,.2f}\n"
        f"Revenue forecast: {forecast_text}\n"
    )


def _parse_report_sections(report_text: str) -> dict[str, str]:
    """Organize existing generated Markdown for display without changing its text."""
    aliases = {
        "Executive Summary": {"executive summary"},
        "Key Findings": {"key findings"},
        "Business Risks": {"business risks", "risks"},
        "Growth Opportunities": {"growth opportunities", "opportunities"},
        "Strategic Recommendations": {
            "recommended actions",
            "strategic recommendations",
        },
        "90-Day Action Plan": {"90-day action plan", "90 day action plan"},
        "Revenue Forecast Outlook": {"revenue forecast outlook"},
        "Conclusion": {"conclusion"},
    }
    parsed: dict[str, list[str]] = {}
    current: str | None = None
    for line in report_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip().casefold()
            month_heading = re.fullmatch(r"month\s+([123])", heading)
            if current == "90-Day Action Plan" and month_heading:
                parsed[current].append(f"## Month {month_heading.group(1)}")
                continue
            current = next(
                (
                    section
                    for section, names in aliases.items()
                    if heading in names
                ),
                None,
            )
            if current:
                parsed.setdefault(current, [])
            continue
        if current:
            parsed[current].append(line)
    return {
        section: "\n".join(lines).strip()
        for section, lines in parsed.items()
        if "\n".join(lines).strip()
    }


def _render_report_section(
    title: str,
    body: str,
) -> None:
    """Present existing report content inside a consistent executive card."""
    with st.container(border=True):
        st.markdown(f"### {title}")
        if body:
            st.markdown(body)


def _report_items(body: str) -> list[str]:
    """Split existing section text into readable display-only card items."""
    items: list[str] = []
    for line in body.splitlines():
        text = line.strip()
        if not text:
            continue
        text = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", text)
        if text:
            items.append(text)
    return items or [body.strip()]


def _render_report_roadmap(body: str) -> None:
    """Render exactly three month cards, each containing its action bullets."""
    month_items: dict[int, list[str]] = {1: [], 2: [], 3: []}
    active_month: int | None = None
    for line in body.splitlines():
        text = line.strip()
        month_heading = re.fullmatch(r"#{1,3}\s*Month\s+([123])", text, re.IGNORECASE)
        if month_heading:
            active_month = int(month_heading.group(1))
            continue
        month_prefix = re.match(
            r"^(?:[-*+]\s*)?\**Month\s+([123])\**\s*(?:[-—:]\s*)?(.+)$",
            text,
            re.IGNORECASE,
        )
        if month_prefix:
            active_month = int(month_prefix.group(1))
            text = month_prefix.group(2).strip()
        else:
            text = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", text)
        if active_month is not None and text:
            month_items[active_month].append(text)

    columns = st.columns(3)
    for column, month in zip(columns, (1, 2, 3)):
        with column:
            with st.container(border=True):
                st.markdown(f"### Month {month}")
                items = month_items[month][:3]
                if items:
                    st.markdown("\n".join(f"- {item}" for item in items))
                else:
                    st.caption("No actions were supplied for this month.")


def _render_report_item_cards(
    title: str,
    body: str,
    *,
    icon: str,
) -> None:
    """Present report items without adding synthetic impact or action text."""
    structured_items = _parse_report_insight_items(body)
    item_label = {
        "Key Findings": "Finding",
        "Business Risks": "Risk",
        "Growth Opportunities": "Opportunity",
        "Strategic Recommendations": "Recommendation",
    }.get(title, title)
    if structured_items:
        items = [
            (f"{item_label} {index + 1}", item)
            for index, item in enumerate(structured_items)
        ]
    else:
        items = []
        for index, item in enumerate(_report_items(body)):
            match = re.match(r"\*\*(.+?)\*\*\s*[:—-]\s*(.*)", item)
            if not match:
                match = re.match(r"^([^—:]+)\s*[—:]\s*(.*)$", item)
            item_title = (
                match.group(1).strip()
                if match
                else f"{item_label} {index + 1}"
            )
            item_body = match.group(2).strip() if match else item
            items.append((item_title, item_body))

    columns = st.columns(2)
    for index, (item_title, item) in enumerate(items):
        with columns[index % 2]:
            with st.container(border=True):
                st.markdown(f"### {icon} {item_title}".strip())
                if isinstance(item, dict):
                    for field in INSIGHT_FIELD_ORDER:
                        st.markdown(f"**{field}**")
                        st.markdown(item[field])
                else:
                    st.markdown(item)


st.markdown(
    """
    <style>
    .stApp { background: #0b1422; color: #eaf0f8; }
    [data-testid="stHeader"] { background: rgba(11,20,34,.9); }
    [data-testid="stMainBlockContainer"] { padding-top: 1.25rem; }
    .report-hero {
        background: radial-gradient(ellipse at 88% 12%, rgba(66,214,197,.15), transparent 38%),
            linear-gradient(115deg, #172941 0%, #102033 58%, #12343d 100%);
        border: 1px solid rgba(120,169,255,.22); border-radius: 23px;
        padding: 1.4rem 1.75rem; margin: .15rem 0 1rem;
        box-shadow: 0 20px 52px rgba(0,0,0,.24);
    }
    .report-eyebrow { color: #42d6c5; font-size: .74rem; font-weight: 750;
        letter-spacing: .15em; text-transform: uppercase; }
    .report-hero h1 { color: #f5f8fc; font-size: clamp(1.9rem, 3vw, 2.55rem);
        margin: .35rem 0 .3rem; letter-spacing: -.04em; }
    .report-hero p { color: #aebdd0; margin: 0; }
    .report-section { margin: 1.4rem 0 .6rem; padding-bottom: .45rem;
        border-bottom: 1px solid rgba(154,170,192,.18); }
    .report-section h2 { color: #f2f6fb; font-size: 1.4rem; font-weight: 730;
        letter-spacing: -.025em; margin: 0 0 .25rem; }
    .report-section p { color: #9aaac0; font-size: .9rem; margin: 0; }
    .snapshot-note { color: #9aaac0; font-size: .85rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(150deg, rgba(22,36,58,.96), rgba(17,29,47,.96));
        border-color: rgba(154,170,192,.17) !important;
        border-radius: 17px !important; box-shadow: 0 12px 30px rgba(0,0,0,.14);
        transition: transform .18s ease, border-color .18s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(66,214,197,.36) !important; transform: translateY(-2px);
    }
    [data-testid="stMetric"] { background: transparent; }
    [data-testid="stMetricLabel"] { color: #b9c8da !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #f5f8fc !important;
        font-size: clamp(1.45rem, 2.2vw, 2rem) !important;
        font-weight: 740 !important; letter-spacing: -.04em; }
    div.stButton > button[kind="primary"] { background: #138f87; border: 0;
        border-radius: 10px; color: white; font-weight: 700; min-height: 2.65rem; }
    div.stButton > button[kind="primary"]:hover { background: #17a79c; }
    @media (max-width: 700px) {
        .report-hero { padding: 1.1rem; border-radius: 17px; }
        .report-section { margin-top: 1.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

report = st.session_state.get("ai_report")
has_report = isinstance(report, str) and bool(report.strip())
report_text = report.strip() if has_report else ""
sections = _parse_report_sections(report_text) if has_report else {}
report = report_text if has_report else None

header, pdf_controls = st.columns([5, 1.35], vertical_alignment="center")
with header:
    st.markdown(
        """
        <div class="report-hero">
          <div class="report-eyebrow">Leadership briefing · AI business intelligence</div>
          <h1>Executive Reports</h1>
          <p>Evidence-led findings, risk perspective, and a clear agenda for action.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with pdf_controls:
    st.markdown("#### PDF Export")
    st.checkbox(
        "Include supplemental analytics",
        key="include_supplemental_pdf",
        value=True,
        help=(
            "Append the health score, anomaly findings and AI explanation, "
            "customer segmentation, churn overview, and CEO scorecard/briefing."
        ),
    )
    if report is None:
        st.caption("Generate a report to enable PDF export.")
    else:
        if st.button("Generate AI PDF", key="generate_ai_pdf", type="primary", width="stretch"):
            try:
                pdf_report = report
                if st.session_state.get("include_supplemental_pdf", True):
                    pdf_report += "\n\n" + _build_supplemental_appendix()
                st.session_state["ai_pdf_bytes"] = create_pdf_report(pdf_report)
            except (OSError, ValueError) as error:
                st.session_state["ai_pdf_bytes"] = None
                st.error(f"Could not generate the executive PDF: {error}")

        pdf_bytes = st.session_state.get("ai_pdf_bytes")
        if isinstance(pdf_bytes, bytes) and pdf_bytes:
            st.download_button(
                "📘 Download Executive PDF",
                data=pdf_bytes,
                file_name="ai_executive_report.pdf",
                mime="application/pdf",
                width="stretch",
            )

snapshot_revenue = float(st.session_state["total_revenue"])
snapshot_forecast = st.session_state.get("future_revenue")
snapshot_top_product = str(st.session_state["best_product"])
product_sales = st.session_state.get("product_sales")
total_product_revenue = (
    float(product_sales["Revenue"].sum())
    if isinstance(product_sales, pd.DataFrame) and not product_sales.empty
    else 0.0
)
top_product_revenue = (
    float(product_sales.iloc[0]["Revenue"])
    if isinstance(product_sales, pd.DataFrame) and not product_sales.empty
    else 0.0
)
top_product_share = (
    top_product_revenue / total_product_revenue if total_product_revenue > 0 else 0.0
)
report_risk = sections.get("Business Risks", "")
first_risk_line = next(
    (line.strip() for line in report_risk.splitlines() if line.strip()),
    "",
)
first_risk_line = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", first_risk_line)
first_risk_line = re.sub(r"\*+", "", first_risk_line).strip()
if first_risk_line:
    highest_risk = first_risk_line[:40] + ("…" if len(first_risk_line) > 40 else "")
elif top_product_share >= 0.6:
    highest_risk = "Product concentration"
elif snapshot_forecast is None:
    highest_risk = "Limited data visibility"
else:
    highest_risk = "Limited data visibility"

st.markdown(
    '<div class="report-section"><h2>Executive Snapshot</h2>'
    '<p>The headline position and most decision-relevant signals from this dataset.</p></div>',
    unsafe_allow_html=True,
)
snapshot_columns = st.columns(4 if snapshot_forecast is not None else 3)
snapshot_columns[0].metric("💰 Revenue", f"${snapshot_revenue:,.2f}", border=True)
if snapshot_forecast is not None:
    snapshot_columns[1].metric(
        "📈 Forecast Outlook",
        f"${float(snapshot_forecast):,.2f}/day",
        border=True,
    )
product_column = snapshot_columns[2] if snapshot_forecast is not None else snapshot_columns[1]
risk_column = snapshot_columns[3] if snapshot_forecast is not None else snapshot_columns[2]
product_column.metric("🏆 Top Product", snapshot_top_product, border=True)
risk_column.metric("⚠ Highest Risk", highest_risk, border=True)
if snapshot_forecast is None:
    render_forecast_prompt_card()

if not has_report:
    st.info("Generate an AI report to populate the executive sections below.")
if st.button("Generate AI Report", type="primary", key="generate_ai_report"):
    st.session_state["ai_report"] = None
    st.session_state["ai_pdf_bytes"] = None
    try:
        with st.spinner("Preparing your executive report..."):
            generated_report = generate_ai_report()
    except APIError as error:
        st.error(f"Groq could not generate the report: {error}")
    else:
        if not generated_report:
            st.error("Groq returned an empty report. Please try again.")
        else:
            st.session_state["ai_report"] = generated_report
            st.success("Executive report generated.")
            st.rerun()

if has_report:
    st.markdown(
        '<div class="report-section"><h2>Executive Briefing</h2>'
        '<p>A structured reading of the AI report, organized for leadership review.</p></div>',
        unsafe_allow_html=True,
    )
    section_order = (
        ("Executive Summary", "Executive Summary"),
        ("Key Findings", "Key Findings"),
        ("Business Risks", "Business Risks"),
        ("Growth Opportunities", "Growth Opportunities"),
        ("Strategic Recommendations", "Strategic Recommendations"),
        ("90-Day Action Plan", "90-Day Action Plan"),
    )
    for section_title, source_title in section_order:
        source_body = sections.get(source_title)
        if not source_body:
            placeholder = (
                "No explicit 90-day plan was included in this report. Use the "
                "recommendations above to define owners, timing, and measurable "
                "checkpoints before committing resources."
                if section_title == "90-Day Action Plan"
                else f"No separate {section_title.casefold()} section was included "
                "in this generated report."
            )
            _render_report_section(section_title, placeholder)
        elif section_title == "Executive Summary":
            _render_report_section(section_title, source_body)
        else:
            st.markdown(
                f'<div class="report-section"><h2>{section_title}</h2></div>',
                unsafe_allow_html=True,
            )
        if source_body and section_title == "90-Day Action Plan":
            _render_report_roadmap(source_body)
        elif source_body and section_title == "Business Risks":
            _render_report_item_cards(
                section_title,
                source_body,
                icon="🔴",
            )
        elif source_body and section_title == "Growth Opportunities":
            _render_report_item_cards(
                section_title,
                source_body,
                icon="🟢",
            )
        elif source_body and section_title not in {
            "Executive Summary",
            "90-Day Action Plan",
            "Business Risks",
            "Growth Opportunities",
        }:
            _render_report_item_cards(
                section_title,
                source_body,
                icon="",
            )

text_report = _metrics_text_report()
if has_report:
    text_report = f"{text_report}\n\n{report}\n"

st.markdown(
    '<div class="report-section"><h2>Report Downloads</h2>'
    '<p>Keep a portable copy of the current executive analysis.</p></div>',
    unsafe_allow_html=True,
)
st.download_button(
    "📝 Download text report",
    data=text_report,
    file_name="business_report.txt",
    mime="text/plain",
)
