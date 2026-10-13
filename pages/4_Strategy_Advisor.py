"""Groq-powered strategic recommendations based on uploaded sales and forecast data."""

import re
from typing import Any

import pandas as pd
import streamlit as st
from groq import APIError

from utils.shared import (
    apply_compact_page_spacing,
    get_groq_client,
    get_groq_model,
    require_dashboard_data,
    render_forecast_prompt_card,
)

st.set_page_config(
    page_title="Strategy advisor | AI Business Copilot",
    page_icon=":material/strategy:",
    layout="wide",
)

apply_compact_page_spacing()
st.title("Strategy advisor")
st.caption(
    "Turn your sales history and revenue forecast into prioritized, "
    "evidence-based business actions."
)

if not require_dashboard_data():
    st.stop()

st.session_state.setdefault("strategy_report", None)


def _sales_momentum(daily_sales: pd.DataFrame) -> tuple[str, float | None]:
    """Compare the latest two complete 14-day periods when enough history exists."""
    if not {"Date", "Revenue"}.issubset(daily_sales.columns):
        return "Unavailable: daily sales data must include Date and Revenue.", None

    data = daily_sales.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Revenue"] = pd.to_numeric(data["Revenue"], errors="coerce")
    data = data.dropna(subset=["Date", "Revenue"]).sort_values("Date")
    if len(data) < 28:
        return "Unavailable: at least 28 daily totals are needed for comparison.", None

    recent_revenue = float(data.tail(14)["Revenue"].sum())
    prior_revenue = float(data.iloc[-28:-14]["Revenue"].sum())
    if prior_revenue <= 0:
        return "Unavailable: the prior 14-day revenue baseline is zero.", None

    change = (recent_revenue - prior_revenue) / prior_revenue * 100
    return (
        f"{change:+.1f}% for the latest 14 days versus the preceding 14 days.",
        change,
    )


def _build_business_context() -> tuple[str, dict[str, Any]]:
    """Create a bounded prompt context and calculated indicators from session data."""
    daily_sales = st.session_state["daily_sales"]
    product_sales = st.session_state["product_sales"]
    sales = st.session_state["df"]

    if not isinstance(daily_sales, pd.DataFrame) or daily_sales.empty:
        st.error("Daily sales data is unavailable. Upload the CSV again on the Dashboard.")
        st.stop()
    if not isinstance(product_sales, pd.DataFrame) or product_sales.empty:
        st.error("Product sales data is unavailable. Upload the CSV again on the Dashboard.")
        st.stop()
    if not isinstance(sales, pd.DataFrame) or sales.empty:
        st.error("Uploaded sales data is unavailable. Upload the CSV again on the Dashboard.")
        st.stop()

    revenue = float(st.session_state["total_revenue"])
    orders = int(st.session_state["total_orders"])
    average_order_value = float(st.session_state["average_order_value"])
    best_product = str(st.session_state["best_product"])
    forecast_value = st.session_state.get("future_revenue")
    momentum_text, momentum_percent = _sales_momentum(daily_sales)

    if forecast_value is None:
        forecast_text = (
            "No Prophet forecast is available yet. Do not invent a forecast; "
            "state that the Forecasting page must be run first."
        )
    else:
        forecast_text = (
            "Prophet's average predicted daily revenue for the next 30 days is "
            f"${float(forecast_value):,.2f}. Treat it as an estimate, not a guarantee."
        )

    recent_products = product_sales.sort_values("Revenue", ascending=False).head(20)
    recent_daily = daily_sales.tail(60)
    date_min = pd.to_datetime(sales["Date"], errors="coerce").min()
    date_max = pd.to_datetime(sales["Date"], errors="coerce").max()
    date_range = (
        f"{date_min:%Y-%m-%d} through {date_max:%Y-%m-%d}"
        if pd.notna(date_min) and pd.notna(date_max)
        else "unavailable"
    )
    context = f"""BUSINESS DATA (currency units are not specified; use the same units):
- Total recorded revenue: {revenue:.2f}
- Recorded orders/rows: {orders}
- Average order value: {average_order_value:.2f}
- Highest-revenue product: {best_product}
- Sales data date range: {date_range}
- Revenue momentum: {momentum_text}
- Forecast: {forecast_text}

Revenue by product (top 20):
{recent_products.to_string(index=False, max_rows=20)}

Recent daily revenue history (up to 60 rows):
{recent_daily.to_string(index=False, max_rows=60)}

DATA LIMITATIONS:
The source contains sales transactions with Date, Product, Quantity, and Revenue.
It contains no cost, profit margin, customer identity, repeat-purchase,
marketing-channel, inventory, competitor, or market-size data. Do not state
unsupported facts about these topics. For cost, retention, and expansion
recommendations, identify the missing data and propose a measurable pilot or
analysis rather than claiming an established result.
"""
    metrics = {
        "revenue": revenue,
        "orders": orders,
        "average_order_value": average_order_value,
        "best_product": best_product,
        "forecast_value": forecast_value,
        "momentum_text": momentum_text,
        "momentum_percent": momentum_percent,
        "product_count": int(product_sales["Product"].nunique())
        if "Product" in product_sales
        else 0,
    }
    return context, metrics


STRATEGY_AREAS = (
    ("Increase Revenue", "Increase revenue without assuming unobserved margins."),
    ("Reduce Costs", "Find cost-control opportunities while acknowledging costs are not provided."),
    (
        "Improve Customer Retention",
        "Recommend retention measurement and tests without inventing customer behavior.",
    ),
    (
        "Product Expansion",
        "Assess product expansion using observed product revenue and propose validation.",
    ),
    (
        "Market Expansion",
        "Recommend evidence-led market expansion research and low-risk pilots.",
    ),
)


def generate_strategy_recommendations(context: str) -> dict[str, str]:
    """Ask Groq for actionable recommendations in each strategic area."""
    area_instructions = "\n".join(
        f"## {title}\n{instruction}\n"
        "Provide: (1) the evidence or limitation, (2) 2-3 specific actions, "
        "(3) an owner or time horizon, and (4) a measurable success metric."
        for title, instruction in STRATEGY_AREAS
    )
    response = get_groq_client().chat.completions.create(
        model=get_groq_model(),
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior strategy consultant advising a small-business "
                    "leadership team. Be specific, commercially practical, and "
                    "transparent about uncertainty. Ground recommendations only in "
                    "the supplied data. Never invent facts, benchmarks, costs, "
                    "customer behavior, market size, or forecast figures."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{context}\n\nCreate a concise but substantive advisory plan "
                    "for these five areas. Use the exact Markdown headings shown "
                    "below, include a short **Strategic objective:** line under "
                    "each heading, then number the prioritized recommendations. "
                    "State when more data or a pilot is needed.\n\n"
                    f"{area_instructions}"
                ),
            },
        ],
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        return {}

    recommendations: dict[str, str] = {}
    for title, _ in STRATEGY_AREAS:
        heading_pattern = re.escape(title)
        match = re.search(
            rf"(?im)^##\s+{heading_pattern}\s*$([\s\S]*?)"
            r"(?=^##\s+|\Z)",
            content,
        )
        if match and match.group(1).strip():
            recommendations[title] = match.group(1).strip()
    return recommendations


def _brief_values(metrics: dict[str, Any]) -> dict[str, Any]:
    """Translate existing dashboard indicators into a concise, trend-led brief."""
    momentum = metrics["momentum_percent"]
    score = (
        max(0, min(100, round(50 + momentum)))
        if momentum is not None
        else 50
    )
    if momentum is None:
        outlook = "Trend forming"
        growth = "Not yet established"
        risk = "Evidence gap"
        risk_description = (
            "There is not yet enough comparable daily history to assess momentum."
        )
        primary_action = (
            f"Run a measured revenue pilot around {metrics['best_product']} and "
            "track results against the current sales baseline."
        )
    elif momentum > 0:
        outlook = f"Positive · {momentum:+.1f}%"
        growth = "Promising"
        risk = "Monitor"
        risk_description = (
            "Recent growth should be validated before increasing inventory or spend."
        )
        primary_action = (
            f"Test a focused offer around {metrics['best_product']}; scale only "
            "if the next comparable period sustains the improvement."
        )
    elif momentum < 0:
        outlook = f"Under pressure · {momentum:+.1f}%"
        growth = "Validate"
        risk = "Elevated"
        risk_description = (
            f"Revenue momentum is {momentum:+.1f}% versus the preceding period."
        )
        primary_action = (
            f"Review recent performance for {metrics['best_product']} and test "
            "one targeted response before committing additional spend."
        )
    else:
        outlook = "Stable"
        growth = "Emerging"
        risk = "Monitor"
        risk_description = (
            "Revenue is broadly level; cost, margin, and customer signals are not "
            "available in this dataset."
        )
        primary_action = (
            f"Pilot a focused offer around {metrics['best_product']} and define "
            "a measurable threshold before expanding it."
        )

    forecast_value = metrics["forecast_value"]
    revenue_outlook = (
        f"${float(forecast_value):,.2f}/day forecast"
        if forecast_value is not None
        else outlook
    )
    return {
        "health_score": score,
        "outlook": outlook,
        "growth": growth,
        "risk": risk,
        "risk_description": risk_description,
        "revenue_outlook": revenue_outlook,
        "top_opportunity": (
            f"Use {metrics['best_product']} as a focused test anchor; validate "
            "incremental demand before scaling."
        ),
        "primary_action": primary_action,
    }


def _default_priorities(metrics: dict[str, Any]) -> dict[str, dict[str, str]]:
    """Create actionable fallback cards from observed sales signals and limits."""
    product = str(metrics["best_product"])
    momentum = metrics["momentum_percent"]
    if momentum is not None and momentum < 0:
        revenue_action = (
            f"Review the latest sales for {product}, identify the largest change, "
            "and test one focused corrective offer."
        )
        revenue_impact = "Respond to the observed decline before broadening spend."
    else:
        revenue_action = (
            f"Pilot a focused offer around {product} and compare revenue with the "
            "current comparable-period baseline."
        )
        revenue_impact = "Test whether the leading product can support measurable revenue growth."
    return {
        "Increase Revenue": {
            "impact": revenue_impact,
            "action": revenue_action,
            "timeline": "Month 1 · Establish the baseline and launch a bounded test.",
        },
        "Product Expansion": {
            "impact": "A controlled product test can reveal demand without assuming market fit.",
            "action": (
                f"Select one complementary offer to {product}; measure its "
                "contribution before expanding the range."
            ),
            "timeline": "Month 2 · Run a limited product or offer pilot.",
        },
        "Improve Customer Retention": {
            "impact": "Customer-level evidence is needed to distinguish repeat demand from one-off sales.",
            "action": (
                "Begin tracking repeat purchases with a consistent customer "
                "identifier, then test one retention intervention."
            ),
            "timeline": "Month 2 · Establish tracking and test one intervention.",
        },
        "Reduce Costs": {
            "impact": "Cost and margin visibility is needed before judging which sales drive contribution.",
            "action": (
                "Capture product-level costs and review contribution alongside "
                "revenue before changing pricing or spend."
            ),
            "timeline": "Month 1 · Build the cost baseline for review.",
        },
        "Market Expansion": {
            "impact": "No market-size or regional evidence is available to support a scale decision.",
            "action": (
                "Choose one candidate segment, validate demand through a small "
                "test, and compare results with the current baseline."
            ),
            "timeline": "Month 3 · Assess pilot evidence before expansion.",
        },
    }


def _recommendation_details(
    title: str,
    report: dict[str, str],
    fallback: dict[str, dict[str, str]],
) -> dict[str, str]:
    """Separate generated objective and recommendations for card presentation."""
    body = report.get(title, "")
    objective: str | None = None
    action_lines: list[str] = []
    for line in body.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        objective_match = re.match(
            r"^\**\s*Strategic objective:\s*\**\s*(.*)$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if objective_match:
            objective = objective_match.group(1).strip()
        else:
            action_lines.append(cleaned)
    return {
        "impact": objective or fallback[title]["impact"],
        "action": "\n".join(action_lines) or fallback[title]["action"],
        "timeline": fallback[title]["timeline"],
    }


def _render_priority_card(
    title: str,
    details: dict[str, str],
    priority_number: int,
) -> None:
    with st.container(border=True):
        st.markdown(f"**Priority {priority_number:02d} · {title}**")
        st.markdown("**Business impact**")
        st.write(details["impact"])
        st.markdown("**Recommended action**")
        st.markdown(details["action"])
        st.markdown("**Timeline**")
        st.caption(details["timeline"])


context, metrics = _build_business_context()
brief = _brief_values(metrics)
fallback_priorities = _default_priorities(metrics)

st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] { padding-top: 1.25rem; padding-bottom: 2rem; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: rgba(139, 164, 195, .22);
        border-radius: 16px;
        background: linear-gradient(145deg, rgba(20, 34, 54, .88), rgba(14, 26, 43, .92));
        box-shadow: 0 14px 36px rgba(0, 0, 0, .16);
    }
    .strategy-hero {
        padding: 1.4rem 1.75rem;
        margin: .25rem 0 1rem;
        border: 1px solid rgba(100, 215, 194, .28);
        border-radius: 22px;
        background: radial-gradient(ellipse at 86% 8%, rgba(65, 208, 188, .16), transparent 38%),
            linear-gradient(115deg, #172b45 0%, #102235 58%, #10353a 100%);
        box-shadow: 0 22px 55px rgba(0, 0, 0, .22);
    }
    .strategy-eyebrow {
        color: #65d8c6;
        font-size: .75rem;
        font-weight: 750;
        letter-spacing: .14em;
        text-transform: uppercase;
    }
    .strategy-hero h2 {
        margin: .45rem 0 .35rem;
        color: #f3f7fc;
        font-size: clamp(1.75rem, 3vw, 2.5rem);
        letter-spacing: -.04em;
    }
    .strategy-hero p { color: #b7c6d8; margin: 0; }
    .strategy-section { margin: 1.4rem 0 .6rem; }
    .strategy-section h2 { color: #f0f5fb; font-size: 1.45rem; letter-spacing: -.025em; }
    @media (max-width: 700px) {
        .strategy-hero { padding: 1.1rem; border-radius: 17px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.markdown(
        '<div class="strategy-hero">'
        '<div class="strategy-eyebrow">Executive strategy briefing</div>'
        '<h2>Strategic Executive Brief</h2>'
        '<p>A focused view of business trajectory, opportunities, and the next '
        "practical move.</p></div>",
        unsafe_allow_html=True,
    )
    score_col, outlook_col, opportunity_col, risk_col, action_col = st.columns(
        [0.85, 1.1, 1.2, 1.15, 1.7], vertical_alignment="top"
    )
    score_col.metric(
        "💯 Business health",
        f'{brief["health_score"]}/100',
        border=True,
    )
    outlook_col.metric("📈 Business outlook", brief["outlook"], border=True)
    with opportunity_col:
        st.markdown("**Top opportunity**")
        st.write(brief["top_opportunity"])
    with risk_col:
        st.markdown("**Top risk**")
        st.write(brief["risk_description"])
    with action_col:
        st.markdown("**Primary recommendation**")
        st.write(brief["primary_action"])
st.caption(
    "Health score is a trend index: 50 is neutral, with the latest comparable "
    "14-day revenue change applied and the result bounded from 0 to 100. "
    "It is not a profitability measure."
)

st.markdown('<div class="strategy-section"><h2>Strategic scorecards</h2></div>', unsafe_allow_html=True)
scorecard_values = (
    ("🟢 Growth potential", brief["growth"]),
    ("🟡 Market opportunity", "Pilot to validate"),
    ("🔴 Risk level", brief["risk"]),
)
if metrics["forecast_value"] is not None:
    scorecard_values += (("📈 Revenue outlook", brief["revenue_outlook"]),)
with st.container(horizontal=True):
    for label, value in scorecard_values:
        st.metric(label, value, border=True)
if metrics["forecast_value"] is None:
    render_forecast_prompt_card()

st.markdown(
    '<div class="strategy-section"><h2>Strategic priorities</h2></div>',
    unsafe_allow_html=True,
)

if st.button(
    "Refresh AI recommendations" if st.session_state["strategy_report"] else "Generate detailed recommendations",
    type="primary",
    icon=":material/auto_awesome:",
):
    st.session_state["strategy_report"] = None
    try:
        with st.spinner("Developing your strategic advisory plan..."):
            strategy = generate_strategy_recommendations(context)
    except APIError as error:
        st.error(f"Groq could not generate recommendations: {error}")
    else:
        missing_areas = [title for title, _ in STRATEGY_AREAS if title not in strategy]
        if missing_areas:
            st.error(
                "The advisor response was incomplete. Missing sections: "
                + ", ".join(missing_areas)
                + ". Please generate the recommendations again."
            )
        else:
            st.session_state["strategy_report"] = strategy
            st.success("Strategic advisory plan generated.")

strategy_report = st.session_state.get("strategy_report")
has_strategy = isinstance(strategy_report, dict) and bool(strategy_report)
priority_details: dict[str, dict[str, str]] = {}
for title, _ in STRATEGY_AREAS:
    if has_strategy:
        priority_details[title] = _recommendation_details(
            title,
            strategy_report,
            fallback_priorities,
        )
    else:
        priority_details[title] = fallback_priorities[title]

for start in range(0, len(STRATEGY_AREAS), 2):
    columns = st.columns(2)
    for index, (title, _) in enumerate(STRATEGY_AREAS[start : start + 2], start + 1):
        with columns[index - start - 1]:
            _render_priority_card(title, priority_details[title], index)
if len(STRATEGY_AREAS) % 2:
    title, _ = STRATEGY_AREAS[-1]
    with st.container(border=True):
        st.markdown(f"**Priority {len(STRATEGY_AREAS):02d} · {title}**")
        st.markdown("**Business impact**")
        st.write(priority_details[title]["impact"])
        st.markdown("**Recommended action**")
        st.markdown(priority_details[title]["action"])
        st.markdown("**Timeline**")
        st.caption(priority_details[title]["timeline"])

st.markdown(
    '<div class="strategy-section"><h2>90-day strategic roadmap</h2></div>',
    unsafe_allow_html=True,
)
roadmap = (
    ("Month 1", ("Increase Revenue", "Reduce Costs")),
    ("Month 2", ("Product Expansion", "Improve Customer Retention")),
    ("Month 3", ("Market Expansion",)),
)
roadmap_columns = st.columns(3)
for column, (month, titles) in zip(roadmap_columns, roadmap):
    with column:
        with st.container(border=True):
            st.subheader(month)
            for title in titles:
                st.markdown(f"**{title}**")
                st.markdown(priority_details[title]["action"])

with st.expander("Evidence base and interpretation"):
    st.markdown(
        "This brief uses recorded revenue by product, daily sales history, "
        "order totals, average order value, and any stored forecast. The uploaded "
        "data does not include margins, customer identity, or market sizing, so "
        "related recommendations are framed as tests rather than established facts."
    )
    if metrics["forecast_value"] is None:
        st.caption(
            "A forecast is not included in this view; the outlook is based on "
            "historical sales momentum."
        )
