"""Groq-powered business analysis page."""

import streamlit as st

from utils.shared import (
    GroqCompletionError,
    apply_compact_page_spacing,
    generate_groq_text,
    require_dashboard_data,
    render_forecast_prompt_card,
)

st.set_page_config(
    page_title="AI Consultant | AI Business Copilot",
    page_icon=":material/psychology:",
    layout="wide",
)

apply_compact_page_spacing()

if not require_dashboard_data():
    st.stop()

sales = st.session_state["df"]
product_sales = st.session_state["product_sales"]
daily_sales = st.session_state["daily_sales"]

business_context = f"""
Business metrics:
- Total revenue: ${st.session_state['total_revenue']:,.2f}
- Total orders: {st.session_state['total_orders']:,}
- Best product: {st.session_state['best_product']}
- Average order value: ${st.session_state['average_order_value']:,.2f}

Revenue by product:
{product_sales.to_string(index=False)}

Daily revenue:
{daily_sales.to_string(index=False)}

Date range: {sales['Date'].min().date()} through {sales['Date'].max().date()}
"""

consultation_actions = {
    "Analyze business": (
        "Analyze business performance. Give a concise summary, key findings, "
        "and practical recommendations."
    ),
    "Identify risks": (
        "Identify the most important business risks and weak areas. Explain "
        "their likely impact and suggest mitigations."
    ),
    "Growth opportunities": (
        "Identify the three strongest growth opportunities, why they matter, "
        "and a prioritized action plan."
    ),
    "Revenue recommendations": (
        "Recommend practical ways to increase revenue and average order value, "
        "prioritized by impact and effort."
    ),
    "Executive summary": (
        "Write an executive summary with performance, key findings, risks, "
        "opportunities, and prioritized recommendations."
    ),
}


def request_analysis(instructions: str) -> str:
    """Send a business-context prompt to the configured Groq chat model."""
    return generate_groq_text(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a pragmatic senior business analyst. Use the "
                    "provided data, distinguish observations from assumptions, "
                    "and do not invent missing facts."
                ),
            },
            {
                "role": "user",
                "content": f"{business_context}\n\nTask:\n{instructions}",
            },
        ],
    )


st.markdown(
    """
    <style>
    .stApp { background: #0b1422; color: #eaf0f8; }
    [data-testid="stHeader"] { background: rgba(11,20,34,.9); }
    [data-testid="stMainBlockContainer"] { padding-top: 1.25rem; }
    .consult-hero {
        background: radial-gradient(ellipse at 88% 14%, rgba(66,214,197,.15), transparent 38%),
            linear-gradient(115deg, #172941 0%, #102033 58%, #12343d 100%);
        border: 1px solid rgba(120,169,255,.22); border-radius: 23px;
        padding: 1.45rem 1.8rem; margin: .15rem 0 1rem;
        box-shadow: 0 20px 52px rgba(0,0,0,.24);
    }
    .consult-eyebrow { color: #42d6c5; font-size: .74rem; font-weight: 750;
        letter-spacing: .15em; text-transform: uppercase; }
    .consult-hero h1 { color: #f5f8fc; font-size: clamp(1.9rem, 3vw, 2.6rem);
        margin: .38rem 0 .3rem; letter-spacing: -.04em; }
    .consult-hero p { color: #aebdd0; margin: 0; font-size: 1rem; }
    .consult-section { margin: 1.4rem 0 .6rem; padding-bottom: .45rem;
        border-bottom: 1px solid rgba(154,170,192,.18); }
    .consult-section h2 { color: #f2f6fb; font-size: 1.42rem;
        font-weight: 730; letter-spacing: -.025em; margin: 0 0 .25rem; }
    .consult-section p { color: #9aaac0; font-size: .9rem; margin: 0; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: linear-gradient(150deg, rgba(22,36,58,.96), rgba(17,29,47,.96));
        border-color: rgba(154,170,192,.17) !important;
        border-radius: 17px !important; box-shadow: 0 12px 30px rgba(0,0,0,.14);
        transition: transform .18s ease, border-color .18s ease;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        border-color: rgba(66,214,197,.36) !important; transform: translateY(-2px);
    }
    [data-testid="stMetricLabel"] { color: #b9c8da !important; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #f5f8fc !important;
        font-size: clamp(1.45rem, 2.3vw, 2.05rem) !important;
        font-weight: 740 !important; letter-spacing: -.04em; }
    .consult-tag { color: #66ddcf; font-size: .72rem; font-weight: 750;
        letter-spacing: .12em; text-transform: uppercase; }
    .consult-label { color: #62d9cb; font-size: .73rem; font-weight: 750;
        letter-spacing: .12em; text-transform: uppercase; margin-top: .65rem; }
    .milestone-number { color: #42d6c5; font-size: .74rem; font-weight: 800;
        letter-spacing: .12em; text-transform: uppercase; }
    div.stButton > button[kind="primary"] { background: #138f87; border: 0;
        border-radius: 10px; color: white; font-weight: 700; min-height: 2.65rem; }
    div.stButton > button[kind="primary"]:hover { background: #17a79c; }
    [data-testid="stTextArea"] textarea { background: #111e30; color: #edf3fa;
        border-color: rgba(154,170,192,.25); border-radius: 12px; }
    @media (max-width: 700px) {
        .consult-hero { padding: 1.1rem; border-radius: 17px; }
        .consult-section { margin-top: 1.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="consult-hero">
      <div class="consult-eyebrow">Executive intelligence · AI consulting</div>
      <h1>AI Business Consultant</h1>
      <p>Translate your sales performance into focused decisions and measurable action.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

revenue = float(st.session_state["total_revenue"])
orders = int(st.session_state["total_orders"])
average_order_value = float(st.session_state["average_order_value"])
top_product = str(st.session_state["best_product"])
product_revenue = product_sales.sort_values("Revenue", ascending=False)
top_product_revenue = float(product_revenue.iloc[0]["Revenue"])
top_product_share = top_product_revenue / revenue if revenue > 0 else 0.0
momentum_change = None
if len(daily_sales) >= 4:
    period_size = min(14, len(daily_sales) // 2)
    previous_revenue = float(
        daily_sales.iloc[-2 * period_size : -period_size]["Revenue"].sum()
    )
    recent_revenue = float(daily_sales.iloc[-period_size:]["Revenue"].sum())
    if previous_revenue > 0:
        momentum_change = (recent_revenue - previous_revenue) / previous_revenue * 100

st.markdown(
    '<div class="consult-section"><h2>Executive Brief</h2>'
    '<p>A concise view of business health and the decisions most relevant now.</p></div>',
    unsafe_allow_html=True,
)

forecast_value = st.session_state.get("future_revenue")
if momentum_change is None:
    business_health = "Baseline building"
    health_detail = "More recorded sales dates are needed to confirm a revenue trend."
elif momentum_change > 0:
    business_health = "Positive momentum"
    health_detail = f"Revenue increased {momentum_change:+.1f}% across comparable recorded-date periods."
elif momentum_change < 0:
    business_health = "Needs attention"
    health_detail = f"Revenue declined {momentum_change:+.1f}% across comparable recorded-date periods."
else:
    business_health = "Stable"
    health_detail = "Revenue was unchanged across comparable recorded-date periods."

top_opportunity = (
    f"Build on {top_product}, currently {top_product_share:.1%} of revenue."
    if top_product_share >= 0.3
    else "Test a focused offer to improve average order value and measure conversion."
)
top_risk = (
    f"Product concentration: {top_product} contributes {top_product_share:.1%} "
    "of recorded revenue."
    if top_product_share >= 0.6
    else "Profit, retention, and channel risks cannot be assessed from the available sales fields."
)
key_recommendation = (
    f"Run a small, measurable complementary-offer test around {top_product}; "
    "review revenue and conversion before scaling."
)
expected_outcome = (
    "Establish evidence on incremental revenue and basket value while limiting "
    "the risk of a broad, untested rollout."
)
if momentum_change is not None and momentum_change < 0:
    key_recommendation = (
        "Investigate recent revenue movement by product, then test one targeted "
        "sales action against the preceding comparable period."
    )
    expected_outcome = (
        "Identify which sales areas merit attention and create a measurable "
        "baseline for recovery actions."
    )

brief_cards = (
    ("Overall Business Health", business_health, health_detail),
    ("Top Opportunity", top_product, top_opportunity),
    ("Top Risk", "Concentration" if top_product_share >= 0.6 else "Data visibility", top_risk),
    ("Key Recommendation", "Focused pilot", key_recommendation),
    (
        "Expected Business Outcome",
        "Measured learning",
        expected_outcome,
    ),
)
for start in range(0, len(brief_cards), 3):
    brief_columns = st.columns(min(3, len(brief_cards) - start))
    icons = ("🧭", "🚀", "⚠️", "🎯", "✅")
    for column, (label, value, detail, icon) in zip(
        brief_columns,
        (
            (*card, icons[index])
            for index, card in enumerate(brief_cards[start : start + 3], start)
        ),
    ):
        with column:
            st.metric(f"{icon} {label}", value, border=True)
            st.caption(detail)

st.markdown(
    '<div class="consult-section"><h2>Strengths, Opportunities &amp; Risks</h2>'
    '<p>Signals grounded in recorded revenue, product mix, and forecast availability.</p></div>',
    unsafe_allow_html=True,
)

strengths = [
    f"Revenue totals ${revenue:,.2f} across {orders:,} recorded transactions.",
    f"{top_product} is the current leading product by revenue.",
]
opportunities = [
    top_opportunity,
]
if forecast_value is not None:
    opportunities.append("Use the forecast as a planning input.")
risks = [
    top_risk,
]
if forecast_value is not None:
    risks.append("Forecast values are estimates, not guaranteed outcomes.")

signal_columns = st.columns(3)
for column, icon, title, entries in zip(
    signal_columns,
    ("🟢", "🟡", "🔴"),
    ("Strengths", "Opportunities", "Risks"),
    (strengths, opportunities, risks),
):
    with column:
        with st.container(border=True):
            st.markdown(f"### {icon} {title}")
            for entry in entries:
                st.markdown(f"- {entry}")

if forecast_value is None:
    render_forecast_prompt_card()

st.markdown(
    '<div class="consult-section"><h2>90-Day Roadmap</h2>'
    '<p>A practical sequence that turns analysis into measurable progress.</p></div>',
    unsafe_allow_html=True,
)
roadmap = (
    (
        "Phase 1 · Days 0–30",
        "Establish the baseline",
        "Review recent revenue by product, confirm data quality, and agree on "
        "one success measure for the next commercial test.",
    ),
    (
        "Phase 2 · Days 31–60",
        "Run a focused pilot",
        f"Test a complementary offer around {top_product}; monitor revenue, "
        "average order value, and conversion against the baseline.",
    ),
    (
        "Phase 3 · Days 61–90",
        "Evaluate and scale selectively",
        "Compare pilot results with the baseline, document learnings, and "
        "continue or expand only where measured outcomes support it.",
    ),
)
phase_columns = st.columns(3)
for column, (phase, title, detail) in zip(phase_columns, roadmap):
    with column:
        with st.container(border=True):
            st.markdown(f'<div class="milestone-number">{phase}</div>', unsafe_allow_html=True)
            st.markdown(f"### {title}")
            st.markdown(detail)
            st.caption("Success measure: compare outcome with the agreed baseline.")

st.markdown(
    '<div class="consult-section"><h2>AI Advisory Playbooks</h2>'
    '<p>Choose a consulting brief to generate an analysis grounded in your sales data.</p></div>',
    unsafe_allow_html=True,
)

playbook_descriptions = {
    "Analyze business": (
        "A leadership-ready review of performance, key findings, and practical actions.",
        "Clarify which observed patterns should shape near-term business priorities.",
        "Review the strongest and weakest signals in the recorded sales history.",
        "A concise set of findings and next steps to validate against operating results.",
    ),
    "Identify risks": (
        "A structured review of potential business vulnerabilities and weak signals.",
        "Prioritize areas that may need monitoring or additional data before committing resources.",
        "Assess product concentration, revenue movement, and gaps in the available data.",
        "A risk register with mitigations that can be tested and tracked.",
    ),
    "Growth opportunities": (
        "A prioritized scan for evidence-led opportunities in revenue and product mix.",
        "Focus experiments on measurable potential rather than unsupported market assumptions.",
        "Identify promising product or order-value tests and rank them for validation.",
        "A ranked opportunity list and test plan with measurable indicators.",
    ),
    "Revenue recommendations": (
        "Practical options for increasing revenue and average order value.",
        "Identify specific sales actions while preserving visibility into conversion trade-offs.",
        "Select one focused revenue or basket-value experiment and define how to measure it.",
        "An actionable revenue test plan evaluated against the current baseline.",
    ),
    "Executive summary": (
        "A concise briefing for leadership covering performance, risks, and priorities.",
        "Make the business position and most important decisions easy to scan.",
        "Synthesize the current sales position into clear priorities for leadership.",
        "A clear, shareable view of findings and prioritized next actions.",
    ),
}

playbook_columns = st.columns(2)
for index, (label, instructions) in enumerate(consultation_actions.items()):
    column = playbook_columns[index % 2]
    description, why_it_matters, recommended_action, outcome = playbook_descriptions[
        label
    ]
    with column:
        with st.container(border=True):
            st.markdown(f"### {label}")
            st.caption(description)
            st.markdown('<div class="consult-label">Why It Matters</div>', unsafe_allow_html=True)
            st.markdown(why_it_matters)
            st.markdown('<div class="consult-label">Recommended Action</div>', unsafe_allow_html=True)
            st.markdown(recommended_action)
            st.markdown('<div class="consult-label">Expected Outcome</div>', unsafe_allow_html=True)
            st.markdown(outcome)
            if st.button(
                f"Generate {label.lower()}",
                key=f"consult_{index}",
                type="primary",
                width="stretch",
            ):
                try:
                    with st.spinner(f"{label}..."):
                        recommendation = request_analysis(instructions)
                except GroqCompletionError as error:
                    st.error(f"Groq could not generate {label.lower()}: {error}")
                else:
                    with st.container(border=True):
                        st.markdown(f"#### {label} · Consultant Recommendation")
                        st.markdown(
                            '<div class="consult-label">Why It Matters</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(why_it_matters)
                        st.markdown(
                            '<div class="consult-label">Recommended Action</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(recommendation)
                        st.markdown(
                            '<div class="consult-label">Expected Outcome</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(outcome)

st.markdown(
    '<div class="consult-section"><h2>Ask Your Data</h2>'
    '<p>Ask a specific business question and receive a response using the '
    "current sales context.</p></div>",
    unsafe_allow_html=True,
)
with st.container(border=True):
    with st.form("ask_business_data"):
        question = st.text_area(
            "Question",
            placeholder="For example: Which product should I prioritize and why?",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question before submitting.")
        else:
            try:
                with st.spinner("Analyzing your question..."):
                    answer = request_analysis(question.strip())
            except GroqCompletionError as error:
                st.error(f"Groq could not answer the question: {error}")
            else:
                st.markdown(answer)

with st.expander("Business data used by the consultant"):
    st.markdown(business_context)
    st.caption(
        "The uploaded data contains no cost, customer identity, repeat-purchase, "
        "marketing-channel, inventory, competitor, or market-size details. "
        "Recommendations in these areas should be treated as measurable tests."
    )
