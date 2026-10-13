"""Conversational, data-grounded business analysis using Groq."""

import pandas as pd
import re
import streamlit as st
from groq import APIError

from utils.shared import (
    apply_compact_page_spacing,
    get_groq_client,
    get_groq_model,
    require_dashboard_data,
)

st.set_page_config(
    page_title="Ask Your Data | AI Business Copilot",
    page_icon=":material/chat:",
    layout="wide",
)
apply_compact_page_spacing()
st.title("Ask your data")
st.caption("Your sales data, translated into clear answers for better decisions.")

if not require_dashboard_data():
    st.stop()

st.session_state.setdefault("ask_data_messages", [])
st.session_state.setdefault("pending_ask_question", None)

sales = st.session_state.get("df")
if not isinstance(sales, pd.DataFrame) or sales.empty:
    st.warning("Upload a non-empty sales dataset on the Dashboard first.")
    st.stop()

STARTER_QUESTIONS = (
    "Which product generated the highest revenue?",
    "Which month performed best?",
    "What trends should management focus on?",
    "What business risks are visible in this dataset?",
)

SECTION_ALIASES = {
    "Executive Summary": {"executive summary", "summary"},
    "Evidence": {"evidence"},
    "Insights": {"insights"},
    "Recommendations": {"recommendations"},
    "Detailed Analysis": {"detailed analysis", "analysis"},
}


def _business_context(data: pd.DataFrame) -> str:
    """Create a compact set of reproducible statistics from the uploaded rows."""
    work = data.copy()
    work["Date"] = pd.to_datetime(work["Date"], errors="coerce")
    work["Revenue"] = pd.to_numeric(work["Revenue"], errors="coerce")
    work = work.replace([float("inf"), float("-inf")], pd.NA).dropna(
        subset=["Date", "Revenue"]
    )
    if work.empty:
        return "No valid dated revenue records are available."

    total_revenue = float(work["Revenue"].sum())
    orders = len(work)
    average_order = total_revenue / orders if orders else 0.0
    product_table = (
        work.groupby("Product", dropna=False)["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(12)
        .rename("Revenue")
    )
    daily = work.groupby("Date")["Revenue"].sum().sort_index()
    monthly = work.assign(Month=work["Date"].dt.to_period("M").astype(str)).groupby(
        "Month"
    )["Revenue"].sum()
    momentum = "Insufficient history to compare two periods."
    if len(daily) >= 4:
        period_size = min(14, len(daily) // 2)
        previous = float(daily.iloc[-2 * period_size : -period_size].sum())
        recent = float(daily.iloc[-period_size:].sum())
        if previous > 0:
            momentum = (
                f"{(recent - previous) / previous:+.1%} across the latest "
                f"{period_size} recorded dates versus the preceding {period_size}."
            )

    quantity_context = "Quantity data is unavailable."
    if "Quantity" in work:
        quantity = pd.to_numeric(work["Quantity"], errors="coerce").dropna()
        if not quantity.empty:
            quantity_context = f"Total quantity sold: {float(quantity.sum()):,.2f}"

    forecast = st.session_state.get("forecast_data")
    if isinstance(forecast, pd.DataFrame) and not forecast.empty and "yhat" in forecast:
        forecast_average = pd.to_numeric(forecast["yhat"], errors="coerce").dropna().mean()
        forecast_context = (
            f"Prophet forecast average daily revenue: {float(forecast_average):,.2f}"
            if pd.notna(forecast_average)
            else "No usable Prophet forecast."
        )
    else:
        forecast_context = "No forecast is available; do not infer future revenue."

    return f"""SALES DATA SUMMARY (same revenue units as source; currency unknown):
- Transaction rows: {orders:,}
- Recorded revenue: {total_revenue:,.2f}
- Average revenue per row (not necessarily a unique-customer order): {average_order:,.2f}
- Date range: {work["Date"].min():%Y-%m-%d} through {work["Date"].max():%Y-%m-%d}
- Revenue momentum: {momentum}
- {quantity_context}
- Highest revenue product: {product_table.index[0]} ({product_table.iloc[0]:,.2f})
- Monthly revenue table:
{monthly.tail(24).to_string()}
- Product revenue (top 12):
{product_table.to_string()}
- Forecast: {forecast_context}

LIMITATIONS:
The source contains sales transaction rows. It does not establish causes of a
revenue change, margins, customer retention, inventory, marketing exposure, or
competitor activity unless such fields are explicitly present. Clearly separate
calculated facts from hypotheses and say when more data is needed.
"""


def _ask(question: str) -> str:
    """Send a bounded sales context and recent conversation to Groq."""
    context = _business_context(sales)
    recent = st.session_state["ask_data_messages"][-10:]
    # The submitted question was saved before this call; do not send it twice.
    if recent and recent[-1].get("role") == "user" and recent[-1].get("content") == question:
        recent = recent[:-1]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a business intelligence copilot. Answer only from the "
                "provided dataset summary and the conversation. Never invent "
                "causes or business facts. Use concise, professional Markdown "
                "with exactly these headings: Executive Summary, Evidence, "
                "Detailed Analysis, Recommendations. Keep Executive Summary to "
                "one or two direct sentences. Put supporting figures under "
                "Evidence and interpretation under Detailed Analysis. If the "
                "question cannot be answered, state which additional data is needed."
            ),
        },
        {"role": "user", "content": f"BUSINESS CONTEXT:\n{context}"},
    ]
    messages.extend(
        {"role": item["role"], "content": item["content"]}
        for item in recent
        if item.get("role") in {"user", "assistant"}
    )
    messages.append({"role": "user", "content": question})
    response = get_groq_client().chat.completions.create(
        model=get_groq_model(),
        temperature=0.25,
        messages=messages,
    )
    answer = response.choices[0].message.content
    if not answer or not answer.strip():
        raise ValueError("Groq returned an empty answer.")
    return answer.strip()


def _section_content(answer: str) -> dict[str, str]:
    """Split a Markdown response into display sections without losing content."""
    sections: dict[str, list[str]] = {}
    active: str | None = None
    for line in answer.splitlines():
        heading_match = re.match(r"^\s{0,3}#{1,4}\s+(.+?)\s*#*\s*$", line)
        bold_match = re.match(r"^\s*\*\*(.+?)\*\*\s*:?\s*$", line)
        heading = (heading_match or bold_match)
        if heading:
            label = heading.group(1).strip().rstrip(":").casefold()
            active = next(
                (
                    section
                    for section, aliases in SECTION_ALIASES.items()
                    if label in aliases
                ),
                "Detailed Analysis",
            )
            sections.setdefault(active, [])
            continue
        if active is not None:
            sections.setdefault(active, []).append(line)

    parsed = {
        title: "\n".join(lines).strip()
        for title, lines in sections.items()
        if "\n".join(lines).strip()
    }

    # Keep older responses readable: if no recognized headings are present,
    # show their original Markdown as the visible summary rather than hiding it.
    if not parsed:
        return {"Executive Summary": answer.strip()}

    if not parsed.get("Executive Summary"):
        first_section = next(iter(parsed))
        parsed["Executive Summary"] = parsed.pop(first_section)
    return parsed


def _render_assistant_answer(answer: str) -> None:
    """Show the executive takeaway first and progressively disclose details."""
    sections = _section_content(answer)
    summary = sections.get("Executive Summary", "")
    st.markdown(summary or "I could not find an executive summary in this response.")

    evidence = sections.get("Evidence")
    if evidence:
        with st.expander("Evidence from your data"):
            st.markdown(evidence)

    detail_sections = [
        (title, body)
        for title, body in sections.items()
        if title not in {"Executive Summary", "Evidence"} and body.strip()
    ]
    if detail_sections:
        with st.expander("Detailed analysis and recommendations"):
            for title, body in detail_sections:
                st.markdown(f"**{title}**")
                st.markdown(body)


history = st.session_state["ask_data_messages"]
clear_column, history_column = st.columns([1, 4], vertical_alignment="center")
with clear_column:
    if st.button(
        "Clear chat",
        icon=":material/delete_sweep:",
        key="clear_ask_data_chat",
    ):
        st.session_state["ask_data_messages"] = []
        st.session_state["pending_ask_question"] = None
        st.rerun()
with history_column:
    if history:
        st.caption(
            f"{sum(message.get('role') == 'user' for message in history)} "
            "question(s) in this conversation"
        )

if not history:
    with st.container(border=True):
        st.markdown("### What would you like to understand?")
        st.caption(
            "Ask a follow-up at any time. Answers use the current uploaded sales "
            "data and clearly distinguish evidence from interpretation."
        )
        if hasattr(st, "pills"):
            selected_question = st.pills(
                "Suggested questions",
                list(STARTER_QUESTIONS),
                key="ask_data_starters",
                label_visibility="collapsed",
            )
            if selected_question:
                st.session_state["pending_ask_question"] = selected_question
                st.rerun()
        else:
            question_columns = st.columns(2)
            for index, starter_question in enumerate(STARTER_QUESTIONS):
                with question_columns[index % 2]:
                    if st.button(
                        starter_question,
                        key=f"starter_question_{index}",
                        width="stretch",
                    ):
                        st.session_state["pending_ask_question"] = starter_question
                        st.rerun()

for message in st.session_state["ask_data_messages"]:
    role = message.get("role")
    content = message.get("content", "")
    if role not in {"user", "assistant"} or not isinstance(content, str):
        continue
    avatar = ":material/person:" if role == "user" else ":material/auto_awesome:"
    with st.chat_message(role, avatar=avatar):
        if role == "assistant":
            _render_assistant_answer(content)
        else:
            st.markdown(content)

question = st.chat_input("Ask a question about your business data")
if question is None:
    question = st.session_state.get("pending_ask_question")
st.session_state["pending_ask_question"] = None

if question:
    user_question = str(question).strip()
    if user_question:
        st.session_state["ask_data_messages"].append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user", avatar=":material/person:"):
            st.markdown(user_question)
        with st.chat_message("assistant", avatar=":material/auto_awesome:"):
            try:
                with st.spinner("Analyzing your sales data..."):
                    answer = _ask(user_question)
                _render_assistant_answer(answer)
            except APIError as error:
                answer = (
                    "I couldn't reach the AI service for this question. "
                    "Please check your Groq configuration and try again."
                )
                st.error(f"{answer} Details: {error}")
            except ValueError as error:
                answer = str(error)
                st.error(answer)
        st.session_state["ask_data_messages"].append(
            {"role": "assistant", "content": answer}
        )
