"""Executive dashboard presentation and report helpers."""

from datetime import datetime
from html import escape
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
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
    get_groq_client,
    get_groq_model,
    render_forecast_prompt_card,
)

PANEL = "#16243A"
TEAL = "#42D6C5"
BLUE = "#78A9FF"
GOLD = "#F6C667"
RED = "#FF7777"
CHART_COLORS = [TEAL, BLUE, GOLD, "#A78BFA", "#F28B82", "#7DD3FC", "#F9A8D4"]
CONSULTANT_SECTIONS = (
    "## Executive Summary",
    "## Key Insights",
    "## Risks",
    "## Opportunities",
    "## Strategic Recommendations",
    "## 90-Day Action Plan",
)
REQUIRED_SALES_COLUMNS = {"Date", "Product", "Quantity", "Revenue"}


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Convert a column to numeric values without changing the source data."""
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)


def calculate_data_quality(data: pd.DataFrame) -> dict[str, Any]:
    """Summarize completeness, duplicates, numerical outliers, and source types."""
    missing_columns = sorted(REQUIRED_SALES_COLUMNS.difference(data.columns))
    expected_column_count = data.shape[1] + len(missing_columns)
    total_cells = int(data.shape[0] * expected_column_count)
    missing_by_column = data.isna().sum().to_dict()
    missing_by_column.update({column: len(data) for column in missing_columns})
    missing_cells = int(sum(missing_by_column.values()))
    completeness = (
        100.0 * (total_cells - missing_cells) / total_cells if total_cells else 0.0
    )
    numeric_columns: dict[str, int] = {}
    for column in data.columns:
        column_name = str(column).casefold()
        if (
            "date" in column_name
            or "category" in column_name
            or "region" in column_name
            or "product" in column_name
            or "customer" in column_name
            or column_name.endswith("id")
            or pd.api.types.is_datetime64_any_dtype(data[column])
        ):
            continue
        values = _clean_numeric(data[column])
        if pd.api.types.is_numeric_dtype(data[column]) or values.notna().sum() >= 3:
            values = values.dropna()
            if len(values) >= 4:
                first_quartile = float(values.quantile(0.25))
                third_quartile = float(values.quantile(0.75))
                iqr = third_quartile - first_quartile
                if iqr > 0:
                    lower_bound = first_quartile - 1.5 * iqr
                    upper_bound = third_quartile + 1.5 * iqr
                    count = int(((values < lower_bound) | (values > upper_bound)).sum())
                    if count:
                        numeric_columns[str(column)] = count

    duplicate_rows = int(data.duplicated().sum())
    if (
        missing_columns
        or completeness < 80
        or missing_cells / max(total_cells, 1) >= 0.20
    ):
        quality_status = "critical"
        quality_label = "Critical data issues"
        recommendation = (
            "Resolve substantial missingness before interpreting comparisons or "
            "using the dataset for operational decisions."
        )
    elif missing_cells or duplicate_rows or numeric_columns:
        quality_status = "warning"
        quality_label = "Review recommended"
        recommendation = (
            "Review flagged blanks, duplicate rows, and outlier values against "
            "the source system before changing or removing records."
        )
    else:
        quality_status = "good"
        quality_label = "Good data quality"
        recommendation = (
            "No missing cells, exact duplicate rows, or IQR outliers were detected "
            "in the uploaded file."
        )

    return {
        "missing_cells": missing_cells,
        "missing_columns": missing_columns,
        "duplicate_rows": duplicate_rows,
        "outliers": numeric_columns,
        "completeness": completeness,
        "status": quality_status,
        "label": quality_label,
        "recommendation": recommendation,
        "types": {
            **data.dtypes.astype(str).to_dict(),
            **{column: "Missing required column" for column in missing_columns},
        },
        "missing_by_column": missing_by_column,
        "rows": len(data),
        "columns": len(data.columns),
    }


def render_data_quality_report(data: pd.DataFrame) -> dict[str, Any]:
    """Render the upload quality summary and return its calculated metrics."""
    quality = calculate_data_quality(data)
    _section_header(
        "Data quality report",
        "Source-file diagnostics; outliers are flagged using the 1.5×IQR rule "
        "and are not automatically removed.",
    )
    status_icon = {
        "good": "✅",
        "warning": "⚠️",
        "critical": "❌",
    }[quality["status"]]
    with st.container(border=True):
        st.markdown(f"### {status_icon} {quality['label']}")
        quality_columns = st.columns(4)
        quality_columns[0].metric(
            "✅ Data completeness", f"{quality['completeness']:.1f}%", border=True
        )
        quality_columns[1].metric(
            "🕳️ Missing values", f"{quality['missing_cells']:,}", border=True
        )
        quality_columns[2].metric(
            "📑 Duplicate rows", f"{quality['duplicate_rows']:,}", border=True
        )
        quality_columns[3].metric(
            "📍 Outlier cells", f"{sum(quality['outliers'].values()):,}", border=True
        )
        st.caption(quality["recommendation"])
        if quality["missing_columns"]:
            st.error(
                "Required sales columns missing: "
                + ", ".join(quality["missing_columns"])
            )
        with st.expander("Column types and missing-value detail"):
            type_table = pd.DataFrame(
                {
                    "Column": list(quality["types"]),
                    "Source type": list(quality["types"].values()),
                    "Missing values": [
                        int(quality["missing_by_column"][column])
                        for column in quality["types"]
                    ],
                    "Null percentage": [
                        (
                            quality["missing_by_column"][column]
                            / max(quality["rows"], 1)
                            * 100
                        )
                        for column in quality["types"]
                    ],
                }
            )
            st.dataframe(type_table, hide_index=True, width="stretch")
        if quality["outliers"]:
            st.warning(
                "IQR outlier counts: "
                + ", ".join(
                    f"{column}: {count:,}"
                    for column, count in quality["outliers"].items()
                )
                + ". Confirm these values in the source before excluding them."
            )
    return quality


def _revenue_growth(daily_sales: pd.DataFrame) -> tuple[float | None, int, float, float]:
    """Compare equal groups of recent distinct recorded dates."""
    if not {"Date", "Revenue"}.issubset(daily_sales.columns):
        return None, 0, 0.0, 0.0
    daily = daily_sales.loc[:, ["Date", "Revenue"]].copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    daily["Revenue"] = _clean_numeric(daily["Revenue"])
    daily = (
        daily.dropna(subset=["Date", "Revenue"])
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")
        .reset_index(drop=True)
    )
    if len(daily) < 4:
        return None, 0, 0.0, 0.0
    period_size = min(14, len(daily) // 2)
    prior = float(daily.iloc[-2 * period_size : -period_size]["Revenue"].sum())
    recent = float(daily.iloc[-period_size:]["Revenue"].sum())
    growth = (recent - prior) / prior * 100 if prior > 0 else None
    return growth, period_size, prior, recent


def _section_header(title: str, description: str) -> None:
    st.markdown(
        f'<div class="exec-section"><h2>{escape(title)}</h2>'
        f'<p>{escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def _add_dashboard_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0b1422; color: #eaf0f8; }
        [data-testid="stHeader"] { background: rgba(11, 20, 34, .88); }
        [data-testid="stMainBlockContainer"] { padding-top: 1.25rem; }
        .exec-hero {
            background: radial-gradient(ellipse at 88% 18%, rgba(66,214,197,.14), transparent 38%),
                linear-gradient(115deg, #172941 0%, #102033 58%, #12343d 100%);
            border: 1px solid rgba(120,169,255,.22); border-radius: 24px;
            padding: 1.5rem 1.85rem; margin: .15rem 0 1rem;
            box-shadow: 0 20px 55px rgba(0,0,0,.24);
        }
        .exec-eyebrow { color: #42d6c5; font-size: .76rem; font-weight: 750;
            letter-spacing: .14em; text-transform: uppercase; }
        .exec-hero h1 { color: #f5f8fc; font-size: clamp(1.8rem, 3vw, 2.65rem);
            margin: .35rem 0 .35rem; letter-spacing: -.035em; }
        .exec-hero p { color: #aebdd0; font-size: 1rem; margin: 0; }
        .exec-section { margin: 1.45rem 0 .6rem; padding-bottom: .45rem;
            border-bottom: 1px solid rgba(154,170,192,.17); }
        .exec-section h2 { color: #f1f5fb; font-size: 1.48rem;
            font-weight: 720; margin: 0 0 .28rem; letter-spacing: -.025em; }
        .exec-section p { color: #9aaac0; font-size: .9rem; margin: 0; }
        .briefing-panel { background: linear-gradient(145deg, rgba(22,36,58,.98), rgba(16,34,49,.98));
            border: 1px solid rgba(66,214,197,.25); border-radius: 20px;
            padding: 1.3rem 1.5rem; box-shadow: 0 18px 42px rgba(0,0,0,.18); }
        .briefing-panel h3 { color: #eaf5f5; font-size: 1.02rem; margin: 0 0 .4rem; }
        .briefing-panel p, .briefing-panel li { color: #c1cddd; line-height: 1.5; }
        .briefing-panel ul { padding-left: 1.15rem; margin: .2rem 0 .7rem; }
        .briefing-tag { display: inline-block; color: #69e0d2; font-size: .72rem;
            text-transform: uppercase; letter-spacing: .12em; font-weight: 750;
            margin-bottom: .55rem; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: linear-gradient(150deg, rgba(22,36,58,.94), rgba(17,29,47,.94));
            border-color: rgba(154,170,192,.16) !important;
            border-radius: 16px !important; box-shadow: 0 10px 28px rgba(0,0,0,.12);
            transition: transform .18s ease, border-color .18s ease;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            border-color: rgba(66,214,197,.38) !important; transform: translateY(-2px);
        }
        [data-testid="stMetricLabel"] { color: #b9c8da !important;
            font-size: .88rem !important; font-weight: 600 !important; }
        [data-testid="stMetricValue"] { color: #f5f8fc !important;
            font-size: clamp(1.65rem, 2.45vw, 2.45rem) !important;
            font-weight: 750 !important; letter-spacing: -.045em; }
        .insight-label { color: #65dccd; font-size: .72rem; font-weight: 750;
            letter-spacing: .12em; text-transform: uppercase; margin-top: .8rem; }
        .quality-label { font-size: .73rem; color: #9aaac0; text-transform: uppercase;
            letter-spacing: .1em; font-weight: 700; }
        @media (max-width: 700px) {
            .exec-hero { padding: 1.1rem; border-radius: 16px; }
            .briefing-panel { padding: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _consultant_prompt(
    data: pd.DataFrame,
    raw_data: pd.DataFrame,
    insights: list[dict[str, str]],
    quality: dict[str, Any],
    growth: float | None,
) -> str:
    """Build a bounded, evidence-led prompt from the current upload."""
    product_revenue = (
        data.groupby("Product", dropna=False)["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .to_string()
    )
    revenue = float(data["Revenue"].sum())
    order_count = len(data)
    date_min = pd.to_datetime(data["Date"], errors="coerce").min()
    date_max = pd.to_datetime(data["Date"], errors="coerce").max()
    date_range = (
        f"{date_min:%Y-%m-%d} to {date_max:%Y-%m-%d}"
        if pd.notna(date_min) and pd.notna(date_max)
        else "unavailable"
    )
    growth_text = f"{growth:+.1f}%" if growth is not None else "not calculable"
    observed_insights = "\n".join(
        f"- {item['title']}: {item['metric']}. {item['detail']}" for item in insights
    )
    unavailable = [
        label
        for column, label in (
            ("Profit", "profit"),
            ("Customer ID", "unique customer count"),
            ("Category", "category"),
            ("Region", "regional performance"),
        )
        if column not in raw_data.columns
    ]
    limitation_text = ", ".join(unavailable) if unavailable else "none of these fields"
    forecast_value = st.session_state.get("future_revenue")
    forecast_text = (
        f"average daily forecast ${float(forecast_value):,.2f}"
        if forecast_value is not None
        else "not generated"
    )
    optional_facts: list[str] = []
    if "Profit" in raw_data.columns:
        profit = _clean_numeric(raw_data["Profit"])
        optional_facts.append(
            f"- Total profit supplied in source: ${profit.sum():,.2f}"
            if profit.notna().any()
            else "- Profit column supplied, but no numeric profit values are available"
        )
    if "Customer ID" in raw_data.columns:
        optional_facts.append(
            "- Unique customer IDs: "
            f"{raw_data['Customer ID'].nunique(dropna=True):,}"
        )
    for dimension in ("Category", "Region"):
        if dimension in raw_data.columns:
            dimension_data = raw_data.loc[:, [dimension, "Revenue"]].copy()
            dimension_data["Revenue"] = _clean_numeric(dimension_data["Revenue"])
            ranking = (
                dimension_data.dropna(subset=[dimension, "Revenue"])
                .groupby(dimension)["Revenue"]
                .sum()
                .sort_values(ascending=False)
                .head(5)
                .to_dict()
            )
            optional_facts.append(
                f"- Revenue by {dimension.casefold()} (leading values): {ranking}"
            )
    return f"""Prepare a concise executive consulting report using only the facts below.
Do not invent causation, profit, customers, category, geography, market benchmarks,
or forecast values. Distinguish measured observations from hypotheses and use the
status indicators exactly: 🟢 Strength, 🟡 Opportunity, 🔴 Risk.
For each recommendation give an action and measurable check. Make a practical
90-day action plan grouped into 0-30, 31-60, and 61-90 days.
Use these exact Markdown headings and bullet points:
## Executive Summary
## Key Insights
## Risks
## Opportunities
## Strategic Recommendations
## 90-Day Action Plan

Verified dataset facts:
- Revenue total: ${revenue:,.2f}
- Transaction rows: {order_count:,}
- Average transaction value: ${revenue / order_count if order_count else 0:,.2f}
- Highest-revenue product: {data.groupby("Product")["Revenue"].sum().idxmax()}
- Revenue comparison across recent recorded dates: {growth_text}
- Dataset date range: {date_range}
- Forecast status: {forecast_text}
- Data completeness: {quality["completeness"]:.1f}% across {quality["rows"]:,} rows
- Exact duplicate rows: {quality["duplicate_rows"]:,}
- Outlier counts (IQR): {quality["outliers"] or "none detected"}
- Important unavailable dimensions: {limitation_text}
{chr(10).join(optional_facts)}

Revenue by leading products:
{product_revenue}

Evidence-based dashboard observations:
{observed_insights}
"""


def _generate_consultant_report(
    data: pd.DataFrame,
    raw_data: pd.DataFrame,
    insights: list[dict[str, str]],
    quality: dict[str, Any],
    growth: float | None,
) -> str:
    response = get_groq_client().chat.completions.create(
        model=get_groq_model(),
        temperature=0.25,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a senior management consultant writing for an "
                    "executive leadership team. Be commercially precise, concise, "
                    "actionable, and transparent about data limitations."
                ),
            },
            {
                "role": "user",
                "content": _consultant_prompt(
                    data, raw_data, insights, quality, growth
                ),
            },
        ],
    )
    content = response.choices[0].message.content
    if not content or not content.strip():
        raise ValueError("The AI provider returned an empty consultant report.")
    report = content.strip()
    missing_sections = [
        heading for heading in CONSULTANT_SECTIONS if heading not in report
    ]
    if missing_sections:
        raise ValueError(
            "The AI report was incomplete. Missing sections: "
            + ", ".join(heading.removeprefix("## ") for heading in missing_sections)
            + ". Please generate it again."
        )
    return report


def _pdf_from_markdown(report: str) -> bytes:
    """Create a small multi-page, styled PDF from the consultant Markdown."""
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=0.72 * inch,
        leftMargin=0.72 * inch,
        topMargin=0.72 * inch,
        bottomMargin=0.68 * inch,
        title="AI Business Copilot Executive Report",
        author="AI Business Copilot",
    )
    base = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ExecutiveTitle",
        parent=base["Title"],
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17324D"),
        fontSize=20,
        leading=25,
        spaceAfter=10,
    )
    heading_style = ParagraphStyle(
        "ExecutiveHeading",
        parent=base["Heading2"],
        textColor=colors.HexColor("#176E70"),
        fontSize=12,
        leading=15,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "ExecutiveBody",
        parent=base["BodyText"],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#263447"),
    )
    story: list[Any] = [
        Paragraph("AI Business Copilot", title_style),
        Paragraph("Executive Consultant Report", heading_style),
        Paragraph(
            f"Generated {datetime.now().astimezone().strftime('%B %d, %Y %H:%M %Z')}",
            body_style,
        ),
        Spacer(1, 12),
    ]
    paragraph_lines: list[str] = []
    bullet_items: list[ListItem] = []

    def pdf_safe_text(text: str) -> str:
        replacements = {
            "🟢": "[STRENGTH]",
            "🟡": "[OPPORTUNITY]",
            "🔴": "[RISK]",
            "—": "--",
            "–": "-",
            "’": "'",
            "“": '"',
            "”": '"',
        }
        for original, replacement in replacements.items():
            text = text.replace(original, replacement)
        return text.encode("latin-1", errors="replace").decode("latin-1")

    def flush_paragraph() -> None:
        if paragraph_lines:
            text = escape(pdf_safe_text(" ".join(paragraph_lines)), quote=False)
            story.append(Paragraph(text, body_style))
            story.append(Spacer(1, 5))
            paragraph_lines.clear()

    def flush_bullets() -> None:
        if bullet_items:
            story.append(
                ListFlowable(
                    bullet_items.copy(),
                    bulletType="bullet",
                    leftIndent=18,
                    bulletFontName="Helvetica",
                    bulletFontSize=7,
                    spaceAfter=6,
                )
            )
            bullet_items.clear()

    for source_line in report.splitlines():
        line = source_line.strip()
        if not line:
            flush_paragraph()
            flush_bullets()
        elif line.startswith("#"):
            flush_paragraph()
            flush_bullets()
            heading = line.lstrip("#").strip()
            story.append(
                Paragraph(escape(pdf_safe_text(heading), quote=False), heading_style)
            )
        elif line.startswith(("- ", "* ")):
            flush_paragraph()
            bullet_items.append(
                ListItem(
                    Paragraph(
                        escape(pdf_safe_text(line[2:]), quote=False), body_style
                    )
                )
            )
        else:
            flush_bullets()
            paragraph_lines.append(line)
    flush_paragraph()
    flush_bullets()
    document.build(story)
    return buffer.getvalue()


def _plot_layout(figure: go.Figure, height: int = 360) -> go.Figure:
    figure.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#DCE6F3", "family": "Inter, Segoe UI, sans-serif"},
        margin={"l": 14, "r": 14, "t": 24, "b": 12},
        height=height,
        hoverlabel={"bgcolor": PANEL, "font_color": "#F5F8FC"},
        legend={"orientation": "h", "y": -0.2},
    )
    figure.update_xaxes(showgrid=False, zeroline=False)
    figure.update_yaxes(gridcolor="rgba(154,170,192,.12)", zeroline=False)
    return figure


def _render_kpis(
    data: pd.DataFrame,
    raw_data: pd.DataFrame,
    growth: float | None,
    forecast_growth: float | None,
) -> None:
    _section_header(
        "Executive KPIs",
        "The uploaded file contains sales transactions; unavailable business "
        "dimensions are not inferred.",
    )
    cards: list[tuple[str, str, str | None]] = [
        ("💰", "Total Revenue", f"${data['Revenue'].sum():,.2f}"),
        (
            "📈",
            "Revenue Growth",
            f"{growth:+.1f}%" if growth is not None else "N/A",
        ),
        (
            "🧾",
            "Average Order Value",
            f"${data['Revenue'].sum() / len(data):,.2f}" if len(data) else "N/A",
        ),
    ]
    if growth is None:
        st.caption(
            "Revenue growth requires at least four recorded dates and a nonzero baseline."
        )
    if "Profit" in raw_data.columns:
        profit_values = _clean_numeric(raw_data["Profit"])
        profit_value = float(profit_values.sum()) if profit_values.notna().any() else None
        if profit_value is not None:
            cards.append(("🪙", "Total Profit", f"${profit_value:,.2f}"))
    if "Customer ID" in raw_data.columns:
        cards.append(
            ("👥", "Customers", f"{raw_data['Customer ID'].nunique(dropna=True):,}")
        )
    cards.append(("📦", "Transactions", f"{len(data):,}"))
    top_product = (
        data.groupby("Product")["Revenue"].sum().idxmax()
        if not data.empty
        else "N/A"
    )
    cards.append(("🏆", "Top Product", str(top_product)))
    if forecast_growth is not None:
        cards.append(("🔭", "Forecast Growth", f"{forecast_growth:+.1f}%"))
    for start in range(0, len(cards), 4):
        columns = st.columns(min(4, len(cards) - start))
        for column, (icon, label, value) in zip(columns, cards[start : start + 4]):
            _render_metric_card(column, icon, label, value)


def _render_metric_card(
    column: Any,
    icon: str,
    label: str,
    value: str,
) -> None:
    column.metric(f"{icon} {label}", value, border=True)


def _executive_brief_sections(
    data: pd.DataFrame,
    insights: list[dict[str, str]],
    growth: float | None,
) -> tuple[dict[str, str], bool]:
    """Present a saved AI report when available, else an automatic data brief."""
    report = st.session_state.get("consultant_report") or st.session_state.get(
        "ai_report"
    )
    section_aliases = {
        "Executive Summary": {"executive summary"},
        "Key Insights": {"key insights", "key findings"},
        "Risks": {"risks", "business risks"},
        "Opportunities": {"opportunities", "growth opportunities"},
        "Strategic Recommendations": {
            "strategic recommendations",
            "recommended actions",
        },
        "90-Day Action Plan": {"90-day action plan", "action plan"},
    }
    sections: dict[str, str] = {}
    if isinstance(report, str) and report.strip():
        active_section: str | None = None
        content: list[str] = []
        for line in report.splitlines():
            heading = line.strip().lstrip("#").strip().casefold()
            matched = next(
                (
                    name
                    for name, aliases in section_aliases.items()
                    if heading in aliases
                ),
                None,
            )
            if matched:
                if active_section:
                    sections[active_section] = "\n".join(content).strip()
                active_section = matched
                content = []
            elif active_section:
                content.append(line)
        if active_section:
            sections[active_section] = "\n".join(content).strip()
        is_ai_report = bool(sections)
    else:
        is_ai_report = False

    total_revenue = float(data["Revenue"].sum())
    transaction_count = len(data)
    top_product = (
        data.groupby("Product")["Revenue"].sum().idxmax()
        if transaction_count
        else "Unavailable"
    )
    date_start = pd.to_datetime(data["Date"], errors="coerce").min()
    date_end = pd.to_datetime(data["Date"], errors="coerce").max()
    date_range = (
        f"{date_start:%b %d, %Y} to {date_end:%b %d, %Y}"
        if pd.notna(date_start) and pd.notna(date_end)
        else "the uploaded date range"
    )
    trend_text = (
        f"Recent recorded-date revenue changed {growth:+.1f}% "
        "versus the preceding comparable period."
        if growth is not None
        else "There is not enough recorded-date history to calculate a comparable trend."
    )
    fallback_sections = {
        "Executive Summary": (
            f"Recorded revenue is ${total_revenue:,.2f} across "
            f"{transaction_count:,} transactions from {date_range}. "
            f"{trend_text} {top_product} is the leading product by revenue."
        ),
        "Key Insights": "\n".join(
            f"- **{item['title']}** ({item['metric']}): {item['detail']}"
            for item in insights
        ),
        "Risks": "\n".join(
            [
                *(
                    f"- **{item['title']}:** {item['detail']}"
                    for item in insights
                    if item["tone"] == "caution"
                ),
                *(
                    [
                        "- Profitability and customer retention cannot be assessed "
                        "without cost and customer identifiers in the source data."
                    ]
                    if "Profit" not in st.session_state.get("raw_df", data).columns
                    or "Customer ID"
                    not in st.session_state.get("raw_df", data).columns
                    else []
                ),
            ]
        )
        or "- No elevated risk signal was identified in the available dashboard metrics.",
        "Opportunities": "\n".join(
            f"- **{item['title']}**: {item['detail']}"
            for item in insights
            if item["title"]
            in {
                "Product opportunity insight",
                "Average order value analysis",
                "Actionable recommendation",
            }
        ),
        "Strategic Recommendations": "\n".join(
            f"- {item['detail']}"
            for item in insights
            if item["title"] == "Actionable recommendation"
        )
        or "- Continue monitoring revenue and validate one small commercial test against a baseline.",
        "90-Day Action Plan": (
            "- **Days 0-30:** Establish a comparable revenue and product-mix baseline.\n"
            "- **Days 31-60:** Run one controlled product or basket-value test and track conversion.\n"
            "- **Days 61-90:** Compare results with baseline and scale only if the "
            "measured outcome supports it."
        ),
    }
    for section, fallback in fallback_sections.items():
        if not sections.get(section):
            sections[section] = fallback
    return sections, is_ai_report


def render_executive_dashboard(
    data: pd.DataFrame,
    daily_sales: pd.DataFrame,
    product_sales: pd.DataFrame,
    insights: list[dict[str, str]],
) -> None:
    """Render the executive analytics view for an uploaded business dataset."""
    _add_dashboard_styles()
    raw_data = st.session_state.get("raw_df", data)
    growth, _, _, _ = _revenue_growth(daily_sales)
    forecast = st.session_state.get("forecast_data")
    forecast_growth: float | None = None
    forecast_summary: dict[str, Any] | None = None
    if isinstance(forecast, pd.DataFrame) and not forecast.empty:
        forecast = forecast.copy()
        forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
        forecast["yhat"] = _clean_numeric(forecast["yhat"])
        forecast = forecast.dropna(subset=["ds", "yhat"])
        recent_daily_average = float(daily_sales.tail(14)["Revenue"].mean())
        forecast_average = float(forecast["yhat"].mean()) if not forecast.empty else 0.0
        if recent_daily_average > 0 and forecast_average >= 0:
            forecast_growth = (forecast_average / recent_daily_average - 1) * 100
        if not forecast.empty:
            forecast["Month"] = forecast["ds"].dt.strftime("%b %Y")
            monthly = forecast.groupby("Month", sort=False)["yhat"].mean()
            interval_width = None
            if {"yhat_lower", "yhat_upper"}.issubset(forecast.columns):
                interval_width = float(
                    ((forecast["yhat_upper"] - forecast["yhat_lower"]).clip(lower=0).mean())
                    / 2
                )
            forecast_summary = {
                "average": forecast_average,
                "growth": forecast_growth,
                "high_month": str(monthly.idxmax()),
                "low_month": str(monthly.idxmin()),
                "interval_half_width": interval_width,
            }

    st.markdown(
        """
        <div class="exec-hero">
          <div class="exec-eyebrow">Executive intelligence · Sales performance</div>
          <h1>AI Business Copilot</h1>
          <p>Decision-ready analysis grounded in your uploaded transactions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _section_header(
        "🤖 AI Consultant Report",
        "An executive briefing with clear priorities and a practical 90-day agenda.",
    )
    st.caption("🟢 Strength  ·  🟡 Opportunity  ·  🔴 Risk")
    briefing, has_ai_report = _executive_brief_sections(data, insights, growth)
    with st.container(border=True):
        st.markdown(
            '<div class="briefing-tag">'
            + ("AI-generated report" if has_ai_report else "Live dataset briefing")
            + "</div>",
            unsafe_allow_html=True,
        )
        for heading, section in briefing.items():
            st.markdown(f"### {heading}")
            st.markdown(section)
        if not has_ai_report:
            st.caption(
                "This briefing updates automatically from the uploaded data. "
                "It does not call the AI provider or infer unavailable business facts."
            )

    _render_kpis(data, raw_data, growth, forecast_growth)

    _section_header(
        "Visual analytics",
        "Revenue trajectory, product contribution, and practical operating patterns.",
    )
    trend_column, product_column = st.columns([1.35, 1])
    with trend_column:
        with st.container(border=True):
            st.markdown("**Revenue trend**")
            trend = px.area(
                daily_sales,
                x="Date",
                y="Revenue",
                markers=True,
                labels={"Date": "Recorded date", "Revenue": "Revenue"},
                custom_data=["Date"],
            )
            trend.update_traces(
                line={"color": TEAL, "width": 3, "shape": "spline"},
                fillcolor="rgba(66,214,197,.18)",
                marker={"size": 6, "color": TEAL},
                hovertemplate="%{x|%b %d, %Y}<br>Revenue: $%{y:,.2f}<extra></extra>",
            )
            st.plotly_chart(_plot_layout(trend), width="stretch")
    with product_column:
        with st.container(border=True):
            st.markdown("**Top products by revenue**")
            top_products = product_sales.sort_values("Revenue", ascending=True).tail(12)
            bar = px.bar(
                top_products,
                x="Revenue",
                y="Product",
                orientation="h",
                text="Revenue",
                color="Revenue",
                color_continuous_scale=[[0, "#28506D"], [1, TEAL]],
                labels={"Revenue": "Revenue", "Product": "Product"},
            )
            bar.update_traces(
                texttemplate="$%{x:,.0f}",
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>Revenue: $%{x:,.2f}<extra></extra>",
            )
            bar.update_layout(coloraxis_showscale=False)
            st.plotly_chart(_plot_layout(bar), width="stretch")

    with st.container(border=True):
        st.markdown("**Product mix**")
        product_mix = px.treemap(
            product_sales,
            path=["Product"],
            values="Revenue",
            color="Revenue",
            color_continuous_scale=[[0, "#21475E"], [0.48, "#287C83"], [1, TEAL]],
            custom_data=["Revenue"],
        )
        product_mix.update_traces(
            texttemplate="<b>%{label}</b><br>$%{value:,.0f}",
            hovertemplate="%{label}<br>Revenue: $%{value:,.2f}<extra></extra>",
            marker={"line": {"color": "#122337", "width": 3}},
        )
        product_mix.update_layout(coloraxis_showscale=False)
        st.plotly_chart(_plot_layout(product_mix, 520), width="stretch")

    if "Category" in raw_data.columns:
        categories = raw_data.loc[:, ["Category", "Revenue"]].copy()
        categories["Revenue"] = _clean_numeric(categories["Revenue"])
        category_sales = (
            categories.dropna(subset=["Category", "Revenue"])
            .groupby("Category", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=False)
        )
        if not category_sales.empty:
            with st.container(border=True):
                st.markdown("**Revenue by category**")
                category_chart = px.treemap(
                    category_sales,
                    path=["Category"],
                    values="Revenue",
                    color="Revenue",
                    color_continuous_scale=[[0, "#21475E"], [1, BLUE]],
                )
                category_chart.update_traces(
                    texttemplate="<b>%{label}</b><br>$%{value:,.0f}",
                    hovertemplate="%{label}<br>$%{value:,.2f}<extra></extra>",
                )
                category_chart.update_layout(coloraxis_showscale=False)
                st.plotly_chart(_plot_layout(category_chart, 360), width="stretch")

    if "Region" in raw_data.columns:
        regional = raw_data.loc[:, ["Region", "Revenue"]].copy()
        regional["Revenue"] = _clean_numeric(regional["Revenue"])
        regional_sales = (
            regional.dropna(subset=["Region", "Revenue"])
            .groupby("Region", as_index=False)["Revenue"]
            .sum()
            .sort_values("Revenue", ascending=True)
            .tail(12)
        )
        if not regional_sales.empty:
            with st.container(border=True):
                st.markdown("**Regional performance**")
                regional_chart = px.bar(
                    regional_sales,
                    x="Revenue",
                    y="Region",
                    orientation="h",
                    text="Revenue",
                    color="Revenue",
                    color_continuous_scale=[[0, "#28506D"], [1, BLUE]],
                )
                regional_chart.update_traces(
                    texttemplate="$%{x:,.0f}",
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="%{y}<br>$%{x:,.2f}<extra></extra>",
                )
                regional_chart.update_layout(coloraxis_showscale=False)
                st.plotly_chart(_plot_layout(regional_chart, 360), width="stretch")

    daily_chart_data = daily_sales.copy()
    daily_chart_data["Date"] = pd.to_datetime(daily_chart_data["Date"], errors="coerce")
    daily_chart_data["Revenue"] = _clean_numeric(daily_chart_data["Revenue"])
    daily_chart_data = daily_chart_data.dropna(subset=["Date", "Revenue"])
    revenue_days, distribution, monthly = st.columns([1, 1, 1.2])
    with revenue_days:
        with st.container(border=True):
            st.markdown("**Top revenue days**")
            top_days = daily_chart_data.nlargest(8, "Revenue").sort_values("Revenue")
            top_day_chart = px.bar(
                top_days,
                x="Revenue",
                y="Date",
                orientation="h",
                text="Revenue",
                color="Revenue",
                color_continuous_scale=[[0, "#28506D"], [1, TEAL]],
            )
            top_day_chart.update_traces(
                texttemplate="$%{x:,.0f}",
                textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y|%b %d, %Y}<br>$%{x:,.2f}<extra></extra>",
            )
            top_day_chart.update_layout(coloraxis_showscale=False)
            top_day_chart.update_yaxes(tickformat="%b %d")
            st.plotly_chart(_plot_layout(top_day_chart, 330), width="stretch")
    with distribution:
        with st.container(border=True):
            st.markdown("**Revenue distribution**")
            distribution_chart = px.histogram(
                data,
                x="Revenue",
                nbins=16,
                color_discrete_sequence=[BLUE],
                labels={"Revenue": "Transaction revenue", "count": "Transactions"},
            )
            distribution_chart.update_traces(
                marker_line_color="#132338",
                marker_line_width=1,
                hovertemplate="Revenue: $%{x:,.2f}<br>Transactions: %{y}<extra></extra>",
            )
            st.plotly_chart(_plot_layout(distribution_chart, 330), width="stretch")
    with monthly:
        with st.container(border=True):
            st.markdown("**Monthly revenue breakdown**")
            monthly_sales = daily_chart_data.copy()
            monthly_sales["Month"] = (
                monthly_sales["Date"].dt.to_period("M").dt.to_timestamp()
            )
            monthly_sales = (
                monthly_sales.groupby("Month", as_index=False)["Revenue"]
                .sum()
                .sort_values("Month")
            )
            monthly_chart = px.bar(
                monthly_sales,
                x="Month",
                y="Revenue",
                color_discrete_sequence=[TEAL],
                labels={"Month": "Month", "Revenue": "Revenue"},
            )
            monthly_chart.update_traces(
                hovertemplate="%{x|%B %Y}<br>$%{y:,.2f}<extra></extra>"
            )
            monthly_chart.update_xaxes(tickformat="%b %Y")
            st.plotly_chart(_plot_layout(monthly_chart, 330), width="stretch")

    _section_header(
        "Business insights",
        "Concise implications and clear next steps from the sales evidence.",
    )
    actions = {
        "Revenue trend insight": (
            "Compare the next equivalent recorded-date period and investigate "
            "the largest contributing products."
        ),
        "Top product contribution insight": (
            "Monitor the leading product's share and test complementary offers."
        ),
        "Product concentration risk analysis": (
            "Pilot a measured cross-sell initiative and track contribution "
            "outside the leading product."
        ),
        "Product opportunity insight": (
            "Test a limited offer for the identified product before scaling investment."
        ),
        "Average order value analysis": (
            "Test a bundle or complementary offer while monitoring conversion "
            "alongside order value."
        ),
        "Forecast insight": (
            "Compare forecast with realized revenue and revisit the estimate "
            "as new sales arrive."
        ),
        "Actionable recommendation": (
            "Run one small, measurable pilot and assess it against a defined baseline."
        ),
    }
    for start in range(0, len(insights), 2):
        cards = st.columns(2)
        for column, insight in zip(cards, insights[start : start + 2]):
            recommendation = actions.get(
                insight["title"],
                "Reconcile the source data and validate the calculation before acting.",
            )
            impact = (
                "May require attention in the next planning cycle; confirm the pattern "
                "against operating context."
                if insight["tone"] == "caution"
                else "Provides a measured signal to guide commercial focus and subsequent testing."
            )
            with column:
                with st.container(border=True):
                    st.markdown(f"### {insight['title']}")
                    st.markdown('<div class="insight-label">Insight</div>', unsafe_allow_html=True)
                    st.markdown(f"**{insight['metric']}**  \n{insight['detail']}")
                    st.markdown('<div class="insight-label">Impact</div>', unsafe_allow_html=True)
                    st.markdown(impact)
                    st.markdown('<div class="insight-label">Recommendation</div>', unsafe_allow_html=True)
                    st.markdown(recommendation)

    _section_header(
        "Forecast summary",
        "Planning outlook from the Prophet forecast, when generated for this dataset.",
    )
    if forecast_summary is None:
        render_forecast_prompt_card()
    else:
        with st.container(border=True):
            forecast_metrics = st.columns(4)
            forecast_metrics[0].metric(
                "💰 Average daily forecast",
                f"${forecast_summary['average']:,.2f}",
                border=True,
            )
            forecast_metrics[1].metric(
                "📈 Expected growth",
                f"{forecast_growth:+.1f}%" if forecast_growth is not None else "N/A",
                border=True,
            )
            forecast_metrics[2].metric(
                "🔺 Highest forecast month",
                forecast_summary["high_month"],
                border=True,
            )
            forecast_metrics[3].metric(
                "🔻 Lowest forecast month",
                forecast_summary["low_month"],
                border=True,
            )
            interval_width = forecast_summary["interval_half_width"]
            if interval_width is None:
                st.caption(
                    "**Forecast confidence:** model uncertainty interval was not "
                    "included in the saved results."
                )
            else:
                st.caption(
                    "**Forecast confidence:** Prophet's 80% prediction interval "
                    "has an average half-width "
                    f"of ±${interval_width:,.2f} per day. This model interval is not "
                    "a guarantee of realized outcomes. Month averages may cover "
                    "partial calendar months."
                )
            if forecast_growth is not None and forecast_growth < 0:
                st.markdown("**Business recommendation**")
                st.warning(
                    "Plan conservatively and monitor forecast error against actual "
                    "revenue before changing spend or inventory."
                )
            else:
                st.markdown("**Business recommendation**")
                st.info(
                    "Use the forecast as a planning input only; revisit decisions "
                    "when actual revenue is recorded."
                )

    render_data_quality_report(raw_data)

    _section_header(
        "Downloads",
        "Export the current executive briefing, validated transactions, forecast, and PDF.",
    )
    report = st.session_state.get("consultant_report") or st.session_state.get(
        "ai_report"
    )
    has_ai_report = isinstance(report, str) and bool(report.strip())
    if not has_ai_report:
        report = (
            "AI Business Copilot - Evidence-based business insights\n\n"
            + "\n\n".join(
                f"{item['title']}\n{item['metric']}\n{item['detail']}"
                for item in insights
            )
        )
    text_bytes = report.encode("utf-8")
    processed_csv = data.to_csv(index=False).encode("utf-8")
    pdf_bytes = st.session_state.get("consultant_report_pdf") or st.session_state.get(
        "ai_pdf_bytes"
    )
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
        pdf_bytes = _pdf_from_markdown(report)
    download_columns = st.columns(4)
    download_columns[0].download_button(
        "📝 AI Insights (.txt)" if has_ai_report else "📝 Insights (.txt)",
        data=text_bytes,
        file_name=(
            "ai_business_insights.txt"
            if has_ai_report
            else "business_insights.txt"
        ),
        mime="text/plain",
        width="stretch",
    )
    download_columns[1].download_button(
        "📄 Processed Dataset (.csv)",
        data=processed_csv,
        file_name="processed_sales_dataset.csv",
        mime="text/csv",
        width="stretch",
    )
    forecast_csv = (
        forecast.to_csv(index=False).encode("utf-8")
        if isinstance(forecast, pd.DataFrame) and not forecast.empty
        else b""
    )
    download_columns[2].download_button(
        "🔭 Forecast Results (.csv)",
        data=forecast_csv,
        file_name="sales_forecast.csv",
        mime="text/csv",
        disabled=not bool(forecast_csv),
        width="stretch",
    )
    download_columns[3].download_button(
        "📘 Executive PDF Report",
        data=pdf_bytes,
        file_name="ai_business_executive_report.pdf",
        mime="application/pdf",
        width="stretch",
    )
