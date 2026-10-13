"""Executive scorecard and evidence-led CEO dashboard calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_forecast_summary(
    daily_sales: pd.DataFrame,
    forecast_data: pd.DataFrame | None,
) -> dict[str, Any]:
    """Summarize a forecast against recent observed daily sales."""
    result: dict[str, Any] = {
        "available": False,
        "average_daily_revenue": None,
        "total_revenue": None,
        "growth_pct": None,
        "trend": "Unavailable",
        "outlook": "No forecast generated",
        "data": pd.DataFrame(),
    }
    if not isinstance(forecast_data, pd.DataFrame) or forecast_data.empty:
        return result
    if not {"ds", "yhat"}.issubset(forecast_data.columns):
        return result

    forecast = forecast_data.loc[:, ["ds", "yhat"]].copy()
    forecast["ds"] = pd.to_datetime(forecast["ds"], errors="coerce")
    forecast["yhat"] = pd.to_numeric(forecast["yhat"], errors="coerce")
    forecast = (
        forecast.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["ds", "yhat"])
        .sort_values("ds")
    )
    if forecast.empty:
        return result

    recent = pd.DataFrame()
    if isinstance(daily_sales, pd.DataFrame) and {"Date", "Revenue"}.issubset(
        daily_sales.columns
    ):
        recent = daily_sales.loc[:, ["Date", "Revenue"]].copy()
        recent["Date"] = pd.to_datetime(recent["Date"], errors="coerce")
        recent["Revenue"] = pd.to_numeric(recent["Revenue"], errors="coerce")
        recent = recent.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["Date", "Revenue"]
        )
        recent = recent.sort_values("Date").tail(14)

    forecast_average = float(forecast["yhat"].mean())
    recent_average = float(recent["Revenue"].mean()) if not recent.empty else None
    growth_pct = (
        (forecast_average / recent_average - 1) * 100
        if recent_average is not None and recent_average > 0
        else None
    )
    endpoints = float(forecast["yhat"].iloc[-1] - forecast["yhat"].iloc[0])
    relative_endpoint = (
        endpoints / max(abs(float(forecast["yhat"].iloc[0])), 1.0) * 100
    )
    trend = "Rising" if relative_endpoint > 2 else "Falling" if relative_endpoint < -2 else "Stable"
    outlook = (
        "Positive"
        if growth_pct is not None and growth_pct > 2
        else "Cautious"
        if growth_pct is not None and growth_pct < -2
        else "Stable"
        if growth_pct is not None
        else "Forecast available; baseline unavailable"
    )
    result.update(
        {
            "available": True,
            "average_daily_revenue": forecast_average,
            "total_revenue": float(forecast["yhat"].sum()),
            "growth_pct": growth_pct,
            "trend": trend,
            "outlook": outlook,
            "data": forecast,
            "recent_daily_average": recent_average,
        }
    )
    return result


def build_ceo_scorecard(
    health: dict[str, Any],
    forecast: dict[str, Any],
    churn: dict[str, Any],
    segmentation: dict[str, Any],
    anomalies: pd.DataFrame,
) -> dict[str, Any]:
    """Combine health, forecast, customer stability, and risk exposure into a score."""
    health_score = float(np.clip(health.get("overall_score", 0), 0, 100))

    growth = forecast.get("growth_pct")
    forecast_score = (
        float(np.clip(50 + float(growth) * 2, 0, 100))
        if growth is not None
        else 50.0
    )

    if churn.get("available"):
        total_customers = len(churn["customers"])
        high_share = (
            float(churn["high_risk_count"]) / total_customers
            if total_customers
            else 0.0
        )
        medium_share = (
            float(churn["medium_risk_count"]) / total_customers
            if total_customers
            else 0.0
        )
        customer_score = float(np.clip(100 - high_share * 100 - medium_share * 35, 0, 100))
        customer_basis = "Customer churn risk distribution"
    elif segmentation.get("available"):
        summary = segmentation["summary"]
        customers = int(summary["Customers"].sum())
        recent = float(summary["Recent Customers"].sum()) if customers else 0.0
        customer_score = float(np.clip(recent / customers * 100, 0, 100)) if customers else 50.0
        customer_basis = "Recent-purchase indicators from customer segments"
    else:
        customer_score = 50.0
        customer_basis = "Unavailable; neutral score used"

    if isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
        high_count = int(anomalies["Severity"].isin(["High", "Critical"]).sum())
        total_count = len(anomalies)
        exposure_score = float(np.clip(100 - min(high_count / max(total_count, 1), 1) * 100, 0, 100))
        exposure_basis = f"{high_count} High/Critical finding(s) of {total_count}"
    else:
        exposure_score = 100.0
        exposure_basis = "No detected anomaly findings"

    component_scores = {
        "Business Health": health_score,
        "Forecast Outlook": forecast_score,
        "Customer Stability": customer_score,
        "Risk Exposure": exposure_score,
    }
    score = int(
        round(
            health_score * 0.40
            + forecast_score * 0.25
            + customer_score * 0.20
            + exposure_score * 0.15
        )
    )
    category = (
        "Exceptional"
        if score >= 90
        else "Strong"
        if score >= 75
        else "Stable"
        if score >= 60
        else "Needs Attention"
    )
    return {
        "score": score,
        "category": category,
        "components": component_scores,
        "customer_basis": customer_basis,
        "exposure_basis": exposure_basis,
    }


def build_executive_priorities(
    health: dict[str, Any],
    forecast: dict[str, Any],
    churn: dict[str, Any],
    segmentation: dict[str, Any],
    anomalies: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """Derive short risk and opportunity lists from available measured signals."""
    risks: list[str] = []
    opportunities: list[str] = []

    risks.extend(str(item) for item in health.get("risks", []) if item)
    recommendations = health.get("recommendations", [])
    opportunities.extend(str(item) for item in recommendations if item)

    growth = forecast.get("growth_pct")
    if growth is not None and float(growth) < -2:
        risks.insert(
            0,
            f"Forecast average daily revenue is projected {abs(float(growth)):.1f}% below the recent recorded-date average.",
        )
    elif growth is not None and float(growth) > 2:
        opportunities.insert(
            0,
            f"Forecast average daily revenue is {float(growth):.1f}% above the recent recorded-date average; validate against realized sales.",
        )
    elif not forecast.get("available"):
        risks.append("No forecast is available to inform forward revenue planning.")

    if churn.get("available"):
        high_count = int(churn["high_risk_count"])
        medium_count = int(churn["medium_risk_count"])
        if high_count:
            risks.append(
                f"{high_count} customer(s) are classified High Risk by the {churn['mode'].replace('_', ' ')} churn method."
            )
        if medium_count:
            risks.append(f"{medium_count} customer(s) are classified Medium Risk.")
        if high_count:
            opportunities.append(
                f"Prioritize a measured retention outreach test for {high_count} High Risk customer(s)."
            )
    else:
        risks.append("Customer-level churn risk is unavailable without a customer identifier.")

    if segmentation.get("available"):
        summary = segmentation["summary"].set_index("Segment")
        at_risk = summary.loc["At-Risk Customers"] if "At-Risk Customers" in summary.index else None
        vip = summary.loc["VIP Customers"] if "VIP Customers" in summary.index else None
        if at_risk is not None and int(at_risk["Customers"]) > 0:
            risks.append(
                f"{int(at_risk['Customers'])} customers fall in the lower-value/recency At-Risk segment."
            )
            opportunities.append(
                "Test targeted reactivation for the At-Risk segment and compare repeat purchase activity."
            )
        if vip is not None and int(vip["Customers"]) > 0:
            opportunities.append(
                f"VIP customers contribute {float(vip['Revenue Contribution %']):.1f}% of identified-customer revenue; protect this cohort with a measured loyalty test."
            )
    else:
        risks.append("Customer segment profiles are unavailable for account-level opportunity planning.")

    if isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
        severe = anomalies.loc[anomalies["Severity"].isin(["High", "Critical"])]
        if not severe.empty:
            first = severe.iloc[0]
            risks.append(
                f"{len(severe)} High/Critical anomaly finding(s) require source verification; leading item: {first['Category']} on {first['Date']}."
            )
        else:
            risks.append(
                f"{len(anomalies)} low-to-medium anomaly finding(s) warrant routine review."
            )
    else:
        opportunities.append("No anomaly findings were detected in the available analysis.")

    def _unique(items: list[str]) -> list[str]:
        unique: list[str] = []
        seen: set[str] = set()
        for item in items:
            normalized = item.casefold().strip()
            if item.strip() and normalized not in seen:
                seen.add(normalized)
                unique.append(item)
            if len(unique) == 5:
                break
        return unique

    return _unique(risks), _unique(opportunities)


def build_ceo_ai_context(
    sales: pd.DataFrame,
    health: dict[str, Any],
    forecast: dict[str, Any],
    churn: dict[str, Any],
    segmentation: dict[str, Any],
    anomalies: pd.DataFrame,
    scorecard: dict[str, Any],
    risks: list[str],
    opportunities: list[str],
) -> str:
    """Build a bounded, privacy-conscious context for the CEO Groq briefing."""
    total_revenue = float(pd.to_numeric(sales["Revenue"], errors="coerce").fillna(0).sum())
    total_orders = int(len(sales))
    average_order_value = total_revenue / total_orders if total_orders else 0.0
    products = (
        sales.assign(Revenue=pd.to_numeric(sales["Revenue"], errors="coerce"))
        .groupby("Product")["Revenue"]
        .sum()
        .sort_values(ascending=False)
    )
    top_product = (
        f"{products.index[0]} ({float(products.iloc[0]):,.2f})"
        if not products.empty
        else "Unavailable"
    )
    churn_summary = (
        {
            "method": churn["mode"],
            "high_risk": churn["high_risk_count"],
            "medium_risk": churn["medium_risk_count"],
            "low_risk": churn["low_risk_count"],
            "metrics": churn.get("metrics"),
        }
        if churn.get("available")
        else {"available": False, "reason": churn.get("message")}
    )
    segment_summary = (
        segmentation["summary"].to_dict(orient="records")
        if segmentation.get("available")
        else {"available": False, "reason": segmentation.get("message")}
    )
    anomaly_summary: dict[str, Any]
    if isinstance(anomalies, pd.DataFrame) and not anomalies.empty:
        counts = anomalies["Severity"].value_counts().to_dict()
        anomaly_summary = {"total": len(anomalies), "severity_counts": counts}
    else:
        anomaly_summary = {"total": 0, "severity_counts": {}}

    return f"""EXECUTIVE BUSINESS SNAPSHOT (revenue units follow the source; currency unknown)
- Recorded revenue: {total_revenue:,.2f}
- Transaction rows: {total_orders:,}
- Revenue per row: {average_order_value:,.2f} (may not represent a distinct order)
- Highest-revenue product: {top_product}
- Business health: {health.get('overall_score')}/100, {health.get('category')}
- CEO score: {scorecard['score']}/100, {scorecard['category']}
- Forecast: {forecast.get('outlook')}; average daily={forecast.get('average_daily_revenue')}; growth vs. recent baseline={forecast.get('growth_pct')}
- Churn profile: {churn_summary}
- Customer segments: {segment_summary}
- Anomalies: {anomaly_summary}
- Top identified risks: {risks}
- Top opportunities: {opportunities}

DATA LIMITS
Churn rules and customer-segment recency are indicators derived from uploaded
transaction history, not confirmed future outcomes or causal explanations.
Do not infer margins, customer motivations, churn causes, inventory, channel
performance, or market conditions unless included in the evidence above.
"""
