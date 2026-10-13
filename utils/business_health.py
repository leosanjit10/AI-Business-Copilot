"""Evidence-based business health scoring for sales dashboards and reports."""

from __future__ import annotations

import numpy as np
import pandas as pd

COMPONENT_MAX = 25


def _daily_revenue(data: pd.DataFrame) -> pd.DataFrame:
    """Return validated daily revenue totals from transaction-level sales data."""
    if not {"Date", "Revenue"}.issubset(data.columns):
        return pd.DataFrame(columns=["Date", "Revenue"])

    daily = data.loc[:, ["Date", "Revenue"]].copy()
    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    daily["Revenue"] = pd.to_numeric(daily["Revenue"], errors="coerce")
    daily = daily.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Date", "Revenue"]
    )
    return (
        daily.groupby("Date", as_index=False)["Revenue"]
        .sum()
        .sort_values("Date")
        .reset_index(drop=True)
    )


def _score_directional_change(change: float) -> int:
    """Map relative growth to 0-25 points, centered on neutral performance."""
    return int(round(np.clip(12.5 + 12.5 * change, 0, COMPONENT_MAX)))


def calculate_business_health_score(
    df: pd.DataFrame,
    forecast_df: pd.DataFrame | None,
) -> dict[str, object]:
    """Score growth, stability, outlook, and product diversification.

    Component scores are each worth 25 points. Insufficient history or a
    missing forecast receives a neutral component score rather than being
    misrepresented as a business failure; the limitation is stated explicitly.
    """
    strengths: list[str] = []
    risks: list[str] = []
    recommendations: list[str] = []

    if not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "overall_score": 0,
            "category": "At Risk",
            "strengths": [],
            "risks": ["No valid sales data is available to calculate business health."],
            "recommendations": [
                "Upload a valid sales CSV containing Date, Product, Quantity, and Revenue."
            ],
            "component_scores": {
                "Revenue Growth": 0,
                "Revenue Stability": 0,
                "Forecast Outlook": 0,
                "Product Diversification": 0,
            },
        }

    daily = _daily_revenue(df)

    # Compare equally sized groups of recent recorded dates. Missing calendar
    # dates are not treated as zero-sales days because the source may be sparse.
    if len(daily) >= 4:
        period_size = min(14, len(daily) // 2)
        prior = float(daily.iloc[-2 * period_size : -period_size]["Revenue"].sum())
        recent = float(daily.iloc[-period_size:]["Revenue"].sum())
        if prior > 0:
            revenue_change = (recent - prior) / prior
            growth_score = _score_directional_change(revenue_change)
            if revenue_change >= 0.10:
                strengths.append(
                    f"Revenue grew {revenue_change:.1%} across comparable recorded-date periods."
                )
            elif revenue_change < 0:
                risks.append(
                    f"Revenue declined {abs(revenue_change):.1%} across comparable recorded-date periods."
                )
            else:
                strengths.append("Revenue is broadly stable across recent comparable periods.")
        else:
            growth_score = 12
            risks.append("A positive prior-period revenue baseline is unavailable.")
            recommendations.append(
                "Collect additional sales history before drawing a directional growth conclusion."
            )
    else:
        growth_score = 12
        risks.append("There are too few distinct sales dates to compare revenue growth.")
        recommendations.append(
            "Continue collecting distinct sales dates to establish a reliable growth baseline."
        )

    # Coefficient of variation is a scale-independent measure of daily revenue
    # variability; a 0-25 inverse mapping rewards steadier observed revenue.
    if len(daily) >= 2:
        mean_revenue = float(daily["Revenue"].mean())
        if mean_revenue > 0:
            coefficient = float(daily["Revenue"].std(ddof=0) / mean_revenue)
            stability_score = int(round(COMPONENT_MAX / (1 + coefficient)))
            if coefficient <= 0.35:
                strengths.append("Daily recorded revenue is relatively consistent.")
            elif coefficient >= 0.8:
                risks.append("Daily revenue varies substantially around its average.")
                recommendations.append(
                    "Investigate high- and low-revenue dates and avoid planning from a single-day result."
                )
        else:
            stability_score = 0
            risks.append("Recorded daily revenue averages zero, preventing a stability assessment.")
    else:
        stability_score = 12
        risks.append("There is insufficient daily history to assess revenue stability.")

    # Use the Prophet prediction schema as well as the common Date/Revenue shape.
    forecast_values = pd.Series(dtype="float64")
    if isinstance(forecast_df, pd.DataFrame) and not forecast_df.empty:
        if {"ds", "yhat"}.issubset(forecast_df.columns):
            forecast_values = pd.to_numeric(forecast_df["yhat"], errors="coerce")
        elif {"Date", "Revenue"}.issubset(forecast_df.columns):
            forecast_values = pd.to_numeric(forecast_df["Revenue"], errors="coerce")
    forecast_values = forecast_values.replace([np.inf, -np.inf], np.nan).dropna()
    if not forecast_values.empty and not daily.empty and float(daily["Revenue"].mean()) > 0:
        forecast_change = float(forecast_values.mean() / daily.tail(14)["Revenue"].mean() - 1)
        forecast_score = _score_directional_change(forecast_change)
        if forecast_change >= 0.10:
            strengths.append(
                f"Forecast average is {forecast_change:.1%} above the latest recorded-date average."
            )
        elif forecast_change < -0.10:
            risks.append(
                f"Forecast average is {abs(forecast_change):.1%} below the recent revenue baseline."
            )
            recommendations.append(
                "Use the forecast as a planning signal and validate inventory or spending decisions against actual sales."
            )
    else:
        forecast_score = 12
        risks.append("A usable revenue forecast is not available; the outlook component is neutral.")
        recommendations.append(
            "Generate a 30-day forecast when sufficient distinct sales dates are available."
        )

    if "Product" in df.columns:
        products = (
            df.assign(
                Product=df["Product"].astype("string").str.strip(),
                Revenue=pd.to_numeric(df["Revenue"], errors="coerce"),
            )
            .dropna(subset=["Product", "Revenue"])
            .groupby("Product")["Revenue"]
            .sum()
        )
        product_revenue = float(products.sum())
        if len(products) and product_revenue > 0:
            shares = products / product_revenue
            hhi = float(shares.pow(2).sum())
            diversification_score = int(round(COMPONENT_MAX * (1 - hhi)))
            leading_share = float(shares.max())
            if leading_share <= 0.40 and len(products) > 1:
                strengths.append(
                    f"Product revenue is distributed across {len(products)} products; the leader contributes {leading_share:.1%}."
                )
            elif leading_share >= 0.60:
                risks.append(
                    f"The leading product accounts for {leading_share:.1%} of product revenue."
                )
                recommendations.append(
                    "Test one measurable cross-sell or complementary-product offer before increasing dependence on the leading product."
                )
        else:
            diversification_score = 0
            risks.append("Product revenue is unavailable for diversification scoring.")
    else:
        diversification_score = 0
        risks.append("Product labels are missing, so product diversification cannot be measured.")

    overall_score = int(
        np.clip(
            growth_score + stability_score + forecast_score + diversification_score,
            0,
            100,
        )
    )
    category = (
        "Excellent"
        if overall_score >= 90
        else "Healthy"
        if overall_score >= 75
        else "Average"
        if overall_score >= 60
        else "At Risk"
    )

    if not recommendations:
        recommendations.append(
            "Sustain current performance and recheck the score as new sales data becomes available."
        )

    return {
        "overall_score": overall_score,
        "category": category,
        "strengths": strengths or ["No strong positive signal was identified in the available data."],
        "risks": risks or ["No material risk signal was identified in the available data."],
        "recommendations": recommendations,
        "component_scores": {
            "Revenue Growth": growth_score,
            "Revenue Stability": stability_score,
            "Forecast Outlook": forecast_score,
            "Product Diversification": diversification_score,
        },
    }
