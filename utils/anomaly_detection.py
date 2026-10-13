"""Daily sales anomaly detection with an Isolation Forest and robust fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd

SEVERITY_LEVELS = ("Low", "Medium", "High", "Critical")
MIN_MODEL_ROWS = 8


def _severity(deviation_pct: float, model_flag: bool) -> str:
    """Assign a consistent user-facing severity from deviation magnitude."""
    if deviation_pct >= 200:
        return "Critical"
    if deviation_pct >= 100:
        return "High"
    if deviation_pct >= 50:
        return "Medium"
    return "Low" if model_flag else "Low"


def _robust_z_scores(values: pd.Series) -> np.ndarray:
    """Calculate modified Z-scores using median absolute deviation."""
    numeric = values.astype(float).to_numpy()
    median = float(np.median(numeric))
    mad = float(np.median(np.abs(numeric - median)))
    if mad <= 1e-12:
        standard_deviation = float(np.std(numeric))
        if standard_deviation <= 1e-12:
            return np.zeros(len(numeric), dtype=float)
        return (numeric - median) / standard_deviation
    return 0.6745 * (numeric - median) / mad


def _base_daily_data(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into dated revenue, order-count, and quantity metrics."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        return pd.DataFrame(columns=["Date", "Revenue", "Orders", "Quantity"])
    if not {"Date", "Revenue"}.issubset(data.columns):
        return pd.DataFrame(columns=["Date", "Revenue", "Orders", "Quantity"])

    clean = data.copy()
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
    clean["Revenue"] = pd.to_numeric(clean["Revenue"], errors="coerce")
    clean["Quantity"] = (
        pd.to_numeric(clean["Quantity"], errors="coerce")
        if "Quantity" in clean
        else 1.0
    )
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(
        subset=["Date", "Revenue", "Quantity"]
    )
    return (
        clean.groupby("Date", as_index=False)
        .agg(Revenue=("Revenue", "sum"), Orders=("Revenue", "size"), Quantity=("Quantity", "sum"))
        .sort_values("Date")
        .reset_index(drop=True)
    )


def detect_sales_anomalies(
    data: pd.DataFrame,
    raw_data: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Detect date-level sales anomalies and source-quality issues.

    Isolation Forest is applied when there are enough dated observations and
    scikit-learn is installed. Otherwise modified Z-scores provide a robust
    fallback. Data-quality anomalies are evaluated independently of model size.
    """
    columns = [
        "Category",
        "Severity",
        "Date",
        "Product",
        "Metric",
        "Expected Value",
        "Actual Value",
        "Deviation %",
        "Detection Method",
        "Details",
    ]
    daily = _base_daily_data(data)
    if daily.empty:
        return pd.DataFrame(columns=columns)

    feature_columns = ["Revenue", "Orders", "Quantity"]
    features = np.log1p(daily[feature_columns].clip(lower=0))
    flagged = np.zeros(len(daily), dtype=bool)
    method = "Z-score fallback"

    # Isolation Forest is intentionally optional at runtime: the requirements
    # include scikit-learn, but the fallback also supports constrained installs.
    if len(daily) >= MIN_MODEL_ROWS:
        try:
            from sklearn.ensemble import IsolationForest

            model = IsolationForest(
                contamination="auto",
                random_state=42,
                n_estimators=200,
            )
            flagged = model.fit_predict(features) == -1
            method = "Isolation Forest"
        except ImportError:
            pass
    elif len(daily) >= 3:
        scores = np.column_stack(
            [_robust_z_scores(features[column]) for column in feature_columns]
        )
        flagged = (np.abs(scores) >= 3.5).any(axis=1)

    result: list[dict[str, object]] = []
    z_scores = (
        np.column_stack(
            [_robust_z_scores(features[column]) for column in feature_columns]
        )
        if len(daily) < MIN_MODEL_ROWS
        else None
    )
    for index, row in daily.iterrows():
        if len(daily) >= MIN_MODEL_ROWS and method == "Isolation Forest":
            is_flagged = bool(flagged[index])
        else:
            is_flagged = bool(
                z_scores is not None and (np.abs(z_scores[index]) >= 3.5).any()
            )
        if not is_flagged:
            continue

        baseline = daily.drop(index=index)
        for metric, category in (
            ("Revenue", "Revenue"),
            ("Orders", "Order"),
            ("Quantity", "Order"),
        ):
            metric_baseline = float(baseline[metric].median()) if not baseline.empty else 0.0
            actual = float(row[metric])
            deviation = (
                abs(actual - metric_baseline) / abs(metric_baseline) * 100
                if metric_baseline != 0
                else (100.0 if actual else 0.0)
            )
            if metric_baseline == 0 or abs(actual - metric_baseline) / max(abs(metric_baseline), 1) >= 0.5:
                direction = "spike" if actual > metric_baseline else "drop"
                result.append(
                    {
                        "Category": (
                            "Revenue spike"
                            if category == "Revenue" and direction == "spike"
                            else "Revenue drop"
                            if category == "Revenue"
                            else "Order anomaly"
                        ),
                        "Severity": _severity(deviation, is_flagged),
                        "Date": pd.Timestamp(row["Date"]).strftime("%Y-%m-%d"),
                        "Product": "",
                        "Metric": metric,
                        "Expected Value": round(metric_baseline, 2),
                        "Actual Value": round(actual, 2),
                        "Deviation %": round(deviation, 1),
                        "Detection Method": method,
                        "Details": f"{metric} {direction} versus the median of other recorded dates.",
                    }
                )

    # Product-level review uses a median across products and reports the date
    # range endpoint for an intelligible date field without implying daily data.
    if isinstance(data, pd.DataFrame) and {"Product", "Revenue"}.issubset(data.columns):
        product_revenue = (
            data.assign(
                Product=data["Product"].astype("string").str.strip(),
                Revenue=pd.to_numeric(data["Revenue"], errors="coerce"),
            )
            .dropna(subset=["Product", "Revenue"])
            .groupby("Product")["Revenue"]
            .sum()
        )
        if len(product_revenue) >= 4:
            median_product = float(product_revenue.median())
            if median_product > 0:
                for product, actual_value in product_revenue.items():
                    actual = float(actual_value)
                    deviation = abs(actual - median_product) / median_product * 100
                    robust_scores = np.abs(_robust_z_scores(product_revenue))
                    if robust_scores[product_revenue.index.get_loc(product)] >= 3.5:
                        direction = "spike" if actual >= median_product else "drop"
                        result.append(
                            {
                                "Category": "Product anomaly",
                                "Severity": _severity(deviation, True),
                                "Date": (
                                    pd.Timestamp(daily["Date"].max()).strftime("%Y-%m-%d")
                                    if not daily.empty
                                    else "Unavailable"
                                ),
                                "Product": str(product),
                                "Metric": "Product revenue",
                                "Expected Value": round(median_product, 2),
                                "Actual Value": round(actual, 2),
                                "Deviation %": round(deviation, 1),
                                "Detection Method": "Robust Z-score",
                                "Details": f"{product} product revenue is an unusual {direction} versus the product median.",
                            }
                        )

    # Duplicate rows and missing optional/source fields are checked on the
    # original upload because normalized data may not retain all source fields.
    source = raw_data if isinstance(raw_data, pd.DataFrame) else data
    if isinstance(source, pd.DataFrame) and not source.empty:
        duplicate_count = int(source.duplicated().sum())
        if duplicate_count:
            result.append(
                {
                    "Category": "Data quality anomaly",
                    "Severity": (
                        "Critical"
                        if duplicate_count / len(source) >= 0.20
                        else "High"
                        if duplicate_count >= 5
                        else "Medium"
                    ),
                    "Date": "Source file",
                    "Product": "",
                    "Metric": "Duplicate rows",
                    "Expected Value": 0,
                    "Actual Value": duplicate_count,
                    "Deviation %": round(duplicate_count / len(source) * 100, 1),
                    "Detection Method": "Data quality check",
                    "Details": f"{duplicate_count} exact duplicate transaction row(s) were found.",
                }
            )
        missing_cells = int(source.isna().sum().sum())
        if missing_cells:
            missing_rate = missing_cells / max(source.size, 1)
            result.append(
                {
                    "Category": "Data quality anomaly",
                    "Severity": (
                        "Critical"
                        if missing_rate >= 0.20
                        else "High"
                        if missing_rate >= 0.10
                        else "Medium"
                    ),
                    "Date": "Source file",
                    "Product": "",
                    "Metric": "Missing cells",
                    "Expected Value": 0,
                    "Actual Value": missing_cells,
                    "Deviation %": round(missing_rate * 100, 1),
                    "Detection Method": "Data quality check",
                    "Details": f"{missing_cells} blank source cell(s) were found.",
                }
            )

    if not result:
        return pd.DataFrame(columns=columns)
    anomalies = pd.DataFrame(result, columns=columns)
    severity_order = {level: position for position, level in enumerate(SEVERITY_LEVELS)}
    anomalies["_severity_order"] = anomalies["Severity"].map(severity_order)
    return (
        anomalies.sort_values(["_severity_order", "Deviation %"], ascending=[False, False])
        .drop(columns="_severity_order")
        .reset_index(drop=True)
    )
