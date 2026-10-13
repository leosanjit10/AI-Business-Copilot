"""Customer churn features, supervised prediction, and rule-based risk scoring."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from utils.customer_segmentation import find_customer_column

FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary Value",
    "Average Order Value",
    "Purchase Count",
    "Days Since Last Purchase",
]
RISK_LEVELS = ("High Risk", "Medium Risk", "Low Risk")
LABEL_ALIASES = {
    "churn",
    "churned",
    "ischurn",
    "churnlabel",
    "churnstatus",
    "attrition",
    "attrited",
    "isattrited",
}
POSITIVE_LABELS = {
    "1",
    "true",
    "yes",
    "y",
    "churn",
    "churned",
    "inactive",
    "attrited",
    "exited",
    "left",
    "cancelled",
    "canceled",
}
NEGATIVE_LABELS = {
    "0",
    "false",
    "no",
    "n",
    "active",
    "retained",
    "notchurn",
    "notchurned",
    "notattrited",
    "stay",
    "stayed",
    "nonchurn",
    "nonchurned",
    "notchurnrisk",
}


def find_churn_label_column(data: pd.DataFrame) -> str | None:
    """Find a recognized churn/attrition label column by normalized name."""
    if not isinstance(data, pd.DataFrame):
        return None
    for column in data.columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).casefold())
        if normalized in LABEL_ALIASES:
            return str(column)
    return None


def _empty_result(message: str) -> dict[str, Any]:
    """Return a stable result structure for empty or unsupported customer data."""
    empty = pd.DataFrame(columns=[*FEATURE_COLUMNS, "Customer", "Risk Level", "Churn Probability"])
    return {
        "available": False,
        "mode": "unavailable",
        "message": message,
        "customers": empty,
        "metrics": None,
        "feature_importance": pd.DataFrame(columns=["Feature", "Importance"]),
        "customer_column": None,
        "churn_label_column": None,
        "high_risk_count": 0,
        "medium_risk_count": 0,
        "low_risk_count": 0,
        "snapshot_date": None,
    }


def _coerce_label(value: object) -> float:
    """Map common binary churn encodings to 0/1 and unknown values to NaN."""
    if pd.isna(value):
        return float("nan")
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        if value in (0, 1):
            return float(value)
        return float("nan")

    normalized = re.sub(r"[^a-z0-9]", "", str(value).strip().casefold())
    if normalized in POSITIVE_LABELS:
        return 1.0
    if normalized in NEGATIVE_LABELS:
        return 0.0
    return float("nan")


def _build_customer_features(
    data: pd.DataFrame,
    customer_column: str,
) -> tuple[pd.DataFrame, pd.Timestamp] | None:
    """Aggregate transactions into customer-level recency, frequency, and value."""
    required = {customer_column, "Date", "Revenue"}
    if not required.issubset(data.columns):
        return None

    clean = data.copy()
    clean[customer_column] = clean[customer_column].astype("string").str.strip()
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
    clean["Revenue"] = pd.to_numeric(clean["Revenue"], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[customer_column, "Date", "Revenue"]
    )
    clean = clean.loc[clean[customer_column].ne("") & clean["Revenue"].ge(0)]
    if clean.empty:
        return None

    # Source data has no required order ID, so Purchase Count/Frequency are
    # transaction-row counts and must not be described as unique orders.
    snapshot_date = pd.Timestamp(clean["Date"].max())
    features = (
        clean.groupby(customer_column)
        .agg(
            **{
                "Monetary Value": ("Revenue", "sum"),
                "Purchase Count": ("Revenue", "size"),
                "Last Purchase": ("Date", "max"),
            }
        )
        .reset_index()
        .rename(columns={customer_column: "Customer"})
    )
    features["Frequency"] = features["Purchase Count"].astype(float)
    features["Average Order Value"] = (
        features["Monetary Value"] / features["Purchase Count"].clip(lower=1)
    )
    features["Days Since Last Purchase"] = (
        snapshot_date - features["Last Purchase"]
    ).dt.days.astype(float)
    features["Recency"] = features["Days Since Last Purchase"]
    return features, snapshot_date


def _rule_based_probability(
    customers: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Estimate risk from inactivity and relative purchase frequency."""
    frequency = customers["Frequency"].astype(float)
    low_frequency_threshold = max(
        1.0,
        float(frequency.quantile(0.25)) if not frequency.empty else 1.0,
    )
    low_frequency = frequency.le(low_frequency_threshold)
    recency = customers["Days Since Last Purchase"].astype(float).clip(lower=0)

    # A transparent heuristic: inactivity is the dominant signal; lower
    # transaction frequency raises estimated risk. This is not a trained model.
    inactivity_component = np.clip(recency / 120.0, 0.0, 1.0) * 0.68
    frequency_component = low_frequency.astype(float) * 0.27
    probability = pd.Series(
        np.clip(inactivity_component + frequency_component, 0.02, 0.98),
        index=customers.index,
    )

    # The app reports high risk for 90+ days only when purchase frequency is
    # low. Frequent historical buyers remain medium risk until re-engaged.
    risk = pd.Series("Low Risk", index=customers.index, dtype="string")
    risk.loc[recency.ge(30)] = "Medium Risk"
    risk.loc[recency.ge(90) & low_frequency] = "High Risk"
    # Keep the heuristic probability bands consistent with the accompanying
    # rule-based classification while retaining gradual variation within bands.
    probability.loc[risk.eq("Medium Risk")] = np.maximum(
        probability.loc[risk.eq("Medium Risk")],
        0.40 + np.clip((recency.loc[risk.eq("Medium Risk")] - 30) / 60, 0, 1) * 0.22,
    )
    probability.loc[risk.eq("High Risk")] = np.maximum(
        probability.loc[risk.eq("High Risk")],
        0.70 + np.clip((recency.loc[risk.eq("High Risk")] - 90) / 90, 0, 1) * 0.25,
    )
    probability.loc[risk.eq("Low Risk")] = np.minimum(
        probability.loc[risk.eq("Low Risk")],
        0.34,
    )
    return probability, risk


def _customer_labels(
    data: pd.DataFrame,
    customer_column: str,
    churn_label_column: str,
) -> pd.DataFrame:
    """Use each customer's latest dated binary label to prevent stale statuses."""
    labelled = data.loc[:, [customer_column, "Date", churn_label_column]].copy()
    labelled[customer_column] = labelled[customer_column].astype("string").str.strip()
    labelled["Date"] = pd.to_datetime(labelled["Date"], errors="coerce")
    labelled["_label"] = labelled[churn_label_column].map(_coerce_label)
    labelled = labelled.dropna(subset=[customer_column, "Date", "_label"])
    labelled = labelled.loc[labelled[customer_column].ne("")]
    if labelled.empty:
        return pd.DataFrame(columns=["Customer", "Churn Label"])
    latest = (
        labelled.sort_values("Date")
        .drop_duplicates(customer_column, keep="last")
        .rename(columns={customer_column: "Customer", "_label": "Churn Label"})
    )
    return latest.loc[:, ["Customer", "Churn Label"]]


def analyze_churn(data: pd.DataFrame) -> dict[str, Any]:
    """Build customer risk scores, optionally train a labeled Random Forest.

    Labeled datasets receive a stratified held-out evaluation when enough
    examples exist. Small datasets with both churn classes still train a model,
    but their evaluation metrics are withheld; invalid one-class labels fall
    back to transparent rule-based scoring.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        return _empty_result("Upload non-empty sales data to analyze customer churn.")

    customer_column = find_customer_column(data)
    if customer_column is None:
        return _empty_result(
            "Churn prediction needs a stable CustomerID, Customer_ID, Customer, "
            "ClientID, Client, or Customer Name column."
        )
    customer_features = _build_customer_features(data, customer_column)
    if customer_features is None:
        result = _empty_result(
            "Customer churn analysis requires valid customer IDs, dates, and numeric revenue."
        )
        result["customer_column"] = customer_column
        return result

    customers, snapshot_date = customer_features
    if customers.empty:
        result = _empty_result("No valid customer transactions remain after cleaning.")
        result["customer_column"] = customer_column
        return result

    churn_label_column = find_churn_label_column(data)
    metrics: dict[str, float] | None = None
    feature_importance = pd.DataFrame(columns=["Feature", "Importance"])
    mode = "rule_based"
    message = (
        "Rule-based risk scores use inactivity and purchase frequency; they are "
        "indicators, not calibrated churn probabilities."
    )

    if churn_label_column is not None:
        labels = _customer_labels(data, customer_column, churn_label_column)
        customers = customers.merge(labels, on="Customer", how="left")
        labelled = customers.dropna(subset=["Churn Label"]).copy()
        class_counts = labelled["Churn Label"].value_counts()
        if len(class_counts) == 2:
            features = labelled[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
            valid_rows = features.notna().all(axis=1)
            labelled = labelled.loc[valid_rows].copy()
            features = features.loc[valid_rows]
            labels_array = labelled["Churn Label"].astype(int)
            class_counts = labels_array.value_counts()
            can_evaluate = len(labelled) >= 8 and int(class_counts.min()) >= 2
            if can_evaluate:
                test_rows = max(2, int(np.ceil(len(labelled) * 0.25)))
                can_evaluate = len(labelled) - test_rows >= 2
            if can_evaluate:
                train_x, test_x, train_y, test_y = train_test_split(
                    features,
                    labels_array,
                    test_size=test_rows,
                    random_state=42,
                    stratify=labels_array,
                )
                evaluation_model = RandomForestClassifier(
                    n_estimators=300,
                    class_weight="balanced",
                    random_state=42,
                    min_samples_leaf=2,
                )
                evaluation_model.fit(train_x, train_y)
                predicted = evaluation_model.predict(test_x)
                metrics = {
                    "Accuracy": float(accuracy_score(test_y, predicted)),
                    "Precision": float(
                        precision_score(test_y, predicted, zero_division=0)
                    ),
                    "Recall": float(recall_score(test_y, predicted, zero_division=0)),
                    "F1 Score": float(f1_score(test_y, predicted, zero_division=0)),
                }

            # Fit a serving model on all available labeled profiles. Tiny
            # datasets can still produce scores, but are not given unreliable
            # performance metrics.
            model = RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=42,
                min_samples_leaf=2 if len(labelled) >= 8 else 1,
            )
            model.fit(features, labels_array)
            churn_index = list(model.classes_).index(1)
            labelled["Churn Probability"] = model.predict_proba(features)[:, churn_index]
            customers = customers.merge(
                labelled.loc[:, ["Customer", "Churn Probability"]],
                on="Customer",
                how="left",
            )
            feature_importance = (
                pd.DataFrame(
                    {
                        "Feature": FEATURE_COLUMNS,
                        "Importance": model.feature_importances_,
                    }
                )
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )
            mode = "supervised"
            message = (
                f"Random Forest trained on {len(labelled)} labeled customer profiles. "
                + (
                    f"Stratified held-out metrics are available. Labels are from {churn_label_column}."
                    if metrics is not None
                    else "The sample is too small for reliable held-out metrics; interpret probabilities cautiously."
                )
            )
        else:
            message = (
                f"Recognized {churn_label_column}, but training requires at least "
                "two valid churn classes. "
                "Displaying rule-based scores instead."
            )

    if "Churn Probability" not in customers:
        probabilities, risks = _rule_based_probability(customers)
        customers["Churn Probability"] = probabilities
        customers["Risk Level"] = risks
    else:
        customers["Churn Probability"] = customers["Churn Probability"].fillna(
            _rule_based_probability(customers)[0]
        )
        # Supervised probabilities are classified with explicit, stable bands.
        customers["Risk Level"] = pd.cut(
            customers["Churn Probability"],
            bins=[-np.inf, 0.35, 0.65, np.inf],
            labels=["Low Risk", "Medium Risk", "High Risk"],
            right=False,
        ).astype("string")

    customers["Churn Probability"] = customers["Churn Probability"].clip(0, 1)
    customers["Risk Level"] = customers["Risk Level"].fillna("Medium Risk")
    customers["Risk Order"] = customers["Risk Level"].map(
        {"High Risk": 0, "Medium Risk": 1, "Low Risk": 2}
    )
    customers = customers.sort_values(
        ["Risk Order", "Churn Probability", "Days Since Last Purchase"],
        ascending=[True, False, False],
    ).drop(columns="Risk Order").reset_index(drop=True)

    counts = customers["Risk Level"].value_counts()
    return {
        "available": True,
        "mode": mode,
        "message": message,
        "customers": customers,
        "metrics": metrics,
        "feature_importance": feature_importance,
        "customer_column": customer_column,
        "churn_label_column": churn_label_column,
        "high_risk_count": int(counts.get("High Risk", 0)),
        "medium_risk_count": int(counts.get("Medium Risk", 0)),
        "low_risk_count": int(counts.get("Low Risk", 0)),
        "snapshot_date": snapshot_date,
    }
