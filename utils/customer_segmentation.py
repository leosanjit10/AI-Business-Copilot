"""Customer-level RFM feature engineering and KMeans segmentation."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

CUSTOMER_KEYS = {
    "customer",
    "customerid",
    "client",
    "clientid",
    "customername",
    "clientname",
}
FEATURE_COLUMNS = ["Total Spend", "Purchase Frequency", "Average Order Value", "Recency Days"]
SEGMENTS = ["VIP Customers", "Loyal Customers", "Regular Customers", "At-Risk Customers"]


def find_customer_column(data: pd.DataFrame) -> str | None:
    """Find a plausible customer identifier without treating products as people."""
    if not isinstance(data, pd.DataFrame):
        return None
    for column in data.columns:
        normalized = re.sub(r"[^a-z0-9]", "", str(column).casefold())
        if normalized in CUSTOMER_KEYS:
            return str(column)
    return None


def build_customer_segments(data: pd.DataFrame) -> dict[str, Any]:
    """Create customer-level RFM features, select KMeans cluster count, and profile tiers."""
    empty_customers = pd.DataFrame(
        columns=[*FEATURE_COLUMNS, "Customer", "Cluster", "Segment", "Recency Status"]
    )
    if not isinstance(data, pd.DataFrame) or data.empty:
        return {
            "available": False,
            "message": "Upload a non-empty sales dataset to analyze customers.",
            "customers": empty_customers,
            "summary": pd.DataFrame(),
            "cluster_count": 0,
            "customer_column": None,
        }

    customer_column = find_customer_column(data)
    if customer_column is None:
        return {
            "available": False,
            "message": (
                "Customer segmentation needs a customer identifier column such as "
                "CustomerID, Customer ID, Customer, ClientID, or Client. The current "
                "sales file contains no such field, so customer identities cannot be inferred."
            ),
            "customers": empty_customers,
            "summary": pd.DataFrame(),
            "cluster_count": 0,
            "customer_column": None,
        }

    required = {customer_column, "Date", "Revenue"}
    missing = required.difference(data.columns)
    if missing:
        return {
            "available": False,
            "message": "Customer segmentation requires customer ID, Date, and Revenue columns.",
            "customers": empty_customers,
            "summary": pd.DataFrame(),
            "cluster_count": 0,
            "customer_column": customer_column,
        }

    clean = data.copy()
    clean[customer_column] = clean[customer_column].astype("string").str.strip()
    clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
    clean["Revenue"] = pd.to_numeric(clean["Revenue"], errors="coerce")
    clean = clean.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[customer_column, "Date", "Revenue"]
    )
    clean = clean.loc[clean[customer_column].ne("") & clean["Revenue"].ge(0)]
    if clean.empty:
        return {
            "available": False,
            "message": "No valid customer transactions remain after cleaning.",
            "customers": empty_customers,
            "summary": pd.DataFrame(),
            "cluster_count": 0,
            "customer_column": customer_column,
        }

    quantity = (
        pd.to_numeric(clean["Quantity"], errors="coerce").fillna(1).clip(lower=0)
        if "Quantity" in clean
        else pd.Series(1.0, index=clean.index)
    )
    clean["_Quantity"] = quantity
    snapshot_date = clean["Date"].max()
    customers = (
        clean.groupby(customer_column)
        .agg(
            **{
                "Total Spend": ("Revenue", "sum"),
                "Purchase Frequency": ("Revenue", "size"),
                "Last Purchase": ("Date", "max"),
                "Quantity Purchased": ("_Quantity", "sum"),
            }
        )
        .reset_index()
        .rename(columns={customer_column: "Customer"})
    )
    customers["Average Order Value"] = (
        customers["Total Spend"] / customers["Purchase Frequency"].clip(lower=1)
    )
    customers["Recency Days"] = (
        snapshot_date - customers["Last Purchase"]
    ).dt.days.astype(float)

    if len(customers) < 3:
        return {
            "available": False,
            "message": (
                f"At least 3 distinct customers are required for meaningful clustering; "
                f"the dataset contains {len(customers)}."
            ),
            "customers": customers,
            "summary": pd.DataFrame(),
            "cluster_count": 0,
            "customer_column": customer_column,
        }

    # Clustering needs at least two distinct RFM feature vectors. Log scaling
    # limits domination from a few very large spend values.
    feature_values = customers[FEATURE_COLUMNS].clip(lower=0)
    transformed = np.log1p(feature_values)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler

    scaled = StandardScaler().fit_transform(transformed)
    distinct_rows = len(np.unique(np.round(scaled, decimals=10), axis=0))
    if distinct_rows < 2:
        return {
            "available": False,
            "message": "Customer features are identical; KMeans cannot form meaningful groups.",
            "customers": customers,
            "summary": pd.DataFrame(),
            "cluster_count": 0,
            "customer_column": customer_column,
        }

    max_clusters = min(6, len(customers) - 1, distinct_rows)
    best_model: Any = None
    best_score = -1.0
    for cluster_count in range(2, max_clusters + 1):
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        labels = model.fit_predict(scaled)
        if len(np.unique(labels)) < 2 or len(np.unique(labels)) >= len(customers):
            continue
        score = float(silhouette_score(scaled, labels))
        if score > best_score:
            best_model = model
            best_score = score

    if best_model is None:
        return {
            "available": False,
            "message": "There are too few distinct customer profiles to choose stable clusters.",
            "customers": customers,
            "summary": pd.DataFrame(),
            "cluster_count": 0,
            "customer_column": customer_column,
        }

    customers["Cluster"] = best_model.labels_
    cluster_profiles = customers.groupby("Cluster")[FEATURE_COLUMNS].mean()
    rank_frame = cluster_profiles.copy()
    # Normalize the direction of each RFM signal before deriving segment tiers.
    for column in ("Total Spend", "Purchase Frequency", "Average Order Value"):
        span = float(rank_frame[column].max() - rank_frame[column].min())
        rank_frame[column] = (
            (rank_frame[column] - rank_frame[column].min()) / span if span else 0.5
        )
    recency_span = float(rank_frame["Recency Days"].max() - rank_frame["Recency Days"].min())
    rank_frame["Recency Score"] = (
        (rank_frame["Recency Days"].max() - rank_frame["Recency Days"]) / recency_span
        if recency_span
        else 0.5
    )
    cluster_value = (
        rank_frame["Total Spend"] * 0.40
        + rank_frame["Purchase Frequency"] * 0.25
        + rank_frame["Average Order Value"] * 0.15
        + rank_frame["Recency Score"] * 0.20
    )
    ordered_clusters = cluster_value.sort_values(ascending=False).index.tolist()
    cluster_to_segment: dict[int, str] = {}
    last_tier = len(SEGMENTS) - 1
    for position, cluster in enumerate(ordered_clusters):
        tier = (
            0
            if len(ordered_clusters) == 1
            else round(position / (len(ordered_clusters) - 1) * last_tier)
        )
        cluster_to_segment[int(cluster)] = SEGMENTS[int(tier)]
    customers["Segment"] = customers["Cluster"].map(cluster_to_segment)
    customers["Recency Status"] = np.where(
        customers["Recency Days"] <= 30, "Recent (≤30 days)", "Stale (>30 days)"
    )

    summary = (
        customers.groupby("Segment", as_index=False)
        .agg(
            Customers=("Customer", "nunique"),
            Revenue=("Total Spend", "sum"),
            **{
                "Average Customer Value": ("Total Spend", "mean"),
                "Average Purchase Frequency": ("Purchase Frequency", "mean"),
                "Average Order Value": ("Average Order Value", "mean"),
                "Average Recency Days": ("Recency Days", "mean"),
            },
        )
    )
    summary["Revenue Contribution %"] = (
        summary["Revenue"] / float(customers["Total Spend"].sum()) * 100
        if float(customers["Total Spend"].sum()) > 0
        else 0.0
    )
    summary["Recent Customers"] = customers.loc[
        customers["Recency Days"] <= 30
    ].groupby("Segment")["Customer"].nunique().reindex(summary["Segment"]).fillna(0).to_numpy()
    summary["Retention Indicator"] = np.where(
        summary["Customers"] > 0,
        summary["Recent Customers"] / summary["Customers"] * 100,
        0.0,
    )

    return {
        "available": True,
        "message": (
            f"Clustered {len(customers)} distinct customers into "
            f"{best_model.n_clusters} groups (silhouette score {best_score:.2f})."
        ),
        "customers": customers,
        "summary": summary,
        "cluster_count": int(best_model.n_clusters),
        "silhouette_score": best_score,
        "customer_column": customer_column,
        "snapshot_date": snapshot_date,
    }
