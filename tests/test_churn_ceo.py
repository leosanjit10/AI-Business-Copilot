"""Behavioral tests for customer churn prediction and executive scorecards."""

import unittest

import pandas as pd

from utils.anomaly_detection import detect_sales_anomalies
from utils.business_health import calculate_business_health_score
from utils.ceo_dashboard import (
    build_ceo_scorecard,
    build_executive_priorities,
    build_forecast_summary,
)
from utils.churn_prediction import analyze_churn, find_churn_label_column
from utils.customer_segmentation import build_customer_segments


def _customer_sales(customer_count: int = 24, include_label: bool = False) -> pd.DataFrame:
    """Build a deterministic customer-level transaction dataset for tests."""
    rows: list[dict[str, object]] = []
    anchor = pd.Timestamp("2026-09-30")
    for customer_index in range(customer_count):
        churned = customer_index % 2 == 0
        purchase_count = 1 + customer_index % 4
        last_offset = 130 if churned else customer_index % 20
        for purchase_index in range(purchase_count):
            row: dict[str, object] = {
                "CustomerID": f"C{customer_index:03d}",
                "Date": anchor - pd.Timedelta(
                    days=last_offset + (purchase_count - purchase_index - 1) * 35
                ),
                "Product": f"Product {customer_index % 3}",
                "Quantity": 1 + customer_index % 3,
                "Revenue": float(25 + customer_index * 7 + purchase_index * 4),
            }
            if include_label:
                row["ChurnStatus"] = "Churned" if churned else "Active"
            rows.append(row)
    return pd.DataFrame(rows)


class ChurnPredictionTests(unittest.TestCase):
    def test_missing_customer_identifier_returns_actionable_unavailable_result(self) -> None:
        sales = _customer_sales().drop(columns="CustomerID")

        result = analyze_churn(sales)

        self.assertFalse(result["available"])
        self.assertIn("CustomerID", result["message"])
        self.assertEqual(result["mode"], "unavailable")

    def test_customer_history_uses_transparent_rule_based_risk(self) -> None:
        sales = _customer_sales(include_label=False)

        result = analyze_churn(sales)

        self.assertTrue(result["available"])
        self.assertEqual(result["mode"], "rule_based")
        self.assertIn("High Risk", set(result["customers"]["Risk Level"]))
        self.assertIn("Medium Risk", set(result["customers"]["Risk Level"]))
        self.assertIn("Low Risk", set(result["customers"]["Risk Level"]))
        self.assertTrue(
            result["customers"]["Churn Probability"].between(0, 1).all()
        )
        self.assertIn("Days Since Last Purchase", result["customers"].columns)

    def test_valid_binary_labels_train_random_forest_and_report_metrics(self) -> None:
        sales = _customer_sales(include_label=True)

        self.assertEqual(find_churn_label_column(sales), "ChurnStatus")
        result = analyze_churn(sales)

        self.assertTrue(result["available"])
        self.assertEqual(result["mode"], "supervised")
        self.assertEqual(
            set(result["metrics"]),
            {"Accuracy", "Precision", "Recall", "F1 Score"},
        )
        self.assertEqual(
            set(result["feature_importance"]["Feature"]),
            {
                "Recency",
                "Frequency",
                "Monetary Value",
                "Average Order Value",
                "Purchase Count",
                "Days Since Last Purchase",
            },
        )
        self.assertTrue(
            result["customers"]["Churn Probability"].between(0, 1).all()
        )

    def test_unusable_label_falls_back_to_rule_based_scores(self) -> None:
        sales = _customer_sales(include_label=False)
        sales["ChurnStatus"] = "unknown"

        result = analyze_churn(sales)

        self.assertTrue(result["available"])
        self.assertEqual(result["mode"], "rule_based")
        self.assertIsNone(result["metrics"])
        self.assertIn("two valid churn classes", result["message"])

    def test_small_two_class_label_trains_without_claiming_holdout_metrics(self) -> None:
        sales = _customer_sales(customer_count=6, include_label=True)

        result = analyze_churn(sales)

        self.assertTrue(result["available"])
        self.assertEqual(result["mode"], "supervised")
        self.assertIsNone(result["metrics"])
        self.assertTrue(result["feature_importance"].shape[0])
        self.assertIn("sample is too small", result["message"])


class ExecutiveScorecardTests(unittest.TestCase):
    def test_missing_forecast_and_customer_data_produce_explicit_neutral_score(self) -> None:
        sales = _customer_sales()
        health = calculate_business_health_score(sales, None)
        churn = analyze_churn(sales.drop(columns="CustomerID"))
        segmentation = build_customer_segments(sales.drop(columns="CustomerID"))
        anomaly_results = detect_sales_anomalies(sales)
        forecast = build_forecast_summary(
            sales.groupby("Date", as_index=False)["Revenue"].sum(),
            None,
        )

        scorecard = build_ceo_scorecard(
            health,
            forecast,
            churn,
            segmentation,
            anomaly_results,
        )

        self.assertGreaterEqual(scorecard["score"], 0)
        self.assertLessEqual(scorecard["score"], 100)
        self.assertIn(
            scorecard["category"],
            {"Exceptional", "Strong", "Stable", "Needs Attention"},
        )
        self.assertEqual(scorecard["components"]["Forecast Outlook"], 50)
        self.assertEqual(scorecard["components"]["Customer Stability"], 50)
        self.assertIn("Unavailable", scorecard["customer_basis"])

    def test_forecast_summary_calculates_growth_and_total_horizon_revenue(self) -> None:
        daily = pd.DataFrame(
            {
                "Date": pd.date_range("2026-07-01", periods=5),
                "Revenue": [100.0] * 5,
            }
        )
        forecast_data = pd.DataFrame(
            {
                "ds": pd.date_range("2026-07-06", periods=3),
                "yhat": [110.0, 120.0, 130.0],
            }
        )

        forecast = build_forecast_summary(daily, forecast_data)

        self.assertTrue(forecast["available"])
        self.assertEqual(forecast["total_revenue"], 360.0)
        self.assertEqual(forecast["average_daily_revenue"], 120.0)
        self.assertAlmostEqual(forecast["growth_pct"], 20.0)
        self.assertEqual(forecast["outlook"], "Positive")

    def test_priorities_are_capped_at_five_and_data_grounded(self) -> None:
        sales = _customer_sales()
        health = calculate_business_health_score(sales, None)
        churn = analyze_churn(sales)
        segmentation = build_customer_segments(sales)
        anomalies = detect_sales_anomalies(sales)
        forecast = build_forecast_summary(
            sales.groupby("Date", as_index=False)["Revenue"].sum(),
            None,
        )
        scorecard = build_ceo_scorecard(
            health,
            forecast,
            churn,
            segmentation,
            anomalies,
        )

        risks, opportunities = build_executive_priorities(
            health,
            forecast,
            churn,
            segmentation,
            anomalies,
        )

        self.assertLessEqual(len(risks), 5)
        self.assertLessEqual(len(opportunities), 5)
        self.assertTrue(risks)
        self.assertTrue(opportunities)
        self.assertIsInstance(scorecard["components"], dict)


if __name__ == "__main__":
    unittest.main()
