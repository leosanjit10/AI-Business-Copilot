"""Focused tests for reusable health, anomaly, and segmentation analytics."""

import unittest

import pandas as pd

from utils.anomaly_detection import detect_sales_anomalies
from utils.business_health import calculate_business_health_score
from utils.customer_segmentation import build_customer_segments


class BusinessHealthTests(unittest.TestCase):
    def test_health_score_is_bounded_and_reports_all_four_components(self) -> None:
        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        sales = pd.DataFrame(
            {
                "Date": dates,
                "Product": ["A", "B"] * 5,
                "Quantity": [1] * 10,
                "Revenue": [100, 100, 100, 100, 100, 100, 120, 120, 120, 120],
            }
        )
        forecast = pd.DataFrame(
            {"ds": pd.date_range("2026-01-11", periods=5), "yhat": [130] * 5}
        )

        result = calculate_business_health_score(sales, forecast)

        self.assertGreaterEqual(result["overall_score"], 0)
        self.assertLessEqual(result["overall_score"], 100)
        self.assertIn(result["category"], {"Excellent", "Healthy", "Average", "At Risk"})
        self.assertEqual(len(result["component_scores"]), 4)
        self.assertTrue(result["strengths"])
        self.assertTrue(result["recommendations"])

    def test_empty_health_data_has_actionable_fallback(self) -> None:
        result = calculate_business_health_score(pd.DataFrame(), None)
        self.assertEqual(result["overall_score"], 0)
        self.assertEqual(result["category"], "At Risk")
        self.assertTrue(result["risks"])


class AnomalyDetectionTests(unittest.TestCase):
    def test_isolation_forest_finds_a_revenue_spike(self) -> None:
        dates = pd.date_range("2026-02-01", periods=12, freq="D")
        sales = pd.DataFrame(
            {
                "Date": dates,
                "Product": ["Product A"] * 12,
                "Quantity": [2] * 11 + [30],
                "Revenue": [100] * 11 + [2500],
            }
        )

        anomalies = detect_sales_anomalies(sales)

        self.assertIn("Revenue spike", set(anomalies["Category"]))
        self.assertIn("2026-02-12", set(anomalies["Date"]))

    def test_short_history_uses_z_score_fallback(self) -> None:
        sales = pd.DataFrame(
            {
                "Date": pd.date_range("2026-03-01", periods=6, freq="D"),
                "Product": ["Product A"] * 6,
                "Quantity": [1, 1, 1, 1, 1, 20],
                "Revenue": [100, 99, 101, 98, 102, 900],
            }
        )

        anomalies = detect_sales_anomalies(sales)

        self.assertIn("Revenue spike", set(anomalies["Category"]))
        self.assertIn("Z-score fallback", set(anomalies["Detection Method"]))

    def test_duplicate_source_rows_are_reported(self) -> None:
        sales = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-04-01", "2026-04-01"]),
                "Product": ["A", "A"],
                "Quantity": [1, 1],
                "Revenue": [10, 10],
            }
        )

        anomalies = detect_sales_anomalies(sales, sales)

        self.assertIn("Data quality anomaly", set(anomalies["Category"]))
        self.assertIn("Duplicate rows", set(anomalies["Metric"]))


class CustomerSegmentationTests(unittest.TestCase):
    def test_customer_id_enables_kmeans_segment_profiles(self) -> None:
        rows = []
        start = pd.Timestamp("2026-01-01")
        for customer_index in range(12):
            for purchase_index in range(2 + customer_index % 3):
                rows.append(
                    {
                        "Date": start + pd.Timedelta(days=customer_index * 4 + purchase_index),
                        "CustomerID": f"C{customer_index:02d}",
                        "Product": "Product A",
                        "Quantity": 1 + customer_index % 4,
                        "Revenue": 20 + customer_index * 15 + purchase_index * 3,
                    }
                )
        sales = pd.DataFrame(rows)

        result = build_customer_segments(sales)

        self.assertTrue(result["available"])
        self.assertGreaterEqual(result["cluster_count"], 2)
        self.assertIn("VIP Customers", set(result["customers"]["Segment"]))
        self.assertIn("Revenue Contribution %", result["summary"].columns)
        self.assertIn("Retention Indicator", result["summary"].columns)

    def test_sales_without_customer_id_explain_segmentation_limitation(self) -> None:
        sales = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-05-01"]),
                "Product": ["A"],
                "Quantity": [1],
                "Revenue": [20],
            }
        )

        result = build_customer_segments(sales)

        self.assertFalse(result["available"])
        self.assertIn("customer identifier", result["message"])

    def test_identical_customer_features_do_not_claim_clusters(self) -> None:
        sales = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2026-06-01"] * 3),
                "CustomerID": ["A", "B", "C"],
                "Product": ["A", "A", "A"],
                "Quantity": [1, 1, 1],
                "Revenue": [50, 50, 50],
            }
        )

        result = build_customer_segments(sales)

        self.assertFalse(result["available"])
        self.assertIn("identical", result["message"])


if __name__ == "__main__":
    unittest.main()
