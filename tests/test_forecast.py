import unittest

import pandas as pd

from tests import support
from models.drought_forecast import FEATURES, crop_failure_risk, forecast_next_days, train_model


class ForecastTest(unittest.TestCase):
    def setUp(self):
        self.df = support.make_dataset()
        self.model = train_model(self.df)

    def test_forecast_returns_future_rows_only(self):
        future = support.make_future_df(days=3, location="Alpha")
        past = support.make_future_df(
            days=2, location="Alpha", start=pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
        )
        df = pd.concat([past, self.df, future], ignore_index=True)

        dates, scores = forecast_next_days(self.model, df, "Alpha")

        self.assertEqual(len(dates), 3)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))
        self.assertEqual(dates, sorted(dates))

    def test_no_future_rows_returns_empty(self):
        dates, scores = forecast_next_days(self.model, self.df, "Alpha")
        self.assertEqual(dates, [])
        self.assertEqual(scores, [])

    def test_missing_feature_raises(self):
        future = support.make_future_df(days=2, location="Alpha").drop(columns=["temp_avg"])
        df = pd.concat([self.df, future], ignore_index=True)

        with self.assertRaises(ValueError) as context:
            forecast_next_days(self.model, df, "Alpha")
        self.assertIn("temp_avg", str(context.exception))

    def test_null_feature_raises(self):
        future = support.make_future_df(days=2, location="Alpha")
        future.loc[0, "rain_anomaly"] = None
        df = pd.concat([self.df, future], ignore_index=True)

        with self.assertRaises(ValueError) as context:
            forecast_next_days(self.model, df, "Alpha")
        self.assertIn("rain_anomaly", str(context.exception))

    def test_predictions_are_clipped(self):
        class StubModel:
            def predict(self, frame):
                self.frame = frame
                return [-0.5, 2.0]

        future = support.make_future_df(days=2, location="Alpha")
        df = pd.concat([self.df, future], ignore_index=True)

        dates, scores = forecast_next_days(StubModel(), df, "Alpha")

        self.assertEqual(scores, [0.0, 1.0])
        self.assertEqual(len(dates), 2)

    def test_crop_failure_risk_thresholds_unchanged(self):
        self.assertEqual(crop_failure_risk(0.49), "Low")
        self.assertEqual(crop_failure_risk(0.5), "Moderate")
        self.assertEqual(crop_failure_risk(0.74), "Moderate")
        self.assertEqual(crop_failure_risk(0.75), "High")


if __name__ == "__main__":
    unittest.main()
