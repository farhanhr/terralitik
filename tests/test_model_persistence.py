import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import support
from models import drought_forecast
from xgboost import XGBRegressor


class ModelPersistenceTest(unittest.TestCase):
    def setUp(self):
        self.df = support.make_dataset()
        self.model = drought_forecast.train_model(self.df)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.model_path = Path(self.tmp.name) / "drought_xgb.json"

    def test_training_produces_model_with_expected_feature_order(self):
        self.assertEqual(list(self.model.feature_names_in_), drought_forecast.FEATURES)

    def test_save_and_load_roundtrip_is_consistent(self):
        drought_forecast.save_model(self.model, self.model_path)
        self.assertTrue(self.model_path.exists())

        loaded = drought_forecast.load_model(self.model_path)

        X = self.df[drought_forecast.FEATURES]
        np.testing.assert_allclose(
            self.model.predict(X),
            loaded.predict(X),
            rtol=0,
            atol=1e-6,
        )
        self.assertEqual(list(loaded.feature_names_in_), drought_forecast.FEATURES)

    def test_loading_missing_artifact_raises_informative_error(self):
        missing = Path(self.tmp.name) / "missing.json"
        with self.assertRaises(FileNotFoundError) as context:
            drought_forecast.load_model(missing)
        self.assertIn("train_drought_model.py", str(context.exception))

    def test_loader_does_not_train(self):
        drought_forecast.save_model(self.model, self.model_path)
        with mock.patch.object(
            XGBRegressor, "fit", side_effect=AssertionError("fit tidak boleh dipanggil")
        ):
            loaded = drought_forecast.load_model(self.model_path)
        self.assertIsNotNone(loaded)

    def test_feature_mismatch_artifact_is_rejected(self):
        path = Path(self.tmp.name) / "wrong_features.json"
        wrong = XGBRegressor(n_estimators=2, max_depth=2)
        wrong.fit(self.df[["precipitation", "temp_avg"]], self.df["drought_score"])
        wrong.save_model(path)

        with self.assertRaises(RuntimeError) as context:
            drought_forecast.load_model(path)
        self.assertIn("Urutan fitur", str(context.exception))


if __name__ == "__main__":
    unittest.main()
