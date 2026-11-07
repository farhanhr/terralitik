import json
import tempfile
import unittest
from pathlib import Path

from tests import support
from models import train_drought_model


class TrainingPipelineTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data_path = Path(self.tmp.name) / "drought_risk.csv"
        self.model_path = Path(self.tmp.name) / "artifacts" / "drought_xgb.json"
        self.manifest_path = Path(self.tmp.name) / "artifacts" / "model_manifest.json"

        self.df = support.make_dataset(n_dates=60)
        self.df.to_csv(self.data_path, index=False)

    def test_train_and_save_writes_model_and_manifest(self):
        manifest, model = train_drought_model.train_and_save(
            self.data_path, self.model_path, self.manifest_path
        )

        self.assertTrue(self.model_path.exists())
        self.assertTrue(self.manifest_path.exists())

        on_disk = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, manifest)
        self.assertEqual(manifest["features"], ["precipitation", "temp_avg", "rain_anomaly", "temp_anomaly"])
        self.assertEqual(manifest["n_train_rows"], len(self.df))
        self.assertEqual(manifest["n_locations"], 3)
        self.assertIn("sha256", manifest["dataset"])
        self.assertIn("xgboost_version", manifest)
        self.assertIn("mae", manifest["metrics"]["train"])

    def test_invalid_dataset_fails_without_overwriting_artifacts(self):
        train_drought_model.train_and_save(
            self.data_path, self.model_path, self.manifest_path
        )
        original_model = self.model_path.read_bytes()
        original_manifest = self.manifest_path.read_bytes()

        bad_data = Path(self.tmp.name) / "bad.csv"
        bad_data.write_text("date,location\n", encoding="utf-8")

        with self.assertRaises(ValueError):
            train_drought_model.train_and_save(
                bad_data, self.model_path, self.manifest_path
            )

        self.assertEqual(self.model_path.read_bytes(), original_model)
        self.assertEqual(self.manifest_path.read_bytes(), original_manifest)

    def test_empty_dataset_fails(self):
        empty = Path(self.tmp.name) / "empty.csv"
        empty.write_text("", encoding="utf-8")
        with self.assertRaises(ValueError):
            train_drought_model.load_training_data(empty)

    def test_missing_dataset_fails(self):
        missing = Path(self.tmp.name) / "missing.csv"
        with self.assertRaises(FileNotFoundError):
            train_drought_model.load_training_data(missing)

    def test_nan_feature_rows_are_dropped(self):
        self.df.loc[0, "temp_avg"] = None
        self.df.to_csv(self.data_path, index=False)

        manifest, _ = train_drought_model.train_and_save(
            self.data_path, self.model_path, self.manifest_path
        )
        self.assertEqual(manifest["n_train_rows"], len(self.df) - 1)


if __name__ == "__main__":
    unittest.main()
