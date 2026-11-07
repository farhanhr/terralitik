import sys
import unittest
from pathlib import Path
from unittest import mock

import streamlit as st
from streamlit.testing.v1 import AppTest
import xgboost

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "src" / "dashboard" / "app.py"
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


class AppStartupTest(unittest.TestCase):
    def setUp(self):
        st.cache_data.clear()
        st.cache_resource.clear()

    def test_dashboard_starts_without_training(self):
        with mock.patch.object(
            xgboost,
            "train",
            side_effect=AssertionError("dashboard tidak boleh melakukan training"),
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
        self.assertEqual(len(app.error), 0, [e.value for e in app.error])
        self.assertEqual(len(app.tabs), 3)

    def test_dashboard_shows_error_when_model_missing(self):
        with mock.patch(
            "models.drought_forecast.load_model",
            side_effect=FileNotFoundError("Model artifact tidak ditemukan."),
        ):
            app = AppTest.from_file(str(APP_PATH), default_timeout=60).run()

        self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
        self.assertTrue(any("Model prediksi belum siap" in e.value for e in app.error))


if __name__ == "__main__":
    unittest.main()
