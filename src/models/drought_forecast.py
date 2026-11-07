import json
import os
from pathlib import Path

import pandas as pd
from xgboost import XGBRegressor

FEATURES = ["precipitation", "temp_avg", "rain_anomaly", "temp_anomaly"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "artifacts" / "drought_xgb.json"
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "model_manifest.json"


def build_model():
    return XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
        tree_method="hist",
        n_jobs=2,
    )


def validate_feature_schema(df, require_target=False, allow_nulls=False):
    required = list(FEATURES)
    if require_target:
        required.append("drought_score")

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(
            "Kolom fitur tidak lengkap: "
            + ", ".join(missing)
            + f". Diperlukan: {required}."
        )

    if allow_nulls:
        return

    null_columns = [column for column in FEATURES if df[column].isna().any()]
    if null_columns:
        raise ValueError(
            "Fitur mengandung nilai kosong: " + ", ".join(null_columns) + "."
        )


def train_model(df):
    validate_feature_schema(df, require_target=True, allow_nulls=True)

    train_df = df.dropna(subset=FEATURES + ["drought_score"])
    if train_df.empty:
        raise ValueError("Tidak ada baris valid untuk training setelah dropna().")

    model = build_model()
    model.fit(train_df[FEATURES], train_df["drought_score"])

    return model


def save_model(model, path=DEFAULT_MODEL_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp.json")
    model.save_model(tmp_path)
    os.replace(tmp_path, path)
    return path


def load_model(path=DEFAULT_MODEL_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact tidak ditemukan di {path}. "
            "Jalankan `python src/models/train_drought_model.py` untuk membuatnya."
        )

    model = XGBRegressor()
    try:
        model.load_model(path)
    except Exception as exc:  # corrupted or incompatible artifact
        raise RuntimeError(
            f"Gagal memuat model dari {path}: {exc}. "
            "Latih ulang model dengan `python src/models/train_drought_model.py`."
        ) from exc

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is not None and list(feature_names) != FEATURES:
        raise RuntimeError(
            "Urutan fitur pada artifact tidak cocok dengan FEATURES. "
            f"Artifact: {list(feature_names)}; kode: {FEATURES}. "
            "Latih ulang model dengan `python src/models/train_drought_model.py`."
        )

    return model


def save_manifest(manifest, path=DEFAULT_MANIFEST_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)
    return path


def load_manifest(path=DEFAULT_MANIFEST_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Manifest model tidak ditemukan di {path}. "
            "Jalankan `python src/models/train_drought_model.py` untuk membuatnya."
        )
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Manifest model rusak di {path}: {exc}. Latih ulang model."
        ) from exc


def forecast_next_days(model, df, location):
    loc_df = df[df["location"] == location].copy()
    loc_df["date"] = pd.to_datetime(loc_df["date"])

    tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)

    future_df = loc_df[loc_df["date"] >= tomorrow].copy()

    if future_df.empty:
        return [], []

    validate_feature_schema(future_df)

    predictions = model.predict(future_df[FEATURES])

    forecast_scores = [max(0.0, min(1.0, float(pred))) for pred in predictions]
    forecast_dates = future_df["date"].dt.strftime("%Y-%m-%d").tolist()

    return forecast_dates, forecast_scores


def crop_failure_risk(score):
    if score < 0.5:
        return "Low"
    elif score < 0.75:
        return "Moderate"
    else:
        return "High"
