import argparse
import hashlib
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import xgboost

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.drought_forecast import (  # noqa: E402
    DEFAULT_MANIFEST_PATH,
    DEFAULT_MODEL_PATH,
    FEATURES,
    PROJECT_ROOT,
    build_model,
    save_manifest,
    save_model,
    validate_feature_schema,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "drought_risk.csv"
MIN_TRAIN_ROWS = 100


def load_training_data(path=DATA_PATH):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset training tidak ditemukan di {path}. "
            "Jalankan pipeline data terlebih dahulu."
        )

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Dataset training kosong: {path}.")

    validate_feature_schema(df, require_target=True, allow_nulls=True)
    if "location" not in df.columns or "date" not in df.columns:
        raise ValueError(
            "Dataset training harus memiliki kolom 'location' dan 'date'."
        )

    return df


def dataset_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_sha256(path):
    return dataset_sha256(path)


def _metrics(y_true, y_pred):
    y_true = pd.Series(y_true).reset_index(drop=True)
    y_pred = pd.Series(y_pred).reset_index(drop=True)

    residual = y_true - y_pred
    mae = float(residual.abs().mean())
    rmse = float((residual**2).mean() ** 0.5)
    ss_res = float((residual**2).sum())
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot else 0.0

    return {"mae": round(mae, 6), "rmse": round(rmse, 6), "r2": round(r2, 6)}


def evaluate_time_holdout(df):
    dates = sorted(pd.to_datetime(df["date"]).dt.normalize().unique())
    if len(dates) < 5:
        return None

    cutoff = dates[int(len(dates) * 0.8)]
    train_df = df[pd.to_datetime(df["date"]).dt.normalize() < cutoff]
    holdout_df = df[pd.to_datetime(df["date"]).dt.normalize() >= cutoff]

    if train_df.empty or holdout_df.empty:
        return None

    eval_model = build_model()
    eval_model.fit(train_df[FEATURES], train_df["drought_score"])
    predictions = eval_model.predict(holdout_df[FEATURES])

    result = _metrics(holdout_df["drought_score"], predictions)
    result["cutoff_date"] = pd.Timestamp(cutoff).strftime("%Y-%m-%d")
    result["n_rows"] = int(len(holdout_df))
    return result


def train_and_save(
    data_path=DATA_PATH,
    model_path=DEFAULT_MODEL_PATH,
    manifest_path=DEFAULT_MANIFEST_PATH,
):
    data_path = Path(data_path)
    model_path = Path(model_path)
    manifest_path = Path(manifest_path)

    started = time.perf_counter()
    df = load_training_data(data_path)
    train_df = df.dropna(subset=FEATURES + ["drought_score"])

    if len(train_df) < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Data training valid hanya {len(train_df)} baris; "
            f"minimal {MIN_TRAIN_ROWS} baris diperlukan."
        )

    model = build_model()
    train_started = time.perf_counter()
    model.fit(train_df[FEATURES], train_df["drought_score"])
    train_seconds = time.perf_counter() - train_started

    predictions = model.predict(train_df[FEATURES])
    if pd.isna(predictions).any():
        raise ValueError("Model menghasilkan prediksi NaN; artifact tidak disimpan.")

    holdout = evaluate_time_holdout(train_df)
    train_metrics = _metrics(train_df["drought_score"], predictions)

    trained_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    dataset_hash = dataset_sha256(data_path)

    save_model(model, model_path)
    manifest = {
        "schema_version": 1,
        "model_version": f"{trained_at}-{dataset_hash[:12]}",
        "trained_at": trained_at,
        "xgboost_version": xgboost.__version__,
        "features": FEATURES,
        "n_train_rows": int(len(train_df)),
        "n_locations": int(train_df["location"].nunique()),
        "n_dates": int(pd.to_datetime(train_df["date"]).dt.normalize().nunique()),
        "dataset": {
            "path": str(data_path.relative_to(PROJECT_ROOT))
            if data_path.is_relative_to(PROJECT_ROOT)
            else str(data_path),
            "sha256": dataset_hash,
            "max_date": str(pd.to_datetime(train_df["date"]).max().date()),
        },
        "model_file": str(model_path.relative_to(PROJECT_ROOT))
        if model_path.is_relative_to(PROJECT_ROOT)
        else str(model_path),
        "model_sha256": file_sha256(model_path),
        "metrics": {
            "train": train_metrics,
            "time_holdout": holdout,
        },
        "training_seconds": round(time.perf_counter() - started, 3),
        "fit_seconds": round(train_seconds, 3),
    }
    save_manifest(manifest, manifest_path)

    return manifest, model


def main():
    parser = argparse.ArgumentParser(description="Latih dan simpan model drought Terralitik.")
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    args = parser.parse_args()

    manifest, _ = train_and_save(args.data, args.model, args.manifest)

    print("Training selesai.")
    print(f"  model_version : {manifest['model_version']}")
    print(f"  baris training: {manifest['n_train_rows']}")
    print(f"  wilayah       : {manifest['n_locations']}")
    print(f"  fit_seconds   : {manifest['fit_seconds']}")
    print(f"  total_seconds : {manifest['training_seconds']}")
    print(f"  metrik train  : {manifest['metrics']['train']}")
    print(f"  time_holdout  : {manifest['metrics']['time_holdout']}")
    print(f"  model artifact: {args.model}")
    print(f"  manifest      : {args.manifest}")


if __name__ == "__main__":
    main()
