# Terralitik — Java Drought Risk EWS

**Terralitik** is a smart, AI-powered dashboard designed to help farmers and local governments in Java, Indonesia, monitor, predict, and mitigate the risks of agricultural drought and crop failure.

- **Interactive Heatmap:** Visualizes the current drought risk level across all regencies and cities in Java.
- **14-Day AI Forecast:** Uses a pre-trained XGBoost model to predict the drought risk score for the next two weeks based on real-time satellite data.
- **Economic Risk Analysis:** Calculates the estimated crop loss percentage and potential financial loss (in Rupiah) per hectare.
- **Mitigation Assistant:** Uses Google Gemini to translate technical risk data into practical Indonesian-language recommendations.

---

## Architecture

Training and inference are deliberately separated:

```text
GitHub Actions (scheduled)
  fetch_weather.py -> feature_engineering.py -> drought_index.py
    -> train_drought_model.py -> artifacts/drought_xgb.json + artifacts/model_manifest.json
    -> commit data + artifacts

Streamlit (when a user opens the dashboard)
  read data/processed/drought_risk.csv
    -> load artifacts/drought_xgb.json   (no training)
    -> forecast_next_days()
    -> build visualizations
```

The dashboard never trains the model. `src/dashboard/app.py` only loads the artifact created by `src/models/train_drought_model.py`.

## Project structure

| Path | Responsibility |
|---|---|
| `src/data_pipeline/fetch_weather.py` | Fetch Open-Meteo data and upsert the raw CSV. |
| `src/analysis/feature_engineering.py` | Add `temp_avg`, `rain_anomaly`, `temp_anomaly`. |
| `src/analysis/drought_index.py` | Compute `drought_score` and `risk_level`. |
| `src/models/train_drought_model.py` | Train, validate, and persist the XGBoost artifact. |
| `src/models/drought_forecast.py` | Model build/load helpers and inference (`forecast_next_days`). |
| `src/models/ai_assistant.py` | Gemini mitigation assistant. |
| `src/dashboard/app.py` | Streamlit dashboard (inference only). |
| `artifacts/drought_xgb.json` | Pre-trained model artifact. |
| `artifacts/model_manifest.json` | Training metadata, metrics, feature order, dataset hash. |
| `.github/workflows/daily_update.yaml` | Scheduled data + model update. |

## How to Run the Project Locally

All commands must be run from the repository root because data paths are root-relative.

### Step 1: Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` contains only the dashboard dependencies. The data pipeline and training use a separate file:

```bash
pip install -r requirements-pipeline.txt
```

### Step 2: Set up the Gemini API key

The AI assistant needs a Google Gemini API key:

1. Get a free API key from [Google AI Studio](https://aistudio.google.com/).
2. Create `.streamlit/secrets.toml` in the project root:

```toml
GEMINI_API_KEY = "put_your_api_key_here"
```

Streamlit also reads `src/dashboard/.streamlit/secrets.toml` (script-level secrets take precedence). Never commit real keys.

### Step 3: Run the dashboard

```bash
streamlit run src/dashboard/app.py
```

The dashboard loads `data/processed/drought_risk.csv` and `artifacts/drought_xgb.json`. It does not train the model.

## Data pipeline (local)

Run in this order:

```bash
python src/data_pipeline/fetch_weather.py    # -> data/raw/weather_jawa_master.csv
python src/analysis/feature_engineering.py   # -> data/processed/climate_features.csv
python src/analysis/drought_index.py         # -> data/processed/drought_risk.csv
```

`fetch_weather.py` reads the GeoJSON centroids, fetches batches of up to 50 locations from Open-Meteo (`past_days=3`, `forecast_days=14`), and upserts on `["date", "location"]`. Do not append to the raw CSV manually.

## Train the model (local)

```bash
python src/models/train_drought_model.py
```

This script:

1. Loads `data/processed/drought_risk.csv` and validates the feature schema.
2. Trains an XGBoost `Booster` (native API, no scikit-learn) with `num_boost_round=300`, `max_depth=6`, `eta=0.05`, `seed=42`, `tree_method="hist"`, `nthread=2`.
3. Computes training metrics plus a time-based holdout evaluation (reporting only; the final model still trains on all rows).
4. Writes `artifacts/drought_xgb.json` and `artifacts/model_manifest.json` atomically.
5. Prints the model version, row count, duration, and metrics.

If the dataset is empty, missing required columns, or contains null features, training fails and the existing artifacts are left untouched.

### Model artifact contents

`artifacts/model_manifest.json` records:

- `model_version`, `trained_at`, `xgboost_version`
- `features` (order matters: `precipitation`, `temp_avg`, `rain_anomaly`, `temp_anomaly`)
- `num_boost_round` and `training_params`
- `n_train_rows`, `n_locations`, `n_dates`
- dataset path, SHA-256 hash, and latest date
- model file SHA-256
- training and time-holdout metrics (`mae`, `rmse`, `r2`)

### If the artifact is missing or corrupted

The dashboard shows an informative error instead of silently training. To repair:

```bash
python src/models/train_drought_model.py
```

Then restart the Streamlit app.

## Automated updates (GitHub Actions)

`.github/workflows/daily_update.yaml` runs daily at `17:00 UTC` and on manual dispatch. It:

1. Installs `requirements-pipeline.txt`.
2. Runs the three data pipeline scripts in order.
3. Runs `python src/models/train_drought_model.py`.
4. Commits `data/raw/`, `data/processed/`, and `artifacts/` together.

The workflow only commits after every stage succeeds. A `concurrency` group prevents overlapping runs.

Streamlit Community Cloud redeploys when the repository changes, but the app only loads the committed model artifact.

## Methodology note

`drought_score` is derived from a weighted formula in `src/analysis/drought_index.py`:

```text
rain_score   = 1 - clip(precipitation / 20, upper=1)
temp_score   = clip((temp_max - 25) / 10, lower=0, upper=1)
drought_score = 0.7 * rain_score + 0.3 * temp_score
risk_level   = Low < 0.5, Moderate < 0.75, High otherwise
```

The XGBoost model learns to estimate this formula-derived index; it is not trained on observed crop-loss ground truth. This is a known methodological limitation. Do not change the formulas as part of performance work; evaluate model quality separately if ground-truth yield data becomes available.
