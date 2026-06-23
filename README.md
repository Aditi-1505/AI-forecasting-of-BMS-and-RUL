# NASA Battery BMS — AI-Driven SOH & RUL Forecasting

An end-to-end Battery Management System that predicts **State of Health (SOH)** and **Remaining Useful Life (RUL)** of NASA 18650 Li-ion cells using a Gradient Boosting model trained on the full NASA Battery Aging dataset.

---

## Dataset

| Split | Folder | Batteries |
|-------|--------|-----------|
| Train | `NASA/Train/` | B0005, B0006, B0025, B0026, B0029, B0030, B0033, B0034, B0038, B0041, B0042, B0043, B0045, B0046, B0047, B0049, B0051, B0053, B0054, B0055 |
| Val   | `NASA/Val/`   | B0018, B0027, B0031, B0039, B0050 |
| Test  | `NASA/Test/`  | B0007, B0028, B0032, B0036, B0040, B0044, B0048, B0052, B0056 |

Every CSV in each folder is loaded automatically — no battery IDs are hardcoded. Just place `*_full.csv` files in the right folder.

---

## Models

| Model | Task | Algorithm |
|-------|------|-----------|
| SOH / Capacity | Predict instantaneous discharge capacity (Ah) | XGBoost / sklearn GBM |
| RUL | Predict cycles remaining until 80% EOL | XGBoost / sklearn GBM |

---

## Features (15 engineered, zero data leakage)

| Feature | Description |
|---------|-------------|
| `cycle_index` | Sequential discharge counter (1-based, per battery) |
| `V_measured` | Mean terminal voltage during discharge (V) |
| `T_measured` | Mean cell temperature during discharge (°C) |
| `ambient_temp` | Test ambient temperature (°C) — critical for 4°C vs 24°C batteries |
| `Re` | Electrolyte resistance (Ω) — grows with SEI layer thickening |
| `Rct` | Charge-transfer resistance (Ω) — grows near end of life |
| `Re_growth` | ΔRe from cycle-1 baseline (Ω) |
| `Rct_growth` | ΔRct from cycle-1 baseline (Ω) |
| `V_drop` | Voltage sag vs cycle-1 baseline (V) |
| `cap_fade_rate` | ΔCapacity from previous cycle (Ah) — causal, no future data |
| `eol_proximity` | `cycle_index / 168` — normalised distance to nominal EOL |
| `Re_x_Rct` | Re × Rct interaction term |
| `Rct_Re_ratio` | Rct / Re ratio |
| `V_Re_interaction` | V × Re coupling term |
| `cycle_sq` | `cycle_index²` — non-linear ageing acceleration |

> `eol_proximity` is derived from cycle index only — never from the capacity target (no data leakage). `coulombic_efficiency = cap / rated` was explicitly removed as it is a linear transform of the target.

---

## Project Structure

```
.
├── app.py                  # Flask REST API
├── train_model.py          # SOH/capacity model training
├── train_rul_model.py      # RUL model training
├── data_loader.py          # CSV loading, feature engineering, synthetic fallback
├── config.py               # All constants (battery physics, model params, paths)
├── charts.py               # Matplotlib chart generators → base64 PNG
├── logger.py               # File + stdout logging
├── test_model.py           # 46 unit & integration tests
├── templates/
│   └── index.html          # Dashboard frontend
├── NASA/
│   ├── Train/              # Training battery CSVs
│   ├── Val/                # Validation battery CSVs
│   └── Test/               # Test battery CSVs
├── model/
│   ├── xgb_battery_model.pkl   # Trained SOH model
│   ├── xgb_rul_model.pkl       # Trained RUL model
│   └── model_meta.json         # Metrics, feature importances, predictions
└── logs/                   # Auto-generated log files
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

XGBoost is strongly recommended but optional — sklearn `GradientBoostingRegressor` is used automatically if XGBoost is not installed:
```bash
pip install xgboost
```

### 2. Place NASA CSVs
```bash
mkdir -p NASA/Train NASA/Val NASA/Test
# Copy all *_full.csv files to their respective folders
```

### 3. Train
```bash
python train_model.py          # SOH/capacity model → model/xgb_battery_model.pkl
python train_rul_model.py      # RUL model          → model/xgb_rul_model.pkl
```

Force synthetic data (no CSVs needed):
```bash
python train_model.py --synthetic
```

### 4. Run tests
```bash
python test_model.py           # 46 tests; CSV tests auto-enable when data is present
```

### 5. Launch dashboard
```bash
python app.py                  # → http://localhost:5002
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Dashboard HTML |
| `POST` | `/api/predict` | SOH + RUL prediction |
| `GET`  | `/api/model-info` | Train / Val / Test metrics + model config |
| `GET`  | `/api/feature-importance` | Ranked feature importances |
| `GET`  | `/api/dataset-stats` | Per-battery stats for all three splits |
| `POST` | `/api/charts` | Matplotlib charts as base64 PNG |
| `POST` | `/api/retrain` | Retrain model on full dataset |
| `GET`  | `/api/battery-profile/<id>` | Capacity curve for one battery |

### `/api/predict` — request body

```json
{
  "cycle_index": 50,
  "V_measured": 4.10,
  "T_measured": 24.0,
  "ambient_temp": 24.0,
  "Re": 0.052,
  "Rct": 0.078,
  "Re_growth": 0.008,
  "Rct_growth": 0.009,
  "V_drop": 0.091,
  "cap_fade_rate": -0.001,
  "eol_proximity": 0.298,
  "Re_x_Rct": 0.004056,
  "Rct_Re_ratio": 1.5,
  "V_Re_interaction": 0.213,
  "cycle_sq": 2500
}
```

### `/api/predict` — response

```json
{
  "status": "ok",
  "predicted_capacity_ah": 1.812,
  "raw_capacity_ah": 1.945,
  "soh_pct": 90.6,
  "rul_cycles": 284,
  "days_remaining": 284.0,
  "health_grade": "Excellent",
  "temp_factor": 1.0,
  "cycle_factor": 0.99,
  "at_eol": false,
  "near_eol": false,
  "dod_vs_eol": "Normal operation — battery is healthy"
}
```

---

## Dashboard

The web dashboard at `http://localhost:5002` provides:

- **Cycle Measurements** — voltage, temperature, impedance (Re, Rct) sliders
- **Derived State Features** — Re/Rct growth, voltage drop, EOL proximity, capacity fade rate
- **Auto-fill Helpers** — pre-fills inputs with typical fresh/aged/EOL battery values
- **Prediction Results** — predicted capacity (Ah), SOH (%), RUL (cycles), health grade
- **Visual Analysis** — SOH degradation curve, discharge profile, feature importance chart, temperature derating curve, actual vs predicted scatter plot

---

## Key Design Decisions

**No hardcoded battery IDs** — every `*.csv` in each folder is loaded automatically. Add or remove batteries by moving files.

**Complex Re/Rct strings** — some batteries store impedance measurements as complex numbers e.g. `(0.048+0j)`. The loader extracts the real part automatically.

**Minimum capacity filter** — cycles with capacity ≤ 0.01 Ah are dropped (removes zero/near-zero outliers). All other test protocols including low-current 4°C pulse tests are preserved.

**ambient_temp as a feature** — batteries tested at 4°C vs 24°C differ significantly in measured capacity. Including ambient temperature allows the model to generalise across test conditions.

**3-way split reporting** — Train, Val, and Test R²/MAE are all logged and saved to `model_meta.json`. The Actual vs Predicted chart uses val split predictions so test data is never exposed to the visualisation layer.

**Synthetic fallback** — if any folder is empty or missing, that split is replaced with physics-informed synthetic data so the pipeline always runs end-to-end.

---

## Requirements

```
flask>=3.0
flask-cors>=4.0
numpy>=1.24
pandas>=2.0
scikit-learn>=1.4
matplotlib>=3.7
seaborn>=0.13
xgboost>=2.0   # optional but recommended
```
