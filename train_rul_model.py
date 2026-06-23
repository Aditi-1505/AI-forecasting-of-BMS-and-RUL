"""
train_rul_model.py
------------------
Trains a dedicated RUL (Remaining Useful Life) model on physics-informed
synthetic multi-cycle aging trajectories calibrated to the NASA 18650 cell.

Why synthetic data for RUL?
  Each NASA CSV contains a single discharge-capacity value per cycle but
  no end-of-life label (the batteries in B0005–B0018 are tested to ~168
  discharge cycles, not run to true EOL). The synthetic simulator generates
  full aging curves with proper RUL labels using the same two-regime
  capacity-fade physics as the rest of the app.

Output: model/xgb_rul_model.pkl
"""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import MODEL_CFG, BATTERY
from data_loader import make_synthetic_rul_df
from logger import get_logger

log = get_logger(__name__)

try:
    from xgboost import XGBRegressor as _XGB
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

RUL_FEATURES = [
    "cycle_index",
    "soh_pct",
    "capacity_ah",
    "avg_temp_C",
    "avg_dod",
    "cycle_rate",
    "deep_rate",
]
RUL_TARGET = "RUL"


def train_rul(n_cells: int = 300, seed: int = 0) -> dict:
    log.info("Generating physics-informed RUL training data (%d simulated cells) …", n_cells)
    df = make_synthetic_rul_df(n_cells=n_cells, seed=seed)
    log.info("RUL dataset: %d rows | RUL range %.0f–%.0f cycles",
             len(df), df[RUL_TARGET].min(), df[RUL_TARGET].max())

    X = df[RUL_FEATURES].values
    y = df[RUL_TARGET].values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=42)

    if _HAS_XGB:
        reg = _XGB(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.80,
            reg_alpha=0.1, reg_lambda=1.5,
            objective="reg:squarederror", eval_metric="rmse",
            random_state=42, n_jobs=-1, verbosity=0, tree_method="hist",
        )
    else:
        reg = GradientBoostingRegressor(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, min_samples_leaf=5, random_state=42,
        )

    pipe = Pipeline([("scaler", StandardScaler()), ("gbm", reg)])

    log.info("Fitting RUL model …")
    t0 = time.time()
    pipe.fit(X_tr, y_tr)
    elapsed = round(time.time() - t0, 2)

    y_pred = pipe.predict(X_te)
    r2  = round(float(r2_score(y_te, y_pred)), 4)
    mae = round(float(mean_absolute_error(y_te, y_pred)), 1)
    rmse = round(float(np.sqrt(mean_squared_error(y_te, y_pred))), 1)

    log.info("RUL model — Test R²=%.4f  MAE=%.1f cycles  RMSE=%.1f cycles", r2, mae, rmse)

    os.makedirs(MODEL_CFG.model_dir, exist_ok=True)
    with open(MODEL_CFG.rul_path, "wb") as f:
        pickle.dump(pipe, f)

    fi_arr = pipe.named_steps["gbm"].feature_importances_
    fi = {f: round(float(v), 6) for f, v in zip(RUL_FEATURES, fi_arr)}
    log.info("RUL feature importances: %s", {k: round(v, 3) for k, v in sorted(fi.items(), key=lambda x: -x[1])})
    log.info("RUL model saved → %s  (%.2f s)", MODEL_CFG.rul_path, elapsed)

    return {"r2": r2, "mae": mae, "rmse": rmse, "features": RUL_FEATURES, "fi": fi}


def load_rul_model() -> Pipeline:
    if not os.path.exists(MODEL_CFG.rul_path):
        raise FileNotFoundError(f"No RUL model at '{MODEL_CFG.rul_path}'. Run train_rul_model.py first.")
    with open(MODEL_CFG.rul_path, "rb") as f:
        pipe = pickle.load(f)
    log.info("RUL model loaded from %s", MODEL_CFG.rul_path)
    return pipe


if __name__ == "__main__":
    train_rul()