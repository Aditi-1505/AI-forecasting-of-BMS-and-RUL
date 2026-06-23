"""
train_model.py
--------------
Trains an XGBoost-style Gradient Boosting Regressor on the full NASA
Battery Aging dataset (all batteries in all three split folders).

3-way split:
  Train : NASA/Train/  — model fitting
  Val   : NASA/Val/    — reported during training (not used for early-stop)
  Test  : NASA/Test/   — final held-out generalisation score

All batteries in each folder are loaded automatically.

Usage
-----
    python train_model.py
    python train_model.py --train NASA/Train --val NASA/Val --test NASA/Test
    python train_model.py --synthetic
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import DATA_CFG, MODEL_CFG
from data_loader import FEATURES, TARGET, load_dataset, make_synthetic_df
from logger import get_logger

log = get_logger(__name__)

try:
    from xgboost import XGBRegressor as _XGB
    _HAS_XGB = True
    log.info("XGBoost found — using XGBRegressor")
except ImportError:
    _HAS_XGB = False
    log.info("XGBoost not installed — using sklearn GradientBoostingRegressor")


def _make_regressor(
    n_estimators:  int   = MODEL_CFG.n_estimators,
    max_depth:     int   = MODEL_CFG.max_depth,
    learning_rate: float = MODEL_CFG.learning_rate,
    subsample:     float = MODEL_CFG.subsample,
    random_state:  int   = MODEL_CFG.random_state,
) -> object:
    """Build an XGBRegressor (preferred) or sklearn GradientBoostingRegressor."""
    cfg = dict(
        n_estimators  = n_estimators,
        max_depth     = max_depth,
        learning_rate = learning_rate,
        subsample     = subsample,
        random_state  = random_state,
    )
    if _HAS_XGB:
        return _XGB(
            **cfg,
            colsample_bytree = 0.80,
            reg_alpha        = 0.1,
            reg_lambda       = 1.5,
            objective        = "reg:squarederror",
            eval_metric      = "rmse",
            n_jobs           = -1,
            verbosity        = 0,
            tree_method      = "hist",
        )
    return GradientBoostingRegressor(
        **cfg,
        min_samples_leaf = MODEL_CFG.min_samples_leaf,
    )


def build_pipeline(
    n_estimators:  int   = MODEL_CFG.n_estimators,
    max_depth:     int   = MODEL_CFG.max_depth,
    learning_rate: float = MODEL_CFG.learning_rate,
    subsample:     float = MODEL_CFG.subsample,
    random_state:  int   = MODEL_CFG.random_state,
) -> Pipeline:
    """StandardScaler → GBM regressor."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gbm", _make_regressor(
            n_estimators  = n_estimators,
            max_depth     = max_depth,
            learning_rate = learning_rate,
            subsample     = subsample,
            random_state  = random_state,
        )),
    ])


def _metrics(pipe: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    yp   = pipe.predict(X)
    r2   = round(float(r2_score(y, yp)),                    4)
    mae  = round(float(mean_absolute_error(y, yp)),         4)
    rmse = round(float(np.sqrt(mean_squared_error(y, yp))), 4)
    mse  = round(float(mean_squared_error(y, yp)),          6)
    return {"r2": r2, "mae": mae, "rmse": rmse, "mse": mse,
            "y_true": y.tolist(), "y_pred": yp.tolist()}


def train(
    train_folder:    str  = DATA_CFG.train_folder,
    val_folder:      str  = DATA_CFG.val_folder,
    test_folder:     str  = DATA_CFG.test_folder,
    force_synthetic: bool = False,
) -> dict:
    """
    Load all batteries from the three folders, fit on train,
    evaluate on val + test, persist model + metadata.
    """
    # ── 1. Data ───────────────────────────────────────────────────────────────
    if force_synthetic:
        log.info("Force-synthetic mode — ignoring real CSVs")
        train_df = make_synthetic_df(n=DATA_CFG.synthetic_train_n, seed=DATA_CFG.synthetic_seed_train)
        val_df   = make_synthetic_df(n=DATA_CFG.synthetic_val_n,   seed=DATA_CFG.synthetic_seed_val)
        test_df  = make_synthetic_df(n=DATA_CFG.synthetic_test_n,  seed=DATA_CFG.synthetic_seed_test)
        dataset_label = "NASA Battery Aging (synthetic fallback)"
    else:
        train_df, val_df, test_df = load_dataset(train_folder, val_folder, test_folder)
        bat_tr = sorted(train_df["battery_id"].unique().tolist())
        bat_va = sorted(val_df["battery_id"].unique().tolist())
        bat_te = sorted(test_df["battery_id"].unique().tolist())
        dataset_label = (
            f"NASA Battery Aging | "
            f"Train={bat_tr} | Val={bat_va} | Test={bat_te}"
        )

    bat_tr = sorted(train_df["battery_id"].unique().tolist())
    bat_va = sorted(val_df["battery_id"].unique().tolist())
    bat_te = sorted(test_df["battery_id"].unique().tolist())

    X_tr = train_df[FEATURES].values;  y_tr = train_df[TARGET].values
    X_va = val_df[FEATURES].values;    y_va = val_df[TARGET].values
    X_te = test_df[FEATURES].values;   y_te = test_df[TARGET].values

    log.info("Samples  — Train: %d | Val: %d | Test: %d", len(X_tr), len(X_va), len(X_te))
    log.info("Features (%d): %s", len(FEATURES), FEATURES)

    # ── 2. Fit ────────────────────────────────────────────────────────────────
    log.info("Fitting GBM pipeline …")
    pipe = build_pipeline()
    t0   = time.time()
    pipe.fit(X_tr, y_tr)
    elapsed = round(time.time() - t0, 2)
    log.info("Training complete in %.2f s", elapsed)

    # ── 3. Evaluate all three splits ──────────────────────────────────────────
    sep = "─" * 64
    log.info(sep)
    tr_m = _metrics(pipe, X_tr, y_tr)
    va_m = _metrics(pipe, X_va, y_va)
    te_m = _metrics(pipe, X_te, y_te)
    log.info("  Train  R²=%.4f  MAE=%.4f Ah  RMSE=%.4f Ah", tr_m["r2"], tr_m["mae"], tr_m["rmse"])
    log.info("  Val    R²=%.4f  MAE=%.4f Ah  RMSE=%.4f Ah", va_m["r2"], va_m["mae"], va_m["rmse"])
    log.info("  Test   R²=%.4f  MAE=%.4f Ah  RMSE=%.4f Ah  ← cross-battery generalisation",
             te_m["r2"], te_m["mae"], te_m["rmse"])
    log.info(sep)

    # ── 4. Feature importances ────────────────────────────────────────────────
    gbm = pipe.named_steps["gbm"]
    fi  = {f: round(float(v), 6) for f, v in zip(FEATURES, gbm.feature_importances_)}
    log.info("Feature importances (top 10):")
    for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]:
        log.info("  %-28s  %.4f  %s", feat, imp, "█" * int(imp * 50))

    # ── 5. Persist ────────────────────────────────────────────────────────────
    os.makedirs(MODEL_CFG.model_dir, exist_ok=True)
    with open(MODEL_CFG.model_path, "wb") as f:
        pickle.dump(pipe, f)

    metrics = {
        "model":           "GradientBoostingRegressor (XGBoost-style)",
        "backend":         "xgboost" if _HAS_XGB else "sklearn",
        "dataset":         dataset_label,
        "train_batteries": bat_tr,  "val_batteries": bat_va,  "test_batteries": bat_te,
        "train_n":         int(len(X_tr)),
        "val_n":           int(len(X_va)),
        "test_n":          int(len(X_te)),
        # Per-split metrics
        "train_r2":  tr_m["r2"],  "train_mae":  tr_m["mae"],  "train_rmse": tr_m["rmse"],
        "val_r2":    va_m["r2"],  "val_mae":    va_m["mae"],  "val_rmse":   va_m["rmse"],
        "test_r2":   te_m["r2"],  "test_mae":   te_m["mae"],  "test_rmse":  te_m["rmse"],
        # Primary aliases (test split) used by app.py / dashboard
        "r2":    te_m["r2"],
        "mae":   te_m["mae"],
        "rmse":  te_m["rmse"],
        "mse":   te_m["mse"],
        # Model config
        "n_estimators":    int(getattr(gbm, "n_estimators", MODEL_CFG.n_estimators)),
        "max_depth":       int(getattr(gbm, "max_depth",    MODEL_CFG.max_depth)),
        "learning_rate":   float(getattr(gbm, "learning_rate", MODEL_CFG.learning_rate)),
        "training_time_s": elapsed,
        "fi":       fi,
        "features": FEATURES,
        "target":   TARGET,
        # Val predictions stored for dashboard charts (test data never exposed)
        "y_true": va_m["y_true"],
        "y_pred": va_m["y_pred"],
    }

    # Save everything including y_true/y_pred so charts work after server restart
    with open(MODEL_CFG.meta_path, "w") as f:
        json.dump(metrics, f, indent=2)

    log.info("Model   saved → %s", MODEL_CFG.model_path)
    log.info("Metrics saved → %s", MODEL_CFG.meta_path)
    return metrics


def load_model() -> tuple[Pipeline, dict]:
    if not os.path.exists(MODEL_CFG.model_path):
        raise FileNotFoundError(
            f"No saved model at '{MODEL_CFG.model_path}'.\n"
            "Run `python train_model.py` first."
        )
    with open(MODEL_CFG.model_path, "rb") as f:
        pipe = pickle.load(f)
    with open(MODEL_CFG.meta_path, "r") as f:
        meta = json.load(f)
    log.info(
        "Loaded model from %s  (Test R²=%.4f  Val R²=%.4f  RMSE=%.4f Ah)",
        MODEL_CFG.model_path, meta["r2"], meta.get("val_r2", 0), meta.get("rmse", 0),
    )
    return pipe, meta


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train SOH/capacity model — full NASA dataset")
    p.add_argument("--train",     default=DATA_CFG.train_folder)
    p.add_argument("--val",       default=DATA_CFG.val_folder)
    p.add_argument("--test",      default=DATA_CFG.test_folder)
    p.add_argument("--synthetic", action="store_true",
                   help="Force synthetic data (ignore real CSVs)")
    args = p.parse_args()
    train(
        train_folder    = args.train,
        val_folder      = args.val,
        test_folder     = args.test,
        force_synthetic = args.synthetic,
    )