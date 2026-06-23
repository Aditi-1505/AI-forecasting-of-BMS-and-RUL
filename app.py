"""
app.py  —  Flask REST API for NASA Battery BMS SOH & RUL Forecasting
"""
from __future__ import annotations
import json, os
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from config import APP_CFG, BATTERY, MODEL_CFG
from data_loader import FEATURES, TARGET, get_feature_stats, load_dataset
from logger import get_logger
from train_model import build_pipeline, load_model, train

log  = get_logger(__name__)
app  = Flask(__name__)
CORS(app)

_pipe     = None
_rul_pipe = None
_meta: dict = {}

def _get_model():
    global _pipe, _meta
    if _pipe is not None:
        return _pipe, _meta
    if os.path.exists(MODEL_CFG.model_path):
        _pipe, _meta = load_model()
    else:
        log.info("No saved model — training now …")
        train()
        _pipe, _meta = load_model()
    return _pipe, _meta

def _get_rul_model():
    global _rul_pipe
    if _rul_pipe is not None:
        return _rul_pipe
    if os.path.exists(MODEL_CFG.rul_path):
        from train_rul_model import load_rul_model
        _rul_pipe = load_rul_model()
    else:
        from train_rul_model import train_rul, load_rul_model
        train_rul()
        _rul_pipe = load_rul_model()
    return _rul_pipe

# ── Physics helpers ─────────────────────────────────────────────────────────

def _temp_factor(temp_c: float) -> float:
    if temp_c < 0:    return 0.60
    if temp_c < 10:   return 0.75 + (temp_c / 10.0) * 0.10
    if temp_c < 20:   return 0.85 + ((temp_c - 10) / 10.0) * 0.10
    if temp_c <= 30:  return 1.00
    if temp_c <= 40:  return 1.00 - ((temp_c - 30) / 10.0) * 0.05
    if temp_c <= 50:  return 0.95 - ((temp_c - 40) / 10.0) * 0.10
    return 0.80

def _cycle_factor(n: int) -> float:
    if n <= 1: return 1.0
    lin  = min(n, 500) * 0.0002
    knee = max(0, n - 500) * 0.0003
    return max(0.50, 1.0 - lin - knee)

def _health_grade(soh: float) -> str:
    if soh >= 90: return "Excellent"
    if soh >= 80: return "Good"
    if soh >= 70: return "Fair"
    if soh >= 60: return "Poor"
    return "Critical"

def _combine_dod_eol(deep_discharge: bool, at_eol: bool) -> str:
    if deep_discharge and at_eol:
        return "Deep discharge AND past EOL — recharge will not restore lost capacity"
    if deep_discharge:
        return "Deep discharge only — cell is healthy, expected to recover on recharge"
    if at_eol:
        return "Past EOL (gradual fade) — replace battery; recharge cannot restore capacity"
    return "Normal operation — battery is healthy"

# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    _, meta = _get_model()
    return render_template("index.html", metrics=meta)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "model_ready": _pipe is not None})

@app.route("/api/model-info")
def model_info():
    try:
        _, meta = _get_model()
        return jsonify({
            "status":          "ok",
            "model":           meta.get("model"),
            "backend":         meta.get("backend", "sklearn"),
            "dataset":         meta.get("dataset"),
            "train_batteries": meta.get("train_batteries", []),
            "test_batteries":  meta.get("test_batteries", []),
            "train_n":         meta.get("train_n"),
            "test_n":          meta.get("test_n"),
            "r2":              meta.get("r2"),
            "mae":             meta.get("mae"),
            "rmse":            meta.get("rmse"),
            "n_estimators":    meta.get("n_estimators"),
            "max_depth":       meta.get("max_depth"),
            "learning_rate":   meta.get("learning_rate"),
            "training_time_s": meta.get("training_time_s"),
            "features":        meta.get("features", FEATURES),
        })
    except Exception as exc:
        log.exception("model-info error")
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route("/api/feature-importance")
def feature_importance():
    try:
        _, meta = _get_model()
        fi = meta.get("fi", {})
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        return jsonify({
            "status": "ok",
            "feature_importance": [{"feature": k, "importance": v} for k, v in sorted_fi],
        })
    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST JSON with all 14 FEATURES (+ optional helpers):
      cycle_index, V_measured, T_measured, Re, Rct,
      Re_growth, Rct_growth, V_drop, cap_fade_rate,
      eol_proximity, Re_x_Rct, Rct_Re_ratio,
      V_Re_interaction, cycle_sq
    Optional: T_mean / ambient_temp_C for physics correction.
    """
    try:
        pipe, _ = _get_model()
        body    = request.get_json(force=True)

        # Compute interaction features server-side if not supplied
        Re  = float(body.get("Re",  0.044))
        Rct = float(body.get("Rct", 0.069))
        V   = float(body.get("V_measured", 4.19))
        ci  = float(body.get("cycle_index", 1))
        # Default ambient_temp from ambient_temp_C or T_measured if not supplied
        body.setdefault("ambient_temp",
                        float(body.get("ambient_temp_C",
                                       body.get("T_measured", 24.0))))
        body.setdefault("Re_x_Rct",         Re * Rct)
        body.setdefault("Rct_Re_ratio",      Rct / (Re + 1e-9))
        body.setdefault("V_Re_interaction",  V * Re)
        body.setdefault("cycle_sq",          ci ** 2)

        try:
            x = np.array([[float(body[f]) for f in FEATURES]])
        except KeyError as ke:
            return jsonify({"status": "error", "message": f"Missing field: {ke}"}), 400

        raw_cap = float(pipe.predict(x)[0])
        raw_cap = max(0.0, raw_cap)

        # Physics corrections (temperature + cycle-age)
        t_mean   = float(body.get("T_mean", body.get("T_measured", 24.0)))
        n_cycles = int(ci)
        tf       = _temp_factor(t_mean)
        cf       = _cycle_factor(n_cycles)
        adj_cap  = round(raw_cap * tf * cf, 4)

        soh      = round(min(100.0, adj_cap / BATTERY.rated_capacity_ah * 100.0), 1)
        grade    = _health_grade(soh)
        at_eol   = soh < BATTERY.eol_threshold_pct
        near_eol = soh < BATTERY.eol_strict_pct
        deep_dis = float(body.get("DoD_proxy", 0)) > 0.99

        # RUL from dedicated model
        rul_cycles = 0
        try:
            rul_pipe = _get_rul_model()
            avg_dod    = float(body.get("DoD_proxy", 0.99))
            avg_temp   = float(body.get("ambient_temp_C", t_mean))
            cycle_rate = float(body.get("cycle_rate", 1.0))
            deep_rate  = float(body.get("deep_rate", 0.0))
            rul_x = np.array([[ci, soh, adj_cap, avg_temp, avg_dod, cycle_rate, deep_rate]])
            rul_cycles = max(0, int(rul_pipe.predict(rul_x)[0]))
        except Exception:
            eol_cap = BATTERY.rated_capacity_ah * BATTERY.eol_threshold_pct / 100.0
            rul_cycles = max(0, int((adj_cap - eol_cap) / 0.001)) if adj_cap > eol_cap else 0

        days_rem   = round(rul_cycles / BATTERY.default_cycles_per_day, 1)
        dod_vs_eol = _combine_dod_eol(deep_dis, at_eol)

        log.info(
            "predict: cycle=%d V=%.3f Re=%.4f Rct=%.4f T=%.1f → cap=%.4f Ah SOH=%.1f%% RUL=%d",
            n_cycles, V, Re, Rct, t_mean, adj_cap, soh, rul_cycles,
        )

        return jsonify({
            "status":                "ok",
            "predicted_capacity_ah": adj_cap,
            "raw_capacity_ah":       round(raw_cap, 4),
            "soh_pct":               soh,
            "rul_cycles":            rul_cycles,
            "days_remaining":        days_rem,
            "health_grade":          grade,
            "temp_factor":           round(tf, 4),
            "cycle_factor":          round(cf, 4),
            "at_eol":                at_eol,
            "near_eol":              near_eol,
            "dod_vs_eol":            dod_vs_eol,
        })

    except Exception as exc:
        log.exception("predict error")
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route("/api/battery-profile/<battery_id>")
def battery_profile(battery_id: str):
    try:
        from data_loader import load_csv_file
        import glob
        df = None
        for d in ["NASA/Train", "NASA/Test", "NASA", "."]:
            matches = glob.glob(os.path.join(d, f"{battery_id}*.csv"))
            if matches:
                df = load_csv_file(matches[0])
                break
        if df is None or df.empty:
            return jsonify({"status": "error", "message": f"Battery {battery_id} not found"}), 404
        df = df.sort_values("cycle_index")
        return jsonify({
            "status":    "ok",
            "battery_id": battery_id,
            "cycles":    df["cycle_index"].tolist(),
            "capacity":  [round(v, 5) for v in df[TARGET].tolist()],
            "soh":       [round(min(100, v / BATTERY.rated_capacity_ah * 100), 2) for v in df[TARGET].tolist()],
            "eol_line":  BATTERY.eol_threshold_pct,
            "rated_cap": BATTERY.rated_capacity_ah,
            "n_cycles":  len(df),
        })
    except Exception as exc:
        log.exception("battery-profile error")
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route("/api/dataset-stats")
def dataset_stats():
    try:
        from config import DATA_CFG
        train_df, val_df, test_df = load_dataset(
            DATA_CFG.train_folder, DATA_CFG.val_folder, DATA_CFG.test_folder
        )
        def _stats(df):
            result = []
            for bid, grp in df.groupby("battery_id"):
                grp = grp.sort_values("cycle_index")
                result.append({
                    "battery_id": bid,
                    "n_cycles":   len(grp),
                    "cap_min":    round(float(grp[TARGET].min()), 4),
                    "cap_max":    round(float(grp[TARGET].max()), 4),
                    "cap_mean":   round(float(grp[TARGET].mean()), 4),
                    "soh_final":  round(float(grp[TARGET].iloc[-1]) / BATTERY.rated_capacity_ah * 100, 1),
                    "cycles":     grp["cycle_index"].tolist(),
                    "capacity":   [round(v, 5) for v in grp[TARGET].tolist()],
                })
            return result
        return jsonify({
            "status":    "ok",
            "train":     _stats(train_df),
            "val":       _stats(val_df),
            "test":      _stats(test_df),
            "eol_cap":   round(BATTERY.rated_capacity_ah * BATTERY.eol_threshold_pct / 100, 4),
            "rated_cap": BATTERY.rated_capacity_ah,
        })
    except Exception as exc:
        log.exception("dataset-stats error")
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route("/api/retrain", methods=["POST"])
def retrain():
    global _pipe, _meta, _rul_pipe
    try:
        log.info("Retraining triggered via API …")
        metrics = train()
        _pipe, _meta = load_model()
        _rul_pipe = None
        return jsonify({
            "status":  "ok",
            "message": "Model retrained on NASA data",
            "r2":      metrics["r2"],
            "mae":     metrics["mae"],
            "rmse":    metrics["rmse"],
            "train_n": metrics["train_n"],
            "test_n":  metrics["test_n"],
        })
    except Exception as exc:
        log.exception("retrain error")
        return jsonify({"status": "error", "message": str(exc)}), 500

@app.route("/api/charts", methods=["POST"])
def charts():
    try:
        from charts import (
            make_degradation_curve, make_feature_importance,
            make_temperature_derating, make_actual_vs_predicted,
            make_discharge_profile,
        )
        body     = request.get_json(force=True)
        cycle    = int(body.get("cycle_index", 1))
        soh      = float(body.get("soh_pct", 100.0))
        pred_cap = float(body.get("predicted_capacity_ah", 2.0))
        _, meta  = _get_model()
        fi       = meta.get("fi", {})
        y_true = meta.get("y_true", [])
        y_pred = meta.get("y_pred", [])

        # Regenerate y_true/y_pred from val set if not cached in meta
        if not y_true or not y_pred:
            try:
                from data_loader import load_dataset, FEATURES, TARGET
                from config import DATA_CFG
                import json as _json
                _, val_df, _ = load_dataset(
                    DATA_CFG.train_folder, DATA_CFG.val_folder, DATA_CFG.test_folder
                )
                y_true = val_df[TARGET].tolist()
                _p, _ = _get_model()
                y_pred = _p.predict(val_df[FEATURES].values).tolist()
                meta["y_true"] = y_true
                meta["y_pred"] = y_pred
                with open(MODEL_CFG.meta_path, "w") as _f:
                    _json.dump(meta, _f, indent=2)
                log.info("Cached y_true/y_pred into meta (%d val samples)", len(y_true))
            except Exception as _e:
                log.warning("Could not generate y_true/y_pred: %s", _e)

        result = {
            "status":             "ok",
            "degradation":        make_degradation_curve(soh, cycle),
            "discharge":          make_discharge_profile(cycle, pred_cap),
            "feature_importance": make_feature_importance(fi),
            "temperature":        make_temperature_derating(),
        }
        if y_true and y_pred:
            result["actual_vs_pred"] = make_actual_vs_predicted(y_true, y_pred)
        return jsonify(result)
    except Exception as exc:
        log.exception("charts error")
        return jsonify({"status": "error", "message": str(exc)}), 500

# ── Boot ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting NASA Battery BMS — SOH & RUL Forecasting …")
    _get_model()
    _get_rul_model()
    log.info("API ready → http://%s:%d", APP_CFG.host, APP_CFG.port)
    app.run(host=APP_CFG.host, port=APP_CFG.port, debug=APP_CFG.debug)