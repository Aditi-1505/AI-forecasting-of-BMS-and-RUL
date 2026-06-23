"""
charts.py
---------
Matplotlib / Seaborn chart generators that return base64-encoded PNG strings.
"""
from __future__ import annotations

import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from config import BATTERY

PALETTE = {
    "primary": "#8B5E3C",
    "green":   "#3A6B35",
    "bg":      "#F8F5F0",
    "grid":    "#E8E2D8",
    "eol":     "#D85A30",
    "text":    "#2C2416",
    "muted":   "#8A7F6A",
    "blue":    "#2E6DA4",
    "orange":  "#D0742F",
}

sns.set_theme(style="whitegrid", rc={
    "axes.facecolor":  PALETTE["bg"],
    "figure.facecolor": PALETTE["bg"],
    "axes.edgecolor":  PALETTE["grid"],
    "grid.color":      PALETTE["grid"],
    "text.color":      PALETTE["text"],
    "axes.labelcolor": PALETTE["text"],
    "xtick.color":     PALETTE["muted"],
    "ytick.color":     PALETTE["muted"],
})


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110,
                facecolor=PALETTE["bg"], edgecolor="none")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def make_degradation_curve(soh_pct: float, cycle_index: int) -> str:
    cycles = np.linspace(0, 700, 300)
    soh = np.clip(100 - cycles * 0.03 - np.maximum(0, cycles - 400) * 0.04, 60, 100)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(cycles, soh, color=PALETTE["green"], lw=2, label="Degradation curve")
    ax.axhline(80, color=PALETTE["eol"], lw=1.2, ls="--", label="EOL (80%)")
    ax.fill_between(cycles, soh, 80, where=(soh >= 80), alpha=0.12, color=PALETTE["green"])
    ax.scatter([cycle_index], [soh_pct], color=PALETTE["primary"], s=80, zorder=5, label="Current position")
    ax.set_xlabel("Cycle number", fontsize=10)
    ax.set_ylabel("SOH (%)", fontsize=10)
    ax.set_title("SOH Degradation Curve", fontsize=11, fontweight="bold")
    ax.set_ylim(55, 105)
    ax.set_xlim(0, 700)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    fig.tight_layout(pad=0.8)
    return _fig_to_b64(fig)


def make_feature_importance(fi: dict) -> str:
    sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]
    labels = [k.replace("_", " ") for k, _ in sorted_fi]
    vals   = [v for _, v in sorted_fi]
    colors = [PALETTE["green"] if v > 0.05 else PALETTE["primary"] if v > 0.01 else PALETTE["muted"]
              for v in vals]

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    bars = ax.barh(labels[::-1], vals[::-1], color=colors[::-1], height=0.6)
    for bar, val in zip(bars, vals[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8, color=PALETTE["muted"])
    ax.set_xlabel("Importance", fontsize=10)
    ax.set_title("Feature Importance (XGBoost gain)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.25 if vals else 1)
    fig.tight_layout(pad=0.8)
    return _fig_to_b64(fig)


def make_temperature_derating() -> str:
    temps = np.linspace(-20, 70, 200)

    def tf(t):
        if t < 0:   return 0.60
        if t < 10:  return 0.75 + (t / 10) * 0.10
        if t < 20:  return 0.85 + ((t - 10) / 10) * 0.10
        if t <= 30: return 1.00
        if t <= 40: return 1.00 - ((t - 30) / 10) * 0.05
        if t <= 50: return 0.95 - ((t - 40) / 10) * 0.10
        return 0.80

    factors = np.array([tf(t) * 100 for t in temps])
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(temps, factors, color=PALETTE["primary"], lw=2)
    ax.axvline(25, color=PALETTE["green"], lw=1.2, ls="--", label="Optimal 25°C")
    ax.fill_between(temps, factors, 0, alpha=0.10, color=PALETTE["primary"])
    ax.set_xlabel("Temperature (°C)", fontsize=10)
    ax.set_ylabel("Capacity factor (%)", fontsize=10)
    ax.set_title("Temperature Derating Curve", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=8, framealpha=0.7)
    fig.tight_layout(pad=0.8)
    return _fig_to_b64(fig)


def make_actual_vs_predicted(y_true: list, y_pred: list) -> str:
    from sklearn.metrics import r2_score, mean_absolute_error
    y_true = np.array(y_true[:500])
    y_pred = np.array(y_pred[:500])
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    ax.scatter(y_true, y_pred, alpha=0.35, s=14, color=PALETTE["blue"], edgecolors="none")
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([mn, mx], [mn, mx], color=PALETTE["muted"], lw=1.2, ls="--")
    ax.set_xlabel("Actual (Ah)", fontsize=10)
    ax.set_ylabel("Predicted (Ah)", fontsize=10)
    ax.set_title(f"Actual vs Predicted  (R²={r2:.4f}, MAE={mae:.4f} Ah)",
                 fontsize=10, fontweight="bold")
    fig.tight_layout(pad=0.8)
    return _fig_to_b64(fig)


def make_discharge_profile(cycle_index: int, predicted_cap: float) -> str:
    t   = np.linspace(0, 120, 200)
    cap_norm = predicted_cap / BATTERY.rated_capacity_ah
    v_base   = 3.8 * cap_norm - (3.8 - 2.7) * (t / 120) ** 1.5 * cap_norm
    v_base   = np.clip(v_base, 2.5, 4.2)

    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(t, v_base, color=PALETTE["blue"], lw=2)
    ax.fill_between(t, v_base, 2.5, alpha=0.12, color=PALETTE["blue"])
    ax.axhline(predicted_cap, color=PALETTE["primary"], lw=1.2, ls="--",
               label=f"Pred. capacity: {predicted_cap:.3f} Ah")
    ax.set_xlabel("Time (min)", fontsize=10)
    ax.set_ylabel("Voltage / Capacity proxy", fontsize=10)
    ax.set_title("Discharge Profile", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.7)
    fig.tight_layout(pad=0.8)
    return _fig_to_b64(fig)


def make_rul_timeline(rul_cycles: int, cycle_index: int) -> str:
    total = max(cycle_index + rul_cycles, 1)
    fig, ax = plt.subplots(figsize=(5.5, 2.2))
    ax.barh(["Battery life"], [cycle_index], color=PALETTE["primary"], height=0.4)
    if rul_cycles > 0:
        ax.barh(["Battery life"], [rul_cycles], left=[cycle_index],
                color=PALETTE["green"], height=0.4)
    ax.set_xlim(0, total * 1.05)
    ax.set_xlabel("Cycles", fontsize=10)
    ax.set_title("Remaining Useful Life", fontsize=11, fontweight="bold")
    used_p   = mpatches.Patch(color=PALETTE["primary"], label=f"Used ({cycle_index})")
    remain_p = mpatches.Patch(color=PALETTE["green"],   label=f"Remaining ({rul_cycles})")
    ax.legend(handles=[used_p, remain_p], fontsize=8, loc="upper right", framealpha=0.7)
    fig.tight_layout(pad=0.8)
    return _fig_to_b64(fig)