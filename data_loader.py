from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from config import BATTERY, DATA_CFG
from logger import get_logger

log = get_logger(__name__)

# ── Features (15 physics-grounded, leak-free) ─────────────────────────────────
FEATURES: list[str] = [
    "cycle_index",        # sequential discharge counter (1-based, per battery)
    "V_measured",         # mean terminal voltage during discharge (V)
    "T_measured",         # mean cell temperature during discharge (°C)
    "ambient_temp",       # test ambient temperature (°C) — critical across batteries
    "Re",                 # electrolyte resistance (Ω) — grows with SEI layer
    "Rct",                # charge-transfer resistance (Ω) — grows near EOL
    "Re_growth",          # ΔRe from cycle-1 baseline (Ω)
    "Rct_growth",         # ΔRct from cycle-1 baseline (Ω)
    "V_drop",             # voltage sag vs cycle-1 baseline (V)
    "cap_fade_rate",      # ΔCapacity from previous cycle (Ah) — causal only
    "eol_proximity",      # cycle_index / 168 — proximity to nominal EOL
    "Re_x_Rct",           # Re × Rct interaction
    "Rct_Re_ratio",       # Rct / Re ratio
    "V_Re_interaction",   # V × Re coupling term
    "cycle_sq",           # cycle_index² — non-linear ageing acceleration
]

TARGET = "Discharge_Capacity_Ah"

# Nominal cycle life used for eol_proximity normalisation.
# NASA 18650 cells are tested to ~168 discharge cycles.
_EOL_CYCLE_NORM = 168.0


# ── Single-file loader ────────────────────────────────────────────────────────

def load_csv_file(path: str) -> pd.DataFrame:
    """
    Parse one NASA *_full.csv into a per-discharge-cycle feature DataFrame.

    Steps:
    1. Interpolate Re/Rct from 'impedance' rows onto all row indices.
    2. For each discharge cycle (grouped by 'cycle' column), extract:
         - The single Capacity value (first non-null entry in that cycle).
         - Mean Voltage_measured, Temperature_measured, ambient_temperature.
         - Mean Re_interp, Rct_interp (already continuous after step 1).
    3. Drop cycles with Capacity <= min_valid_capacity_ah (outliers/zeros).
    4. Engineer all 15 features from the per-cycle summary.
    """
    try:
        raw = pd.read_csv(path)
    except Exception:
        log.exception("Cannot read %s", path)
        return pd.DataFrame()

    battery_id = os.path.splitext(os.path.basename(path))[0]

    # ── 1. Interpolate impedance measurements onto every row ──────────────
    # Re/Rct may be stored as complex-number strings e.g. "(0.048+0j)".
    # We extract the real part to get the resistance magnitude.
    def _to_real(series: pd.Series) -> np.ndarray:
        def _parse(v):
            try:
                if isinstance(v, (int, float)) and np.isfinite(v):
                    return float(v)
                return complex(str(v).strip().replace(" ", "")).real
            except Exception:
                return np.nan
        return series.apply(_parse).values.astype(float)

    imp = raw[raw["type"] == "impedance"].copy()
    re_vals  = _to_real(imp["Re"])
    rct_vals = _to_real(imp["Rct"])
    valid    = np.isfinite(re_vals) & np.isfinite(rct_vals)
    imp      = imp[valid].copy()
    re_vals  = re_vals[valid]
    rct_vals = rct_vals[valid]

    if len(imp) > 0:
        idx_all = raw.index.values.astype(float)
        idx_imp = imp.index.values.astype(float)
        raw["Re_interp"]  = np.interp(idx_all, idx_imp, re_vals)
        raw["Rct_interp"] = np.interp(idx_all, idx_imp, rct_vals)
    else:
        raw["Re_interp"]  = np.nan
        raw["Rct_interp"] = np.nan

    # ── 2. Aggregate discharge rows to one row per cycle ──────────────────
    dis = raw[raw["type"] == "discharge"].copy()

    def _first_nonnull(s: pd.Series):
        valid = s.dropna()
        return valid.iloc[0] if len(valid) else np.nan

    agg = (
        dis.groupby("cycle", sort=True)
        .agg(
            Capacity     = ("Capacity",              _first_nonnull),
            V_measured   = ("Voltage_measured",      "mean"),
            T_measured   = ("Temperature_measured",  "mean"),
            ambient_temp = ("ambient_temperature",   "first"),
            Re           = ("Re_interp",             "mean"),
            Rct          = ("Rct_interp",            "mean"),
        )
        .reset_index()
        .rename(columns={"cycle": "cycle_number"})
    )

    # ── 3. Filter outlier / partial-discharge cycles ──────────────────────
    valid_mask = agg["Capacity"].notna() & (agg["Capacity"] > BATTERY.min_valid_capacity_ah)
    agg = agg[valid_mask].reset_index(drop=True)

    if agg.empty:
        log.warning("%s: no valid discharge cycles after filtering", battery_id)
        return pd.DataFrame()

    # ── 4. Build sequential cycle index (1-based) ─────────────────────────
    n  = len(agg)
    ci = np.arange(1, n + 1, dtype=float)

    # Raw arrays
    cap = agg["Capacity"].values.astype(float)
    V   = agg["V_measured"].values.astype(float)
    T   = agg["T_measured"].values.astype(float)
    amb = agg["ambient_temp"].values.astype(float)
    Re  = agg["Re"].values.astype(float)
    Rct = agg["Rct"].values.astype(float)

    # Fill any remaining NaNs (e.g. batteries with no EIS rows)
    def _fill(arr: np.ndarray, default: float) -> np.ndarray:
        finite = arr[np.isfinite(arr)]
        med    = float(np.median(finite)) if len(finite) > 0 else default
        out    = arr.copy()
        out[~np.isfinite(out)] = med
        return out

    Re  = _fill(Re,  0.044)
    Rct = _fill(Rct, 0.069)
    V   = _fill(V,   3.80)
    T   = _fill(T,   24.0)
    amb = _fill(amb, 24.0)

    # ── 5. Feature engineering ────────────────────────────────────────────
    V0, Re0, Rct0 = V[0], Re[0], Rct[0]

    V_drop           = V0 - V
    Re_growth        = Re  - Re0
    Rct_growth       = Rct - Rct0
    cap_fade_rate    = pd.Series(cap).diff().fillna(0.0).values   # causal only
    eol_proximity    = np.clip(ci / _EOL_CYCLE_NORM, 0.0, 1.5)
    Re_x_Rct         = Re * Rct
    Rct_Re_ratio     = Rct / (Re + 1e-9)
    V_Re_interaction = V * Re
    cycle_sq         = ci ** 2

    # ── 6. Assemble ───────────────────────────────────────────────────────
    df = pd.DataFrame({
        "cycle_index":       ci,
        "cycle_number":      agg["cycle_number"].values,
        "V_measured":        np.round(V,                5),
        "T_measured":        np.round(T,                4),
        "ambient_temp":      np.round(amb,              2),
        "Re":                np.round(Re,               6),
        "Rct":               np.round(Rct,              6),
        "Re_growth":         np.round(Re_growth,        6),
        "Rct_growth":        np.round(Rct_growth,       6),
        "V_drop":            np.round(V_drop,           5),
        "cap_fade_rate":     np.round(cap_fade_rate,    6),
        "eol_proximity":     np.round(eol_proximity,    5),
        "Re_x_Rct":          np.round(Re_x_Rct,         7),
        "Rct_Re_ratio":      np.round(Rct_Re_ratio,     5),
        "V_Re_interaction":  np.round(V_Re_interaction, 6),
        "cycle_sq":          np.round(cycle_sq,         1),
        TARGET:              np.round(cap,              6),
        "battery_id":        battery_id,
    })

    # Residual NaN safety net
    for col in FEATURES:
        if df[col].isnull().any():
            med = df[col].median()
            df[col] = df[col].fillna(med if np.isfinite(med) else 0.0)

    log.info(
        "  %-22s → %3d cycles | cap %.4f–%.4f Ah | amb %.0f°C",
        battery_id, n,
        float(df[TARGET].min()), float(df[TARGET].max()),
        float(amb.mean()),
    )
    return df


# ── Folder loader ─────────────────────────────────────────────────────────────

def load_csv_folder(folder: str) -> pd.DataFrame:
    """
    Load every *_full.csv (and *.csv) in `folder` and concatenate.
    Returns an empty DataFrame if the folder is missing or has no valid files.
    """
    files = sorted(
        glob.glob(os.path.join(folder, "*_full.csv")) or
        glob.glob(os.path.join(folder, "*.csv"))
    )
    if not files:
        log.warning("No CSV files found in '%s'", folder)
        return pd.DataFrame()

    log.info("Loading %d files from '%s' …", len(files), folder)
    frames = [load_csv_file(p) for p in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    log.info(
        "  → %d total cycles | %d batteries: %s",
        len(combined),
        combined["battery_id"].nunique(),
        sorted(combined["battery_id"].unique().tolist()),
    )
    return combined


# ── 3-way dataset loader ──────────────────────────────────────────────────────

def load_dataset(
    train_folder: str = DATA_CFG.train_folder,
    val_folder:   str = DATA_CFG.val_folder,
    test_folder:  str = DATA_CFG.test_folder,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train_df, val_df, test_df).

    Reads every CSV in each folder automatically — no battery IDs hardcoded.
    Falls back to synthetic data for any split whose folder is empty/missing.
    """
    log.info("=" * 60)
    log.info("Loading NASA Battery dataset (3-way split)")
    log.info("=" * 60)

    train_df = load_csv_folder(train_folder)
    val_df   = load_csv_folder(val_folder)
    test_df  = load_csv_folder(test_folder)

    missing = []
    if train_df.empty: missing.append("Train")
    if val_df.empty:   missing.append("Val")
    if test_df.empty:  missing.append("Test")

    if missing:
        log.warning("Missing/empty splits: %s — using synthetic fallback", missing)
        if train_df.empty:
            train_df = make_synthetic_df(n=DATA_CFG.synthetic_train_n, seed=DATA_CFG.synthetic_seed_train)
        if val_df.empty:
            val_df   = make_synthetic_df(n=DATA_CFG.synthetic_val_n,   seed=DATA_CFG.synthetic_seed_val)
        if test_df.empty:
            test_df  = make_synthetic_df(n=DATA_CFG.synthetic_test_n,  seed=DATA_CFG.synthetic_seed_test)

    log.info("-" * 60)
    log.info("Train: %4d cycles | %d batteries", len(train_df), train_df["battery_id"].nunique())
    log.info("Val  : %4d cycles | %d batteries", len(val_df),   val_df["battery_id"].nunique())
    log.info("Test : %4d cycles | %d batteries", len(test_df),  test_df["battery_id"].nunique())
    log.info("=" * 60)
    return train_df, val_df, test_df


# ── Feature statistics ────────────────────────────────────────────────────────

def get_feature_stats(df: pd.DataFrame) -> dict:
    stats: dict = {}
    for col in FEATURES:
        if col in df.columns:
            stats[col] = {
                "min":  round(float(df[col].min()),  6),
                "max":  round(float(df[col].max()),  6),
                "mean": round(float(df[col].mean()), 6),
                "std":  round(float(df[col].std()),  6),
            }
    return stats


# ── Synthetic fallback ────────────────────────────────────────────────────────

def make_synthetic_df(
    n:    int = DATA_CFG.synthetic_train_n,
    seed: int = DATA_CFG.synthetic_seed_train,
) -> pd.DataFrame:
    """
    Physics-grounded synthetic dataset mirroring the full NASA collection.
    Includes both 4°C and 24°C ambient temperature groups.
    """
    rng = np.random.default_rng(seed)
    ci  = np.arange(1, n + 1, dtype=float)
    deg = np.clip(ci / _EOL_CYCLE_NORM, 0.0, 1.0)

    # Two ambient temp groups — 4°C reduces capacity by ~25%
    amb       = rng.choice([4.0, 24.0], size=n).astype(float)
    amb_scale = np.where(amb == 4.0, 0.75, 1.0)

    cap = (1.856 - 0.57 * deg) * amb_scale + rng.normal(0, 0.004, n)
    cap = np.clip(cap, BATTERY.min_valid_capacity_ah + 0.001, 1.95)

    V   = 4.19 - 0.03 * deg + rng.normal(0, 0.002, n)
    V0  = V[0]
    T   = amb + rng.normal(0, 0.8, n)
    Re  = 0.044 + 0.032 * deg + rng.uniform(0, 0.003, n)
    Rct = 0.069 + 0.035 * deg + rng.uniform(0, 0.006, n)
    Re0, Rct0 = Re[0], Rct[0]

    V_drop           = V0 - V
    cap_fade_rate    = pd.Series(cap).diff().fillna(0.0).values
    eol_proximity    = np.clip(ci / _EOL_CYCLE_NORM, 0.0, 1.5)
    Re_x_Rct         = Re * Rct
    Rct_Re_ratio     = Rct / (Re + 1e-9)
    V_Re_interaction = V * Re
    cycle_sq         = ci ** 2

    return pd.DataFrame({
        "cycle_index":       ci,
        "cycle_number":      ci,
        "V_measured":        np.round(V,                5),
        "T_measured":        np.round(T,                4),
        "ambient_temp":      amb,
        "Re":                np.round(Re,               6),
        "Rct":               np.round(Rct,              6),
        "Re_growth":         np.round(Re  - Re0,        6),
        "Rct_growth":        np.round(Rct - Rct0,       6),
        "V_drop":            np.round(V_drop,           5),
        "cap_fade_rate":     np.round(cap_fade_rate,    6),
        "eol_proximity":     np.round(eol_proximity,    5),
        "Re_x_Rct":          np.round(Re_x_Rct,         7),
        "Rct_Re_ratio":      np.round(Rct_Re_ratio,     5),
        "V_Re_interaction":  np.round(V_Re_interaction, 6),
        "cycle_sq":          np.round(cycle_sq,         1),
        TARGET:              np.round(cap,              6),
        "battery_id":        "synthetic",
    })


# ── RUL synthetic data ────────────────────────────────────────────────────────

def make_synthetic_rul_df(n_cells: int = 300, seed: int = 0) -> pd.DataFrame:
    """Multi-cell aging simulation with full-life RUL labels."""
    rng     = np.random.default_rng(seed)
    rows    = []
    eol_cap = BATTERY.rated_capacity_ah * BATTERY.eol_threshold_pct / 100.0

    def _tf(t):
        if t < 0:   return 0.60
        if t < 10:  return 0.75 + (t / 10) * 0.10
        if t < 20:  return 0.85 + ((t - 10) / 10) * 0.10
        if t <= 30: return 1.00
        if t <= 40: return 1.00 - ((t - 30) / 10) * 0.05
        if t <= 50: return 0.95 - ((t - 40) / 10) * 0.10
        return 0.80

    for _ in range(n_cells):
        avg_temp   = float(rng.choice([4.0, 24.0, 45.0]))
        avg_dod    = float(rng.uniform(0.5, 1.0))
        cycle_rate = float(rng.uniform(0.5, 2.0))
        deep_rate  = float(rng.uniform(0.0, 0.15))
        stress     = _tf(avg_temp) * avg_dod * cycle_rate * (1 + deep_rate)
        fade_lin   = 0.0003 * stress
        fade_knee  = 0.0008 * stress
        knee_onset = int(rng.uniform(300, 500))
        cap        = BATTERY.rated_capacity_ah
        cycle_rows: list[dict] = []
        eol_cycle  = None

        for c in range(1, 1501):
            cap -= fade_lin
            if c > knee_onset:
                cap -= fade_knee
            cap += float(rng.normal(0, 0.002))
            cap  = max(cap, 0.3)
            soh  = cap / BATTERY.rated_capacity_ah * 100.0
            cycle_rows.append({
                "cycle_index": float(c),
                "soh_pct":     round(soh, 3),
                "capacity_ah": round(cap, 5),
                "avg_temp_C":  avg_temp,
                "avg_dod":     round(avg_dod,    3),
                "cycle_rate":  round(cycle_rate, 3),
                "deep_rate":   round(deep_rate,  4),
                "RUL":         0,
            })
            if cap <= eol_cap and eol_cycle is None:
                eol_cycle = c
                break

        if eol_cycle is not None:
            for row in cycle_rows:
                row["RUL"] = max(0, eol_cycle - int(row["cycle_index"]))
        rows.extend(cycle_rows)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    train_df, val_df, test_df = load_dataset()
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        log.info("%s shape: %s | batteries: %s",
                 name, df.shape, sorted(df["battery_id"].unique().tolist()))
    log.info("Features: %s", FEATURES)
    for feat, s in get_feature_stats(train_df).items():
        log.info("  %-28s  min=%8.4f  max=%8.4f  mean=%8.4f", feat, s["min"], s["max"], s["mean"])