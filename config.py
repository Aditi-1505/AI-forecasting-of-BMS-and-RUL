"""
config.py — All project-wide constants.
Supports full NASA dataset: Train / Val / Test folders,
each containing all batteries assigned to that split.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class BatteryConstants:
    nominal_voltage_v:      float = 3.7
    rated_capacity_ah:      float = 2.0
    eol_threshold_pct:      float = 80.0
    eol_strict_pct:         float = 85.0
    eol_lenient_pct:        float = 70.0
    deep_discharge_v:       float = 2.50
    deep_discharge_soc:     float = 0.15
    default_cycles_per_day: float = 1.0
    # Minimum valid discharge capacity — removes zero/near-zero outliers only
    min_valid_capacity_ah:  float = 0.01


@dataclass(frozen=True)
class ModelConfig:
    n_estimators:     int   = 400
    max_depth:        int   = 5
    learning_rate:    float = 0.05
    subsample:        float = 0.85
    min_samples_leaf: int   = 5
    random_state:     int   = 42

    model_dir:      str = "model"
    model_filename: str = "xgb_battery_model.pkl"
    rul_filename:   str = "xgb_rul_model.pkl"
    meta_filename:  str = "model_meta.json"

    @property
    def model_path(self) -> str:
        return os.path.join(self.model_dir, self.model_filename)

    @property
    def rul_path(self) -> str:
        return os.path.join(self.model_dir, self.rul_filename)

    @property
    def meta_path(self) -> str:
        return os.path.join(self.model_dir, self.meta_filename)


@dataclass(frozen=True)
class DataConfig:
    # Folders — place all assigned battery CSVs in each folder
    train_folder: str  = "NASA/Train"   # e.g. B0005, B0006, B0025, B0026, ...
    val_folder:   str  = "NASA/Val"     # e.g. B0018, B0027, B0031, B0039, B0050
    test_folder:  str  = "NASA/Test"    # e.g. B0007, B0028, B0032, B0036, ...

    # Synthetic fallback (used only when a folder is missing or empty)
    synthetic_train_n:    int = 5000
    synthetic_val_n:      int = 1000
    synthetic_test_n:     int = 1500
    synthetic_seed_train: int = 42
    synthetic_seed_val:   int = 7
    synthetic_seed_test:  int = 99


class AppConfig:
    def __init__(self):
        self.host      = os.getenv("FLASK_HOST", "0.0.0.0")
        self.port      = int(os.getenv("FLASK_PORT", 5002))
        self.debug     = os.getenv("FLASK_DEBUG", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "INFO")


BATTERY   = BatteryConstants()
MODEL_CFG = ModelConfig()
DATA_CFG  = DataConfig()
APP_CFG   = AppConfig()