
import json, os, pickle, sys, tempfile, unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import FEATURES, TARGET, get_feature_stats, load_dataset, make_synthetic_df, load_csv_file
from train_model import build_pipeline, train
from config import BATTERY, DATA_CFG


# ── 1. Synthetic data ─────────────────────────────────────────────────────────

class TestSyntheticData(unittest.TestCase):

    def test_shape(self):
        df = make_synthetic_df(n=500, seed=0)
        self.assertEqual(df.shape[0], 500)
        self.assertIn(TARGET, df.columns)

    def test_all_features_present(self):
        df = make_synthetic_df(n=100, seed=1)
        for f in FEATURES:
            self.assertIn(f, df.columns, f"Missing feature: {f}")

    def test_no_nan_in_features(self):
        df = make_synthetic_df(n=500, seed=2)
        missing = [f for f in FEATURES if df[f].isnull().any()]
        self.assertEqual(missing, [], f"NaN in features: {missing}")

    def test_target_above_minimum(self):
        df = make_synthetic_df(n=500, seed=3)
        self.assertTrue((df[TARGET] > BATTERY.min_valid_capacity_ah).all())

    def test_reproducible(self):
        df1 = make_synthetic_df(n=200, seed=42)
        df2 = make_synthetic_df(n=200, seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = make_synthetic_df(n=200, seed=10)
        df2 = make_synthetic_df(n=200, seed=99)
        self.assertFalse(df1["V_measured"].equals(df2["V_measured"]))

    def test_eol_proximity_range(self):
        df = make_synthetic_df(n=500, seed=5)
        self.assertTrue((df["eol_proximity"] >= 0).all())
        self.assertTrue((df["eol_proximity"] <= 1.5).all())

    def test_eol_proximity_no_leak(self):
        df   = make_synthetic_df(n=300, seed=0)
        corr = df["eol_proximity"].corr(df[TARGET])
        self.assertLess(abs(corr), 0.99,
            f"eol_proximity |r|={abs(corr):.4f} with target — possible data leak!")

    def test_ambient_temp_present_and_varied(self):
        df = make_synthetic_df(n=500, seed=6)
        self.assertIn("ambient_temp", df.columns)
        self.assertGreater(df["ambient_temp"].nunique(), 1,
            "ambient_temp should contain multiple temperature groups")

    def test_cycle_index_starts_at_one(self):
        df = make_synthetic_df(n=100, seed=7)
        self.assertEqual(df["cycle_index"].min(), 1.0)

    def test_impedance_positive(self):
        df = make_synthetic_df(n=200, seed=8)
        self.assertTrue((df["Re"]  > 0).all())
        self.assertTrue((df["Rct"] > 0).all())


# ── 2. Feature stats ──────────────────────────────────────────────────────────

class TestFeatureStats(unittest.TestCase):

    def test_keys_present(self):
        df    = make_synthetic_df(n=200, seed=0)
        stats = get_feature_stats(df)
        for f in FEATURES:
            self.assertIn(f, stats)
            for k in ("min", "max", "mean", "std"):
                self.assertIn(k, stats[f])

    def test_values_finite(self):
        df = make_synthetic_df(n=200, seed=0)
        for feat, s in get_feature_stats(df).items():
            for k, v in s.items():
                self.assertTrue(np.isfinite(v), f"{feat}.{k} is not finite")


# ── 3. CSV / dataset loading ──────────────────────────────────────────────────

class TestCSVLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import glob
        cls.sample_csv = None
        for folder in [DATA_CFG.train_folder, DATA_CFG.val_folder, DATA_CFG.test_folder]:
            hits = glob.glob(os.path.join(folder, "*.csv"))
            if hits:
                cls.sample_csv = hits[0]
                break
        cls.has_real_data = cls.sample_csv is not None

    def test_csv_loads_if_present(self):
        if not self.has_real_data:
            self.skipTest("NASA CSVs not present")
        df = load_csv_file(self.sample_csv)
        self.assertFalse(df.empty)
        self.assertIn(TARGET, df.columns)

    def test_csv_has_all_features(self):
        if not self.has_real_data:
            self.skipTest("NASA CSVs not present")
        df = load_csv_file(self.sample_csv)
        for f in FEATURES:
            self.assertIn(f, df.columns, f"Missing feature: {f}")

    def test_csv_no_nan_in_features(self):
        if not self.has_real_data:
            self.skipTest("NASA CSVs not present")
        df = load_csv_file(self.sample_csv)
        missing = [f for f in FEATURES if df[f].isnull().any()]
        self.assertEqual(missing, [], f"NaN in features: {missing}")

    def test_csv_capacity_above_minimum(self):
        if not self.has_real_data:
            self.skipTest("NASA CSVs not present")
        df = load_csv_file(self.sample_csv)
        self.assertTrue(
            (df[TARGET] > BATTERY.min_valid_capacity_ah).all(),
            "Outlier/zero capacity rows were not filtered"
        )

    def test_all_three_splits_nonempty(self):
        """load_dataset() must return 3 non-empty DataFrames (real or synthetic)."""
        tr, va, te = load_dataset()
        self.assertGreater(len(tr), 0, "Train split empty")
        self.assertGreater(len(va), 0, "Val split empty")
        self.assertGreater(len(te), 0, "Test split empty")

    def test_each_split_has_battery_id(self):
        tr, va, te = load_dataset()
        for name, df in [("Train", tr), ("Val", va), ("Test", te)]:
            self.assertIn("battery_id", df.columns, f"{name} missing battery_id column")
            self.assertGreaterEqual(df["battery_id"].nunique(), 1,
                f"{name}: should have ≥1 battery")

    def test_no_cross_split_battery_overlap(self):
        """No battery should appear in more than one split (real data only)."""
        tr, va, te = load_dataset()
        if tr["battery_id"].iloc[0] == "synthetic":
            self.skipTest("Synthetic data — overlap check not applicable")
        tr_ids = set(tr["battery_id"].unique())
        va_ids = set(va["battery_id"].unique())
        te_ids = set(te["battery_id"].unique())
        self.assertEqual(tr_ids & va_ids, set(), f"Train/Val overlap: {tr_ids & va_ids}")
        self.assertEqual(tr_ids & te_ids, set(), f"Train/Test overlap: {tr_ids & te_ids}")
        self.assertEqual(va_ids & te_ids, set(), f"Val/Test overlap: {va_ids & te_ids}")


# ── 4. Pipeline ───────────────────────────────────────────────────────────────

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.train_df = make_synthetic_df(n=1000, seed=42)
        cls.val_df   = make_synthetic_df(n=300,  seed=7)
        cls.test_df  = make_synthetic_df(n=300,  seed=99)
        X_tr = cls.train_df[FEATURES].values;  y_tr = cls.train_df[TARGET].values
        cls.X_va = cls.val_df[FEATURES].values;  cls.y_va = cls.val_df[TARGET].values
        cls.X_te = cls.test_df[FEATURES].values; cls.y_te = cls.test_df[TARGET].values
        cls.pipe = build_pipeline(n_estimators=80, max_depth=4)
        cls.pipe.fit(X_tr, y_tr)
        cls.preds_va = cls.pipe.predict(cls.X_va)
        cls.preds_te = cls.pipe.predict(cls.X_te)

    def test_has_scaler_and_gbm(self):
        self.assertIn("scaler", self.pipe.named_steps)
        self.assertIn("gbm",    self.pipe.named_steps)

    def test_val_prediction_shape(self):
        self.assertEqual(len(self.preds_va), len(self.X_va))

    def test_test_prediction_shape(self):
        self.assertEqual(len(self.preds_te), len(self.X_te))

    def test_predictions_positive(self):
        self.assertTrue((self.preds_te >= 0).all())

    def test_val_r2_above_threshold(self):
        from sklearn.metrics import r2_score
        r2 = r2_score(self.y_va, self.preds_va)
        self.assertGreater(r2, 0.5, f"Val R²={r2:.3f} too low")

    def test_test_r2_above_threshold(self):
        from sklearn.metrics import r2_score
        r2 = r2_score(self.y_te, self.preds_te)
        self.assertGreater(r2, 0.5, f"Test R²={r2:.3f} too low")

    def test_single_sample(self):
        pred = self.pipe.predict(self.X_te[[0]])
        self.assertEqual(len(pred), 1)
        self.assertFalse(np.isnan(pred[0]))

    def test_feature_importances_sum_to_one(self):
        fi = self.pipe.named_steps["gbm"].feature_importances_
        self.assertAlmostEqual(fi.sum(), 1.0, places=5)
        self.assertEqual(len(fi), len(FEATURES))

    def test_pickle_roundtrip(self):
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fh:
            path = fh.name
        try:
            with open(path, "wb") as fh: pickle.dump(self.pipe, fh)
            with open(path, "rb") as fh: loaded = pickle.load(fh)
            np.testing.assert_array_almost_equal(
                self.preds_te, loaded.predict(self.X_te), decimal=8)
        finally:
            os.unlink(path)

    def test_deterministic(self):
        np.testing.assert_array_equal(
            self.pipe.predict(self.X_te),
            self.pipe.predict(self.X_te))


# ── 5. train() integration ────────────────────────────────────────────────────

class TestTrainFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.metrics = train(force_synthetic=True)

    def test_returns_all_split_metrics(self):
        for k in ("train_r2","train_mae","val_r2","val_mae","test_r2","test_mae",
                  "r2","mae","rmse","fi","y_true","y_pred","features"):
            self.assertIn(k, self.metrics, f"Missing key: {k}")

    def test_test_r2_positive(self):
        self.assertGreater(self.metrics["r2"], 0.5)

    def test_val_r2_positive(self):
        self.assertGreater(self.metrics["val_r2"], 0.5)

    def test_mae_reasonable(self):
        self.assertLess(self.metrics["mae"], 0.5)

    def test_model_file_saved(self):
        self.assertTrue(os.path.exists("model/xgb_battery_model.pkl"))

    def test_meta_json_valid(self):
        self.assertTrue(os.path.exists("model/model_meta.json"))
        with open("model/model_meta.json") as fh:
            meta = json.load(fh)
        for k in ("r2","mae","fi","features","train_r2","val_r2","test_r2",
                  "train_n","val_n","test_n"):
            self.assertIn(k, meta)
        self.assertEqual(meta["features"], FEATURES)

    def test_y_true_y_pred_from_val(self):
        """y_true / y_pred stored in metrics should be the val split."""
        m = self.metrics
        self.assertEqual(len(m["y_true"]), m["val_n"])
        self.assertEqual(len(m["y_pred"]), m["val_n"])


# ── 6. Physics helpers ────────────────────────────────────────────────────────

def _tf(t):
    if t < 0:   return 0.60
    if t < 10:  return 0.75 + (t / 10) * 0.10
    if t < 20:  return 0.85 + ((t - 10) / 10) * 0.10
    if t <= 30: return 1.00
    if t <= 40: return 1.00 - ((t - 30) / 10) * 0.05
    if t <= 50: return 0.95 - ((t - 40) / 10) * 0.10
    return 0.80

def _cf(n):
    if n <= 1: return 1.0
    lin  = min(n, 500) * 0.0002
    knee = max(0, n - 500) * 0.0003
    return max(0.5, 1.0 - lin - knee)


class TestPhysicsHelpers(unittest.TestCase):

    def test_temp_optimal(self):     self.assertEqual(_tf(25), 1.0)
    def test_temp_cold(self):        self.assertLess(_tf(-5), 0.75)
    def test_temp_4C(self):          self.assertGreater(_tf(4), 0.75); self.assertLess(_tf(4), 0.90)
    def test_temp_hot(self):         self.assertLessEqual(_tf(55), 0.80)

    def test_temp_monotone(self):
        vals = [_tf(t) for t in range(-10, 31)]
        for i in range(len(vals) - 1):
            self.assertLessEqual(vals[i], vals[i + 1] + 1e-9)

    def test_cycle_new(self):        self.assertEqual(_cf(1), 1.0)
    def test_cycle_decreases(self):
        for n in [1, 100, 300, 500, 800]:
            self.assertGreater(_cf(n), _cf(n + 100))
    def test_cycle_minimum(self):    self.assertGreaterEqual(_cf(5000), 0.5)
    def test_cycle_not_above_one(self):
        for n in [0, 1, 10, 100, 500]:
            self.assertLessEqual(_cf(n), 1.0)


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite  = loader.discover(start_dir=".", pattern="test_model.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)