import json
import os

import joblib
import pandas as pd
import pytest

DATA_DIR = os.getenv("DATA_DIR", "data/processed")
REQUIRED_COLUMNS = {
    "Time", "Amount", "Class",
    "V1", "V2", "V3", "V4", "V5", "V6", "V7",
    "V8", "V9", "V10", "V11", "V12", "V13", "V14",
}


class TestDataValidation:
    def test_train_file_exists(self):
        assert os.path.exists(os.path.join(DATA_DIR, "train.csv"))

    def test_val_file_exists(self):
        assert os.path.exists(os.path.join(DATA_DIR, "val.csv"))

    def test_test_file_exists(self):
        assert os.path.exists(os.path.join(DATA_DIR, "test.csv"))

    def test_train_schema(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        missing = REQUIRED_COLUMNS - set(df.columns)
        assert not missing, f"Missing columns: {sorted(missing)}"

    def test_target_column_binary(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        assert set(df["Class"].unique()) <= {0, 1}

    def test_no_missing_target(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        assert df["Class"].notna().all()

    def test_train_min_rows(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        assert df.shape[0] >= 100

    def test_fraud_class_present(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        assert (df["Class"] == 1).sum() >= 1

    def test_amount_non_negative(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        assert (df["Amount"] >= 0).all()

    def test_no_duplicate_rows(self):
        df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
        n_dupes = df.duplicated().sum()
        max_allowed = int(len(df) * 0.01) 
        assert n_dupes <= max_allowed, (
            f"Too many duplicate rows: {n_dupes} > {max_allowed} (1% of {len(df)})"
            )


class TestConfigValidation:
    def test_config_yaml_exists(self):
        assert os.path.exists("config/config.yaml")

    def test_dvc_yaml_exists(self):
        assert os.path.exists("dvc.yaml")

    def test_requirements_exists(self):
        assert os.path.exists("requirements.txt")


class TestArtifacts:
    def test_model_pkl_exists(self):
        assert os.path.exists("models/model.pkl"), "models/model.pkl not found"

    def test_model_pkl_loadable(self):
        if not os.path.exists("models/model.pkl"):
            pytest.skip("model.pkl not found")
        model = joblib.load("models/model.pkl")
        assert model is not None

    def test_model_has_predict(self):
        if not os.path.exists("models/model.pkl"):
            pytest.skip("model.pkl not found")
        model = joblib.load("models/model.pkl")
        assert hasattr(model, "predict")

    def test_metrics_json_exists(self):
        assert os.path.exists("metrics.json"), "metrics.json not found"

    def test_metrics_json_valid(self):
        if not os.path.exists("metrics.json"):
            pytest.skip("metrics.json not found")
        with open("metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        required_keys = {"f1", "accuracy", "precision", "recall"}
        missing = required_keys - set(metrics.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_confusion_matrix_exists(self):
        assert os.path.exists("confusion_matrix.png"), "confusion_matrix.png not found"

    def test_confusion_matrix_not_empty(self):
        if not os.path.exists("confusion_matrix.png"):
            pytest.skip("confusion_matrix.png not found")
        assert os.path.getsize("confusion_matrix.png") > 1000


class TestQualityGate:
    def test_quality_gate_f1(self):
        if not os.path.exists("metrics.json"):
            pytest.skip("metrics.json not found")
        threshold = float(os.getenv("F1_THRESHOLD", "0.70"))
        with open("metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        f1 = float(metrics["f1"])
        assert f1 >= threshold, f"Quality Gate FAILED: f1={f1:.4f} < {threshold:.2f}"

    def test_quality_gate_accuracy(self):
        if not os.path.exists("metrics.json"):
            pytest.skip("metrics.json not found")
        threshold = float(os.getenv("ACCURACY_THRESHOLD", "0.90"))
        with open("metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        accuracy = float(metrics["accuracy"])
        assert accuracy >= threshold, f"Quality Gate FAILED: accuracy={accuracy:.4f} < {threshold:.2f}"

    def test_metrics_values_in_range(self):
        if not os.path.exists("metrics.json"):
            pytest.skip("metrics.json not found")
        with open("metrics.json", "r", encoding="utf-8") as f:
            metrics = json.load(f)
        for key, value in metrics.items():
            assert 0.0 <= float(value) <= 1.0, f"Metric '{key}' out of range: {value}"