import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CI_MODE = os.getenv("CI", "false").lower() == "true"
MAX_ROWS = int(os.getenv("MAX_ROWS", "0"))  


def load_data(max_rows=0):
    print("Loading processed data...")
    train_df = pd.read_csv("data/processed/train.csv")
    val_df = pd.read_csv("data/processed/val.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    if max_rows > 0:
        print(f"CI mode: using {max_rows} rows from train set")
        train_df = train_df.head(max_rows)

    X_train = train_df.drop("Class", axis=1)
    y_train = train_df["Class"]
    X_val = val_df.drop("Class", axis=1)
    y_val = val_df["Class"]
    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"]

    return X_train, y_train, X_val, y_val, X_test, y_test


def main(n_estimators, max_depth):
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(
        max_rows=MAX_ROWS if CI_MODE else 0
    )

    if CI_MODE:
        n_estimators = min(n_estimators, 20)
        print(f"CI mode: n_estimators reduced to {n_estimators}")

    mlflow.set_experiment("Credit_Card_Fraud_Detection")

    with mlflow.start_run():
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("ci_mode", CI_MODE)

        print(
            f"Training RandomForest (n_estimators={n_estimators},"
            f" max_depth={max_depth})..."
        )
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)

        def compute_metrics(X_data, y_true, prefix):
            y_pred = model.predict(X_data)
            metrics = {
                f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
                f"{prefix}_f1": float(f1_score(y_true, y_pred, zero_division=0)),
                f"{prefix}_precision": float(
                    precision_score(y_true, y_pred, zero_division=0)
                ),
                f"{prefix}_recall": float(
                    recall_score(y_true, y_pred, zero_division=0)
                ),
            }
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            return y_pred, metrics

        compute_metrics(X_train, y_train, "train")
        y_val_pred, val_metrics = compute_metrics(X_val, y_val, "val")
        _, test_metrics = compute_metrics(X_test, y_test, "test")

        os.makedirs("models", exist_ok=True)
        model_path = "models/model.pkl"
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)
        print(f"Model saved: {model_path}")

        metrics_to_save = {
            "accuracy": test_metrics["test_accuracy"],
            "f1": test_metrics["test_f1"],
            "precision": test_metrics["test_precision"],
            "recall": test_metrics["test_recall"],
            "val_f1": val_metrics["val_f1"],
            "val_accuracy": val_metrics["val_accuracy"],
        }
        metrics_path = "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as mf:
            json.dump(metrics_to_save, mf, ensure_ascii=False, indent=2)
        mlflow.log_artifact(metrics_path)
        print(f"Metrics saved: {metrics_path}")
        print(json.dumps(metrics_to_save, indent=2))

        # ── Збереження confusion_matrix.png ───────────────────────────────
        cm = confusion_matrix(y_val, y_val_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Normal", "Fraud"],
                    yticklabels=["Normal", "Fraud"])
        plt.title("Confusion Matrix (Validation set)")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        cm_path = "confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path)
        print(f"Confusion matrix saved: {cm_path}")

        mlflow.sklearn.log_model(model, "random_forest_model")
        print("Run completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=15)
    parser.add_argument("--max_depth", type=int, default=6)
    args = parser.parse_args()
    main(args.n_estimators, args.max_depth)