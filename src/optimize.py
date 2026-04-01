"""
src/optimize.py
Лабораторна робота №3 — Гіперпараметрична оптимізація з Optuna + MLflow + Hydra
Датасет: Credit Card Fraud Detection (creditcard.csv)
Модель: RandomForestClassifier / LogisticRegression
"""

import json
import os
import random
from typing import Any, Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
from omegaconf import DictConfig, OmegaConf
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import hydra
from hydra.utils import to_absolute_path

# ──────────────────────────────────────────────────────────────────────────────
# Допоміжні функції
# ──────────────────────────────────────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    """Фіксація seed для відтворюваності."""
    random.seed(seed)
    np.random.seed(seed)


def load_processed_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Завантаження даних з трьох окремих CSV файлів: train.csv, val.csv, test.csv
    """
    abs_path = to_absolute_path(path)
    base_dir = os.path.dirname(abs_path)

    train = pd.read_csv(os.path.join(base_dir, "train.csv"))
    val   = pd.read_csv(os.path.join(base_dir, "val.csv"))
    test  = pd.read_csv(os.path.join(base_dir, "test.csv"))

    # Об'єднуємо val у train для навчання
    train_full = pd.concat([train, val], ignore_index=True)

    X_train = train_full.drop(columns=["Class"]).values
    y_train = train_full["Class"].values
    X_test  = test.drop(columns=["Class"]).values
    y_test  = test["Class"].values

    return X_train, X_test, y_train, y_test

def build_model(model_type: str, params: Dict[str, Any], seed: int) -> Any:
    """Побудова моделі за типом та гіперпараметрами."""
    if model_type == "random_forest":
        return RandomForestClassifier(random_state=seed, n_jobs=-1, **params)
    if model_type == "logistic_regression":
        clf = LogisticRegression(random_state=seed, max_iter=500, **params)
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])
    raise ValueError(f"Unknown model_type='{model_type}'. Use 'random_forest' or 'logistic_regression'.")


def evaluate(model: Any, X_train, y_train, X_test, y_test, metric: str) -> float:
    """Тренування моделі та обчислення метрики на тесті."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if metric == "f1":
        return float(f1_score(y_test, y_pred, average="binary"))
    if metric == "roc_auc":
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        else:
            y_score = model.decision_function(X_test)
        return float(roc_auc_score(y_test, y_score))

    raise ValueError(f"Unsupported metric '{metric}'. Use 'f1' or 'roc_auc'.")


def evaluate_cv(model: Any, X, y, metric: str, seed: int, n_splits: int = 5) -> float:
    """Оцінка через крос-валідацію."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = []
    for train_idx, test_idx in cv.split(X, y):
        m = clone(model)
        scores.append(evaluate(m, X[train_idx], y[train_idx], X[test_idx], y[test_idx], metric))
    return float(np.mean(scores))


def make_sampler(sampler_name: str, seed: int, grid_space: Dict = None) -> optuna.samplers.BaseSampler:
    """Створення sampler-а для Optuna."""
    name = sampler_name.lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(seed=seed)
    if name == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if name == "grid":
        if not grid_space:
            raise ValueError("For sampler='grid' you must provide grid_space.")
        return optuna.samplers.GridSampler(search_space=grid_space)
    raise ValueError(f"Unknown sampler '{sampler_name}'. Use: tpe, random, grid.")


def suggest_params(trial: optuna.Trial, model_type: str, cfg: DictConfig) -> Dict[str, Any]:
    """Пропозиція гіперпараметрів для конкретного trial."""
    if model_type == "random_forest":
        space = cfg.hpo.random_forest
        return {
            "n_estimators":      trial.suggest_int("n_estimators",      space.n_estimators.low,      space.n_estimators.high),
            "max_depth":         trial.suggest_int("max_depth",          space.max_depth.low,          space.max_depth.high),
            "min_samples_split": trial.suggest_int("min_samples_split",  space.min_samples_split.low,  space.min_samples_split.high),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf",   space.min_samples_leaf.low,   space.min_samples_leaf.high),
        }
    if model_type == "logistic_regression":
        space = cfg.hpo.logistic_regression
        return {
            "C":       trial.suggest_float("C", space.C.low, space.C.high, log=True),
            "solver":  trial.suggest_categorical("solver",  list(space.solver)),
            "penalty": trial.suggest_categorical("penalty", list(space.penalty)),
        }
    raise ValueError(f"Unknown model_type='{model_type}'.")


# ──────────────────────────────────────────────────────────────────────────────
# Objective factory
# ──────────────────────────────────────────────────────────────────────────────

def objective_factory(cfg: DictConfig, X_train, X_test, y_train, y_test):
    """Фабрика objective функції для Optuna."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg.model.type, cfg)

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number",  trial.number)
            mlflow.set_tag("model_type",    cfg.model.type)
            mlflow.set_tag("sampler",       cfg.hpo.sampler)
            mlflow.set_tag("seed",          cfg.seed)
            mlflow.log_params(params)

            model = build_model(cfg.model.type, params=params, seed=cfg.seed)

            if cfg.hpo.use_cv:
                X = np.concatenate([X_train, X_test], axis=0)
                y = np.concatenate([y_train, y_test], axis=0)
                score = evaluate_cv(model, X, y, metric=cfg.hpo.metric,
                                    seed=cfg.seed, n_splits=cfg.hpo.cv_folds)
            else:
                score = evaluate(model, X_train, y_train, X_test, y_test,
                                 metric=cfg.hpo.metric)

            mlflow.log_metric(cfg.hpo.metric, score)
            print(f"  Trial {trial.number:03d} | {cfg.hpo.metric}={score:.5f} | params={params}")

        return score

    return objective


# ──────────────────────────────────────────────────────────────────────────────
# Порівняння sampler-ів
# ──────────────────────────────────────────────────────────────────────────────

def run_sampler_comparison(cfg: DictConfig, X_train, X_test, y_train, y_test) -> None:
    """
    Запускає HPO для TPE та Random sampler і порівнює результати.
    Результати зберігаються у models/sampler_comparison.json
    """
    results = {}
    samplers_to_compare = ["tpe", "random"]

    for sampler_name in samplers_to_compare:
        print(f"\n{'='*60}")
        print(f"  Sampler: {sampler_name.upper()}")
        print(f"{'='*60}")

        sampler = make_sampler(sampler_name, seed=cfg.seed)
        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=sampler,
            study_name=f"comparison_{sampler_name}"
        )

        with mlflow.start_run(run_name=f"comparison_{sampler_name}") as run:
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler",    sampler_name)
            mlflow.set_tag("role",       "sampler_comparison")

            obj = objective_factory(cfg, X_train, X_test, y_train, y_test)
            study.optimize(obj, n_trials=cfg.hpo.n_trials)

            best = study.best_trial
            all_values = [t.value for t in study.trials if t.value is not None]

            mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best.value))
            mlflow.log_metric(f"mean_{cfg.hpo.metric}", float(np.mean(all_values)))
            mlflow.log_metric(f"std_{cfg.hpo.metric}",  float(np.std(all_values)))

            results[sampler_name] = {
                "best_value": float(best.value),
                "mean_value": float(np.mean(all_values)),
                "std_value":  float(np.std(all_values)),
                "best_params": best.params,
                "run_id": run.info.run_id,
            }

    os.makedirs("models", exist_ok=True)
    with open("models/sampler_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n=== Порівняння sampler-ів ===")
    for name, res in results.items():
        print(f"  {name.upper():8s}  best={res['best_value']:.5f}  "
              f"mean={res['mean_value']:.5f}  std={res['std_value']:.5f}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(cfg: DictConfig) -> None:
    set_global_seed(cfg.seed)
    print(OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    X_train, X_test, y_train, y_test = load_processed_data(cfg.data.processed_path)
    print(f"Дані завантажено: X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Розподіл класів (train): {np.bincount(y_train.astype(int))}")

    # ── Grid sampler: побудова grid_space ──────────────────────────────────────
    grid_space = None
    if cfg.hpo.sampler.lower() == "grid":
        if cfg.model.type == "random_forest":
            grid_space = {
                "n_estimators":      list(cfg.hpo.grid.random_forest.n_estimators),
                "max_depth":         list(cfg.hpo.grid.random_forest.max_depth),
                "min_samples_split": list(cfg.hpo.grid.random_forest.min_samples_split),
                "min_samples_leaf":  list(cfg.hpo.grid.random_forest.min_samples_leaf),
            }
        elif cfg.model.type == "logistic_regression":
            grid_space = {
                "C":       list(cfg.hpo.grid.logistic_regression.C),
                "solver":  list(cfg.hpo.grid.logistic_regression.solver),
                "penalty": list(cfg.hpo.grid.logistic_regression.penalty),
            }

    sampler = make_sampler(cfg.hpo.sampler, seed=cfg.seed, grid_space=grid_space)

    # ── Основний HPO пайплайн (parent run) ────────────────────────────────────
    with mlflow.start_run(run_name="hpo_parent") as parent_run:
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.set_tag("sampler",    cfg.hpo.sampler)
        mlflow.set_tag("seed",       cfg.seed)
        mlflow.set_tag("n_trials",   cfg.hpo.n_trials)
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config_resolved.json")

        study = optuna.create_study(
            direction=cfg.hpo.direction,
            sampler=sampler,
            study_name="main_hpo_study"
        )
        objective = objective_factory(cfg, X_train, X_test, y_train, y_test)

        print(f"\nЗапускаємо оптимізацію: {cfg.hpo.n_trials} trials, sampler={cfg.hpo.sampler}")
        study.optimize(objective, n_trials=cfg.hpo.n_trials)

        best_trial = study.best_trial
        print(f"\nНайкращий trial #{best_trial.number}: {cfg.hpo.metric}={best_trial.value:.5f}")
        print(f"Параметри: {best_trial.params}")

        mlflow.log_metric(f"best_{cfg.hpo.metric}", float(best_trial.value))
        mlflow.log_dict(best_trial.params, "best_params.json")

        # ── Фінальне тренування з найкращими параметрами ──────────────────────
        best_model = build_model(cfg.model.type, params=best_trial.params, seed=cfg.seed)
        final_score = evaluate(best_model, X_train, y_train, X_test, y_test,
                               metric=cfg.hpo.metric)
        mlflow.log_metric(f"final_{cfg.hpo.metric}", final_score)
        print(f"Фінальний score (re-train): {cfg.hpo.metric}={final_score:.5f}")

        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_model.pkl")
        mlflow.log_artifact("models/best_model.pkl")

        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, artifact_path="model")

        if cfg.mlflow.register_model:
            model_uri = f"runs:/{parent_run.info.run_id}/model"
            client = mlflow.tracking.MlflowClient()
            mv = mlflow.register_model(model_uri, cfg.mlflow.model_name)
            client.transition_model_version_stage(
                name=cfg.mlflow.model_name,
                version=mv.version,
                stage=cfg.mlflow.stage,
            )
            print(f"Модель зареєстрована: {cfg.mlflow.model_name} v{mv.version} → {cfg.mlflow.stage}")

    # ── Порівняння TPE vs Random ───────────────────────────────────────────────
    print("\n--- Запускаємо порівняння sampler-ів (TPE vs Random) ---")
    run_sampler_comparison(cfg, X_train, X_test, y_train, y_test)


# ──────────────────────────────────────────────────────────────────────────────
# Hydra entry point
# ──────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()