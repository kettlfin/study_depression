"""
modeling.py — Построение, обучение и оценка моделей.

Структура проекта предполагается следующей:
    <project_root>/
        data/processed/train.csv
        data/processed/val.csv
        data/processed/test.csv
        models/
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    cwd = Path.cwd()
    if cwd.name == "notebooks":
        return cwd.parent
    return cwd


def get_paths(project_root: Path | None = None) -> dict[str, Path]:
    root = project_root or get_project_root()
    return {
        "train": root / "data" / "processed" / "train.csv",
        "val": root / "data" / "processed" / "val.csv",
        "test": root / "data" / "processed" / "test.csv",
        "models_dir": root / "models",
    }


# ---------------------------------------------------------------------------
# Загрузка данных
# ---------------------------------------------------------------------------

def load_splits(
    project_root: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Загружает train/val/test из data/processed/.

    Возвращает
    ----------
    X_train, y_train, X_val, y_val, X_test, y_test
    """
    paths = get_paths(project_root)
    train_df = pd.read_csv(paths["train"])
    val_df = pd.read_csv(paths["val"])
    test_df = pd.read_csv(paths["test"])

    X_train = train_df.drop("Depression", axis=1)
    y_train = train_df["Depression"]
    X_val = val_df.drop("Depression", axis=1)
    y_val = val_df["Depression"]
    X_test = test_df.drop("Depression", axis=1)
    y_test = test_df["Depression"]

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# Признаки
# ---------------------------------------------------------------------------

# Исходные (без engineered) — для baseline
ORIGINAL_COLS = [
    "Gender", "Age", "City", "Profession",
    "Academic Pressure", "Work Pressure",
    "CGPA", "Study Satisfaction", "Job Satisfaction",
    "Sleep Duration", "Dietary Habits", "Degree",
    "Have you ever had suicidal thoughts ?",
    "Work/Study Hours", "Financial Stress",
    "Family History of Mental Illness",
]

# Engineered признаки (добавляются в preprocessing)
ENGINEERED_COLS = ["Overall_Pressure", "Satisfaction_Ratio"]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Строит ColumnTransformer:
      - числовые → StandardScaler
      - категориальные → OneHotEncoder
    Определяет типы автоматически по переданному датафрейму.
    """
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_features),
        ],
        remainder="drop",
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Оценка
# ---------------------------------------------------------------------------

def evaluate(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    label: str = "Model",
    verbose: bool = True,
) -> dict[str, float]:
    """
    Вычисляет F1 и ROC-AUC для переданной модели.

    Возвращает dict с ключами 'f1' и 'roc_auc'.
    """
    y_pred = model.predict(X)

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, y_prob)
    else:
        auc = float("nan")

    f1 = f1_score(y, y_pred)

    if verbose:
        print(f"\n{'=' * 50}")
        print(f"{label}")
        print(f"  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(classification_report(y, y_pred))

    return {"f1": f1, "roc_auc": auc}


# ---------------------------------------------------------------------------
# Baseline
# ---------------------------------------------------------------------------

def build_baseline(X_train: pd.DataFrame) -> Pipeline:
    """
    Baseline: логистическая регрессия только на исходных признаках.
    """
    # Используем только исходные колонки, если они есть
    cols = [c for c in ORIGINAL_COLS if c in X_train.columns]
    X_sub = X_train[cols]
    preprocessor = build_preprocessor(X_sub)

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(random_state=42, max_iter=1000)),
    ])
    return pipeline, cols


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[Pipeline, dict[str, float]]:
    """Обучает baseline и возвращает модель + метрики на val."""
    pipeline, cols = build_baseline(X_train)
    pipeline.fit(X_train[cols], y_train)
    metrics = evaluate(pipeline, X_val[cols], y_val, label="Baseline: Logistic Regression")
    return pipeline, metrics


# ---------------------------------------------------------------------------
# Набор моделей
# ---------------------------------------------------------------------------

def get_model_candidates() -> dict[str, Any]:
    """Возвращает словарь {name: classifier} для экспериментов."""
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss", verbosity=0),
        "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=42),
    }


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[dict[str, Pipeline], pd.DataFrame]:
    """
    Обучает все модели-кандидаты на полном наборе признаков (включая engineered).

    Возвращает
    ----------
    pipelines : dict {name: fitted Pipeline}
    results   : DataFrame с метриками
    """
    preprocessor = build_preprocessor(X_train)
    candidates = get_model_candidates()
    pipelines: dict[str, Pipeline] = {}
    records = []

    for name, clf in candidates.items():
        pipe = Pipeline(steps=[
            ("preprocessor", build_preprocessor(X_train)),
            ("classifier", clf),
        ])
        pipe.fit(X_train, y_train)
        metrics = evaluate(pipe, X_val, y_val, label=name)
        metrics["model"] = name
        records.append(metrics)
        pipelines[name] = pipe

    results_df = pd.DataFrame(records).set_index("model").sort_values("f1", ascending=False)
    print("\nСводная таблица результатов (Val):")
    print(results_df.to_string())
    return pipelines, results_df


# ---------------------------------------------------------------------------
# Ансамбль
# ---------------------------------------------------------------------------

def train_ensemble(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[VotingClassifier, dict[str, float]]:
    """
    Строит Soft Voting Ensemble из переданных пайплайнов.

    Возвращает
    ----------
    ensemble : обученный VotingClassifier
    metrics  : dict с ключами 'f1' и 'roc_auc'
    """
    estimators = list(pipelines.items())
    ensemble = VotingClassifier(estimators=estimators, voting="soft")
    ensemble.fit(X_train, y_train)

    y_pred = ensemble.predict(X_val)
    y_prob = ensemble.predict_proba(X_val)[:, 1]
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    print(f"\n{'=' * 50}")
    print("Ensemble (Soft Voting)")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(classification_report(y_val, y_pred))

    return ensemble, {"f1": f1, "roc_auc": auc}


# ---------------------------------------------------------------------------
# Сохранение / загрузка модели
# ---------------------------------------------------------------------------

def save_model(model: Any, name: str, project_root: Path | None = None) -> Path:
    """Сохраняет модель в models/<name>.pkl."""
    paths = get_paths(project_root)
    paths["models_dir"].mkdir(parents=True, exist_ok=True)
    out_path = paths["models_dir"] / f"{name}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Модель сохранена: {out_path}")
    return out_path


def load_model(name: str, project_root: Path | None = None) -> Any:
    """Загружает модель из models/<name>.pkl."""
    paths = get_paths(project_root)
    model_path = paths["models_dir"] / f"{name}.pkl"
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print(f"Модель загружена: {model_path}")
    return model


# ---------------------------------------------------------------------------
# Полный пайплайн моделирования
# ---------------------------------------------------------------------------

def run_modeling(project_root: Path | None = None) -> dict[str, Any]:
    """
    Запускает полный пайплайн:
        load splits → baseline → all models → ensemble → save best

    Возвращает словарь с обученными моделями и метриками.
    """
    X_train, y_train, X_val, y_val, X_test, y_test = load_splits(project_root)

    # Baseline
    baseline, baseline_metrics = train_baseline(X_train, y_train, X_val, y_val)

    # Все модели
    pipelines, results_df = train_all_models(X_train, y_train, X_val, y_val)

    # Ансамбль
    ensemble, ensemble_metrics = train_ensemble(pipelines, X_train, y_train, X_val, y_val)

    # Выбираем лучшую модель по F1 на val
    best_name = results_df["f1"].idxmax()
    best_f1 = results_df.loc[best_name, "f1"]
    ensemble_f1 = ensemble_metrics["f1"]

    if ensemble_f1 >= best_f1:
        best_model = ensemble
        best_model_name = "ensemble"
    else:
        best_model = pipelines[best_name]
        best_model_name = best_name

    print(f"\nЛучшая модель: {best_model_name}")
    save_model(best_model, best_model_name, project_root)

    # Финальная оценка на test
    print("\n--- Финальная оценка на тестовой выборке ---")
    test_metrics = evaluate(best_model, X_test, y_test, label=f"Best: {best_model_name}")

    return {
        "baseline": baseline,
        "baseline_metrics": baseline_metrics,
        "pipelines": pipelines,
        "results_df": results_df,
        "ensemble": ensemble,
        "ensemble_metrics": ensemble_metrics,
        "best_model": best_model,
        "best_model_name": best_model_name,
        "test_metrics": test_metrics,
    }
