"""
tests/test.py — Базовые тесты пайплайна предобработки и моделирования.

Запуск:
    pytest tests/test.py -v
или из корня проекта:
    python -m pytest tests/test.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import clean_data, engineer_features, split_and_save
from src.modeling import (
    build_preprocessor,
    evaluate,
    get_model_candidates,
    load_splits,
    train_baseline,
)


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Минимальный датафрейм, имитирующий реальный датасет."""
    rng = np.random.default_rng(42)
    n = 200

    df = pd.DataFrame({
        "id": range(n),
        "Gender": rng.choice(["Male", "Female"], n),
        "Age": rng.integers(18, 35, n).astype(float),
        "City": rng.choice(["Delhi", "Mumbai", "Pune"], n),
        "Profession": rng.choice(["Student", "Working Professional"], n),
        "Academic Pressure": rng.integers(1, 6, n).astype(float),
        "Work Pressure": rng.integers(1, 6, n).astype(float),
        "CGPA": rng.uniform(5.0, 10.0, n),
        "Study Satisfaction": rng.integers(1, 6, n).astype(float),
        "Job Satisfaction": rng.integers(1, 6, n).astype(float),
        "Sleep Duration": rng.choice(["5-6 hours", "7-8 hours", "Less than 5 hours"], n),
        "Dietary Habits": rng.choice(["Healthy", "Moderate", "Unhealthy"], n),
        "Degree": rng.choice(["B.Tech", "MBA", "M.Tech"], n),
        "Have you ever had suicidal thoughts ?": rng.choice(["Yes", "No"], n),
        "Work/Study Hours": rng.integers(1, 12, n).astype(float),
        "Financial Stress": rng.integers(1, 6, n).astype(float),
        "Family History of Mental Illness": rng.choice(["Yes", "No"], n),
        "Depression": rng.integers(0, 2, n),
    })
    return df


# ---------------------------------------------------------------------------
# Тесты предобработки
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_clean_data_removes_nulls(self, sample_df):
        df_with_nulls = sample_df.copy()
        df_with_nulls.loc[0, "Age"] = None
        df_clean = clean_data(df_with_nulls)
        assert df_clean.isnull().sum().sum() == 0

    def test_clean_data_removes_duplicates(self, sample_df):
        df_dup = pd.concat([sample_df, sample_df.iloc[:5]], ignore_index=True)
        df_clean = clean_data(df_dup)
        assert len(df_clean) == len(sample_df)

    def test_engineer_features_creates_columns(self, sample_df):
        df = clean_data(sample_df)
        df = engineer_features(df)
        assert "Overall_Pressure" in df.columns
        assert "Satisfaction_Ratio" in df.columns

    def test_overall_pressure_values(self, sample_df):
        df = clean_data(sample_df)
        df = engineer_features(df)
        expected = df["Academic Pressure"] + df["Work Pressure"]
        pd.testing.assert_series_equal(
            df["Overall_Pressure"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_split_sizes(self, tmp_path, sample_df):
        df = clean_data(sample_df)
        df = engineer_features(df)
        # Указываем tmp_path как корень проекта
        (tmp_path / "data" / "processed").mkdir(parents=True)

        train_df, val_df, test_df = split_and_save(
            df, project_root=tmp_path, test_size=0.15, val_size=0.15
        )
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == len(df)
        # Примерные доли с допуском ±5%
        assert abs(len(test_df) / total - 0.15) < 0.05
        assert abs(len(val_df) / total - 0.15) < 0.05

    def test_split_no_leakage(self, tmp_path, sample_df):
        df = clean_data(sample_df)
        df = engineer_features(df)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        train_df, val_df, test_df = split_and_save(df, project_root=tmp_path)

        # Индексы не пересекаются
        train_idx = set(train_df.index)
        val_idx = set(val_df.index)
        test_idx = set(test_df.index)
        assert train_idx.isdisjoint(val_idx)
        assert train_idx.isdisjoint(test_idx)
        assert val_idx.isdisjoint(test_idx)


# ---------------------------------------------------------------------------
# Тесты моделирования
# ---------------------------------------------------------------------------

class TestModeling:
    @pytest.fixture
    def splits(self, tmp_path, sample_df):
        df = clean_data(sample_df)
        df = engineer_features(df)
        (tmp_path / "data" / "processed").mkdir(parents=True)
        split_and_save(df, project_root=tmp_path)
        return load_splits(tmp_path)

    def test_build_preprocessor(self, sample_df):
        df = clean_data(sample_df)
        df = engineer_features(df)
        X = df.drop(columns=["id", "Depression"])
        pre = build_preprocessor(X)
        result = pre.fit_transform(X)
        assert result.shape[0] == len(X)
        assert result.shape[1] > 0

    def test_baseline_returns_pipeline(self, splits):
        X_train, y_train, X_val, y_val, _, _ = splits
        model, metrics = train_baseline(X_train, y_train, X_val, y_val)
        assert isinstance(model, Pipeline)
        assert "f1" in metrics
        assert "roc_auc" in metrics
        assert 0.0 <= metrics["f1"] <= 1.0
        assert 0.0 <= metrics["roc_auc"] <= 1.0

    def test_evaluate_returns_correct_keys(self, splits):
        X_train, y_train, X_val, y_val, _, _ = splits
        model, _ = train_baseline(X_train, y_train, X_val, y_val)

        from src.modeling import ORIGINAL_COLS
        cols = [c for c in ORIGINAL_COLS if c in X_val.columns]
        metrics = evaluate(model, X_val[cols], y_val, verbose=False)
        assert set(metrics.keys()) == {"f1", "roc_auc"}

    def test_model_candidates_exist(self):
        candidates = get_model_candidates()
        assert len(candidates) >= 4
        expected = {"RandomForest", "XGBoost", "LightGBM", "GradientBoosting"}
        assert expected.issubset(set(candidates.keys()))

    def test_save_and_load_model(self, tmp_path, splits):
        from src.modeling import save_model, load_model
        X_train, y_train, X_val, y_val, _, _ = splits
        model, _ = train_baseline(X_train, y_train, X_val, y_val)

        (tmp_path / "models").mkdir()
        save_model(model, "test_model", tmp_path)
        loaded = load_model("test_model", tmp_path)

        from src.modeling import ORIGINAL_COLS
        cols = [c for c in ORIGINAL_COLS if c in X_val.columns]
        preds_original = model.predict(X_val[cols])
        preds_loaded = loaded.predict(X_val[cols])
        np.testing.assert_array_equal(preds_original, preds_loaded)
