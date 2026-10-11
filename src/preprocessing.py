"""
preprocessing.py — Загрузка, очистка и подготовка данных.

Структура проекта предполагается следующей:
    <project_root>/
        data/raw/Student Depression Dataset.csv
        data/processed/train.csv
        data/processed/val.csv
        data/processed/test.csv
        report/images/
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Пути
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Возвращает корень проекта независимо от CWD."""
    # Если запускаем из notebooks/ — подняться на уровень выше
    cwd = Path.cwd()
    if cwd.name == "notebooks":
        return cwd.parent
    return cwd


def get_paths(project_root: Path | None = None) -> dict[str, Path]:
    root = project_root or get_project_root()
    return {
        "raw_csv": root / "data" / "raw" / "Student Depression Dataset.csv",
        "processed_dir": root / "data" / "processed",
        "images_dir": root / "report" / "images",
        "train": root / "data" / "processed" / "train.csv",
        "val": root / "data" / "processed" / "val.csv",
        "test": root / "data" / "processed" / "test.csv",
    }


# ---------------------------------------------------------------------------
# Загрузка
# ---------------------------------------------------------------------------

def load_raw_data(project_root: Path | None = None) -> pd.DataFrame:
    """Загружает исходный датасет из data/raw/."""
    paths = get_paths(project_root)
    raw_path = paths["raw_csv"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Исходный файл не найден: {raw_path}\n"
            "Поместите 'Student Depression Dataset.csv' в data/raw/"
        )

    df = pd.read_csv(raw_path)
    print(f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]} столбцов.")
    return df


# ---------------------------------------------------------------------------
# Очистка
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = [
    "Gender", "City", "Profession", "Sleep Duration",
    "Dietary Habits", "Degree",
    "Have you ever had suicidal thoughts ?",
    "Family History of Mental Illness",
]

ORDINAL_COLS = [
    "Academic Pressure", "Work Pressure",
    "Study Satisfaction", "Job Satisfaction",
    "Work/Study Hours", "Financial Stress",
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Очистка:
      - удаление строк с пропусками
      - удаление дубликатов
      - приведение типов
    """
    df = df.copy()

    # Пропуски
    before = len(df)
    df.dropna(inplace=True)
    print(f"Удалено строк с пропусками: {before - len(df)}")

    # Дубликаты
    before = len(df)
    df.drop_duplicates(inplace=True)
    print(f"Удалено дубликатов: {before - len(df)}")

    # Типы
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    print(f"Итог после очистки: {df.shape[0]} строк, {df.shape[1]} столбцов.")
    return df


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Создаёт новые признаки:
      - Overall_Pressure  = Academic Pressure + Work Pressure
      - Satisfaction_Ratio = Study Satisfaction / (Job Satisfaction + 1e-6)
    """
    df = df.copy()

    df["Overall_Pressure"] = df["Academic Pressure"] + df["Work Pressure"]
    df["Satisfaction_Ratio"] = (
        df["Study Satisfaction"] / (df["Job Satisfaction"] + 1e-6)
    )

    n_features = df.shape[1] - 2  # без id и Depression
    print(f"Количество признаков после feature engineering: {n_features}")
    return df


# ---------------------------------------------------------------------------
# Визуализации
# ---------------------------------------------------------------------------

def plot_target_distribution(df: pd.DataFrame, save_dir: Path | None = None) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        x="Depression", data=df, palette="viridis",
        hue="Depression", legend=False, ax=ax,
    )
    ax.set_title("Распределение целевой переменной: Наличие депрессии")
    ax.set_xlabel("Депрессия (0: Нет, 1: Да)")
    ax.set_ylabel("Количество студентов")
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "target_distribution.png", dpi=150)
    plt.show()


def plot_numerical_distributions(df: pd.DataFrame, save_dir: Path | None = None) -> None:
    numerical_cols = [
        "Age", "CGPA", "Academic Pressure", "Work Pressure",
        "Work/Study Hours", "Financial Stress", "Overall_Pressure",
    ]
    cols_present = [c for c in numerical_cols if c in df.columns]
    n = len(cols_present)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cols_present):
        sns.histplot(df[col], kde=True, bins=20, color="skyblue", ax=axes[i])
        axes[i].set_title(f"Распределение {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Частота")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "numerical_distributions.png", dpi=150)
    plt.show()


def plot_categorical_distributions(df: pd.DataFrame, save_dir: Path | None = None) -> None:
    cat_cols = [
        "Gender", "Profession", "Sleep Duration", "Degree",
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness",
    ]
    cols_present = [c for c in cat_cols if c in df.columns]
    n = len(cols_present)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
    axes = axes.flatten()

    for i, col in enumerate(cols_present):
        order = df[col].value_counts().index
        sns.countplot(
            y=col, data=df, palette="magma",
            order=order, hue=col, legend=False, ax=axes[i],
        )
        axes[i].set_title(f"Распределение {col}")
        axes[i].set_xlabel("Количество студентов")
        axes[i].set_ylabel(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "categorical_distributions.png", dpi=150)
    plt.show()


def plot_features_vs_target_numerical(
    df: pd.DataFrame, save_dir: Path | None = None
) -> None:
    numerical_cols = [
        "Age", "CGPA", "Academic Pressure", "Work Pressure",
        "Work/Study Hours", "Financial Stress", "Overall_Pressure",
    ]
    cols_present = [c for c in numerical_cols if c in df.columns]
    n = len(cols_present)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
    axes = axes.flatten()

    for i, col in enumerate(cols_present):
        sns.boxplot(
            x="Depression", y=col, data=df,
            palette="coolwarm", hue="Depression", legend=False, ax=axes[i],
        )
        axes[i].set_title(f"{col} vs. Depression")
        axes[i].set_xlabel("Депрессия (0: Нет, 1: Да)")
        axes[i].set_ylabel(col)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "numerical_vs_target.png", dpi=150)
    plt.show()


def plot_features_vs_target_categorical(
    df: pd.DataFrame, save_dir: Path | None = None
) -> None:
    cat_cols = [
        "Gender", "Profession", "Sleep Duration", "Degree",
        "Have you ever had suicidal thoughts ?",
        "Family History of Mental Illness",
    ]
    cols_present = [c for c in cat_cols if c in df.columns]
    n = len(cols_present)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
    axes = axes.flatten()

    for i, col in enumerate(cols_present):
        cross_tab = (
            pd.crosstab(df[col], df["Depression"], normalize="index") * 100
        )
        cross_tab.plot(kind="bar", stacked=True, colormap="viridis", ax=axes[i])
        axes[i].set_title(f"Депрессия по {col}")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Доля (%)")
        axes[i].tick_params(axis="x", rotation=45)
        axes[i].legend(title="Depression", labels=["Нет", "Да"])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "categorical_vs_target.png", dpi=150)
    plt.show()


def plot_correlation_matrix(df: pd.DataFrame, save_dir: Path | None = None) -> None:
    num_cols = [
        "Age", "CGPA", "Academic Pressure", "Work Pressure",
        "Work/Study Hours", "Financial Stress", "Overall_Pressure",
        "Depression", "Study Satisfaction", "Job Satisfaction", "Satisfaction_Ratio",
    ]
    cols_present = [c for c in num_cols if c in df.columns]
    corr = df[cols_present].corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Матрица корреляции числовых признаков")
    plt.tight_layout()
    if save_dir:
        fig.savefig(save_dir / "correlation_matrix.png", dpi=150)
    plt.show()


def run_all_plots(df: pd.DataFrame, save_dir: Path | None = None) -> None:
    """Запускает все визуализации."""
    plot_target_distribution(df, save_dir)
    plot_numerical_distributions(df, save_dir)
    plot_categorical_distributions(df, save_dir)
    plot_features_vs_target_numerical(df, save_dir)
    plot_features_vs_target_categorical(df, save_dir)
    plot_correlation_matrix(df, save_dir)


# ---------------------------------------------------------------------------
# Сплит и сохранение
# ---------------------------------------------------------------------------

def split_and_save(
    df: pd.DataFrame,
    project_root: Path | None = None,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Стратифицированный сплит Train/Val/Test и сохранение в data/processed/.

    Параметры
    ----------
    df          : датасет после clean_data + engineer_features
    test_size   : доля теста (от всего датасета)
    val_size    : доля валидации (от всего датасета)
    random_state: сид

    Возвращает
    ----------
    train_df, val_df, test_df
    """
    paths = get_paths(project_root)
    paths["processed_dir"].mkdir(parents=True, exist_ok=True)

    # Удаляем id — нечинформативный идентификатор
    drop_cols = [c for c in ["id"] if c in df.columns]
    X = df.drop(columns=drop_cols + ["Depression"])
    y = df["Depression"]

    # Сначала отделяем тест
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Из оставшегося выделяем val так, чтобы итоговая доля ≈ val_size от всего
    relative_val_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=y_train_val,
    )

    train_df = X_train.assign(Depression=y_train)
    val_df = X_val.assign(Depression=y_val)
    test_df = X_test.assign(Depression=y_test)

    train_df.to_csv(paths["train"], index=False)
    val_df.to_csv(paths["val"], index=False)
    test_df.to_csv(paths["test"], index=False)

    print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    print("Файлы сохранены в data/processed/")
    return train_df, val_df, test_df


# ---------------------------------------------------------------------------
# Полный пайплайн предобработки
# ---------------------------------------------------------------------------

def run_preprocessing(project_root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Запускает полный пайплайн:
        load → clean → engineer → split & save
    """
    root = project_root or get_project_root()
    paths = get_paths(root)
    paths["images_dir"].mkdir(parents=True, exist_ok=True)

    df = load_raw_data(root)
    df = clean_data(df)
    df = engineer_features(df)
    train_df, val_df, test_df = split_and_save(df, root)
    return train_df, val_df, test_df
