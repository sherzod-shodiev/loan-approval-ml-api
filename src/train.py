"""
Обучение финальной модели (LightGBM) и сохранение пайплайна в model.joblib.

Как запустить (из корня репозитория):
    python src/train.py

Гиперпараметры ниже — результат RandomizedSearchCV (50 итераций, 5-fold CV,
scoring='roc_auc') из notebooks/pipeline_and_modeling.ipynb. Здесь они зашиты
намеренно: этот скрипт воспроизводимо собирает УЖЕ выбранную модель,
а не подбирает параметры заново при каждом запуске.
"""

import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from lightgbm import LGBMClassifier


# ==========================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ==========================================================================
def load_data():
    # TODO: поправь путь, если структура репозитория отличается.
    # Предполагается: repo_root/data/clean_data.csv, repo_root/src/train.py
    return pd.read_csv('data/clean_data.csv')


# ==========================================================================
# 2. НАСТРОЙКИ
# ==========================================================================
TARGET = "Is Rejected"

NUMERIC_FEATURES = [
    "Age", "Person Income", "Employee Experience", "Loan Amount",
    "Loan interest Rate", "Loan percentage", "Credit History", "Credit Score",
]
CATEGORICAL_FEATURES = ["Gender", "Education", "Home Ownership", "Loan Intent"]

TASK = "classification"
MODEL_PATH = "lgbm_model.joblib"

# Лучшие гиперпараметры из RandomizedSearchCV (best_score_ ROC-AUC = 0.9386)
BEST_PARAMS = dict(
    n_estimators=962,
    learning_rate=0.0214795108988544,
    max_depth=7,
    min_child_samples=23,
    num_leaves=62,
    reg_alpha=0.02023778954048508,
    reg_lambda=0.19132683954526483,
    subsample=0.7604171300129129,
    colsample_bytree=0.7248770666848828,
)


# ==========================================================================
# 3. СБОРКА ПАЙПЛАЙНА
# ==========================================================================
def build_pipeline():
    # LightGBM (деревья) не требует масштабирования числовых признаков —
    # только импутация пропусков, как в исходном ноутбуке (prep_tree).
    numeric_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
    ])
    categorical_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])

    model = LGBMClassifier(
        **BEST_PARAMS,
        class_weight="balanced",
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])


# ==========================================================================
# 4. ОБУЧЕНИЕ, ОЦЕНКА, СОХРАНЕНИЕ
# ==========================================================================
def main():
    df = load_data()
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y,
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    print("=" * 50)
    baseline = DummyClassifier(strategy="prior").fit(X_train, y_train)
    print(f"baseline accuracy: {accuracy_score(y_test, baseline.predict(X_test)):.3f}")

    pred = pipe.predict(X_test)
    print(f"model accuracy:    {accuracy_score(y_test, pred):.3f}")

    proba = pipe.predict_proba(X_test)[:, 1]
    print(f"model ROC-AUC:     {roc_auc_score(y_test, proba):.3f}")

    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"cross-val ROC-AUC: {cv.mean():.3f} +/- {cv.std():.3f}")
    print("=" * 50)

    joblib.dump(pipe, MODEL_PATH)
    print(f"\nМодель сохранена в {MODEL_PATH}")
    print("Признаки на вход API:", NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    print("\nТеперь запустите API:  uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
