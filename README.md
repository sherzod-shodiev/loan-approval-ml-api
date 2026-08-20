# Loan Approval ML API — Сервис скоринга кредитных заявок

FastAPI-сервис с моделью LightGBM для бинарной классификации заявок на кредит (одобрить / отказать) на основе анкетных и финансовых данных заёмщика.

* 📄 **Swagger UI (ручное тестирование):** `/docs`
* ❤️ **Health-check:** `GET /health`
* 🎯 **Предсказание:** `POST /predict`

## 📁 Структура проекта

```text
My_Project/
├── data/
│   ├── loan_data_new.csv        # Оригинальный датасет (45 000 строк)
│   └── clean_data.csv           # Очищенный датасет после EDA (22 135 строк)
├── notebooks/
│   ├── eda.ipynb                    # Исследовательский анализ данных
│   └── pipeline_and_modeling.ipynb  # Сравнение моделей, тюнинг, финальная оценка
├── models/
│   └── lgbm_model.joblib        # Обученный пайплайн (препроцессинг + LightGBM)
├── src/
│   └── train.py                 # Обучение финальной модели, сохранение в models/
├── app/
│   └── main.py                  # FastAPI-сервис (/health, /predict)
├── requirements.txt              # Зависимости для деплоя API
├── requirements-dev.txt          # Доп. зависимости для ноутбуков (jupyter, catboost, statsmodels, seaborn...)
└── README.md

```

## 📊 Данные и предобработка

Исходный датасет — 45 000 заявок на кредит, 14 колонок, без пропусков и дубликатов. Целевая переменная переименована из Loan Status в Is Rejected (1 = отказ, 0 = одобрение) для однозначности.

В ходе EDA было сделано три ключевых решения по очистке:

1. **Физически невозможные значения.** Удалено 10 строк с Age > 80 или Employee Experience > 60.
2. **Опечатка в названии колонки.** Home Onwership → Home Ownership.
3. **Устранение утечки данных (data leakage).** Признак Previous Loan оказался детерминированным маркером: у всех заёмщиков с Previous Loan = Yes заявка была одобрена в 100% случаев (Mutual Information ≈ 0.191 — почти треть энтропии таргета). Такая связь нереалистична для банковского скоринга и, вероятнее всего, является артефактом генерации синтетических данных. Решение: все строки с Previous Loan = Yes удалены из выборки, сама колонка — тоже (после удаления она стала бы константой). Из-за этого размер датасета сократился до 22 135 строк, а баланс классов сместился с 78% / 22% до почти сбалансированных 55% / 45% — это ожидаемое следствие, а не ошибка.

⚠️ **Важно для продакшена:** модель обучена и валидна только для заёмщиков без предыдущих кредитов в системе (Previous Loan = No). Для повторных заёмщиков в исходных данных действовало жёсткое бизнес-правило (100% одобрение) — такие заявки не должны идти через ML-модель, а обрабатываться отдельным правилом.

Также был проведён анализ мультиколлинеарности: Age, Employee Experience и Credit History сильно коррелируют между собой (r > 0.8, «временной кластер»). Для линейной модели (LogReg) из этого кластера оставлен только Employee Experience, для градиентного бустинга — коллинеарность не устранялась (деревья к ней устойчивы).

### Итоговый набор признаков (12)

| Признак | Тип | Пример |
| --- | --- | --- |
| Age | number | 28 |
| Person Income | number | 60000 |
| Employee Experience | number | 5 |
| Loan Amount | number | 15000 |
| Loan interest Rate | number | 11.2 |
| Credit History | number | 6 |
| Credit Score | number | 720 |
| Gender | category | female / male |
| Education | category | High School / Associate / Bachelor / Master / Doctorate |
| Home Ownership | category | RENT / OWN / MORTGAGE / OTHER |
| Loan Intent | category | PERSONAL / EDUCATION / MEDICAL / VENTURE / HOMEIMPROVEMENT / DEBTCONSOLIDATION |

## 🧪 Методология моделирования

На train-выборке (5-fold CV, scoring='roc_auc') сравнивались:

| Модель | ROC-AUC (CV) |
| --- | --- |
| Dummy Classifier | 0.500 |
| LogReg (raw features) | 0.868 |
| LogReg (лог-трансформация скошенных признаков) | 0.874 |
| LightGBM | 0.933 |
| CatBoost (One-Hot) | 0.933 |
| CatBoost (нативные категории) | 0.933 |

Переход от линейной модели к градиентному бустингу дал прирост ROC-AUC ~0.06, а выбор конкретного алгоритма бустинга внутри тройки лидеров почти не влиял на качество (разброс < 0.001). Обе топовые модели дополнительно тюнились через RandomizedSearchCV (50 итераций, 5-fold CV): LightGBM — 0.9386 за ~3 минуты, CatBoost — 0.9339 за ~16 минут. Так как качество статистически неотличимо, для финальной модели выбран LightGBM — из-за существенно более быстрого подбора гиперпараметров и обучения.

### Качество модели (hold-out тест, 20% данных):

| Метрика | CV (train, 5-fold) | Test (hold-out) |
|---|---|---|
| ROC-AUC | 0.9386 | 0.9434 |
| PR-AUC | 0.9404 | 0.9435 |
| Accuracy | 0.8667 | 0.8735 |
| Precision | 0.8822 | 0.8883 |
| Recall | 0.8138 | 0.8235 |
| F1 | 0.8465 | 0.8547 |

Baseline (DummyClassifier, strategy='prior'): accuracy 0.548. Метрики на тесте практически совпадают с CV на train — переобучения не наблюдается.

## 🚀 Запуск локально

```bash
# 1. Клонировать репозиторий и перейти в папку
git clone <repo_url> && cd My_Project

# 2. Создать окружение и поставить зависимости
python -m venv venv
source venv/bin/activate          # Windows: venv\\Scripts\\activate
pip install -r requirements.txt   # для запуска API
pip install -r requirements-dev.txt  # если нужно перезапускать ноутбуки/train.py

# 3. Обучить модель (создаст models/lgbm_model.joblib)
python src/train.py

# 4. Запустить API
uvicorn app.main:app --reload

```

После запуска Swagger доступен на http://127.0.0.1:8000/docs.

## 🔌 Использование API

### Health-check

```bash
curl [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
# {"status": "ok"}

```

### Предсказание

```bash
curl -X POST [http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict) \\
  -H "Content-Type: application/json" \\
  -d '{
    "Age": 28,
    "Person Income": 60000,
    "Employee Experience": 5,
    "Loan Amount": 15000,
    "Loan interest Rate": 11.2,
    "Credit History": 6,
    "Credit Score": 720,
    "Gender": "Female",
    "Education": "Master",
    "Home Ownership": "Mortgage",
    "Loan Intent": "Education"
  }'

```

Ответ:

```json
{
  "prediction": 0,
  "probability": 0.132
}

```

`prediction = 1` — модель рекомендует отказ, `prediction = 0` — одобрение;

`probability` — вероятность класса «отказ» (порог по умолчанию 0.5).

## 🛠 Технологии

Python · pandas · scikit-learn · LightGBM · CatBoost (эксперименты) · FastAPI · Pydantic · Docker · Hugging Face Spaces

## ⚠️ Известные ограничения

* Порог классификации 0.5 задан по умолчанию, без явной оптимизации под бизнес-стоимость ошибок (FP/FN).
* Модель обучена только на заёмщиках без предыдущих кредитов в системе (см. раздел «Данные и предобработка»).
"""

