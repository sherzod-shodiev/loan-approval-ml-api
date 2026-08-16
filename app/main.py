import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, ConfigDict

# Загружаем твою обученную модель
MODEL_PATH = "lgbm_model.joblib"
model = joblib.load(MODEL_PATH)

app = FastAPI(title="ML Inference API", version="1.0")

# ==========================================================================
# СХЕМА ВХОДНЫХ ДАННЫХ (Pydantic)
# ==========================================================================
class Features(BaseModel):
    # Используем alias, чтобы FastAPI принимал названия с пробелами
    age: int = Field(alias="Age")
    person_income: float = Field(alias="Person Income")
    employee_experience: float = Field(alias="Employee Experience")
    loan_amount: float = Field(alias="Loan Amount")
    loan_interest_rate: float = Field(alias="Loan interest Rate")
    loan_percentage: float = Field(alias="Loan percentage")
    credit_history: float = Field(alias="Credit History")
    credit_score: int = Field(alias="Credit Score")
    
    gender: str = Field(alias="Gender")
    education: str = Field(alias="Education")
    home_ownership: str = Field(alias="Home Ownership")
    loan_intent: str = Field(alias="Loan Intent")

    # Пример значений для страницы Swagger (docs)
    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "examples": [{
                "Age": 28,
                "Person Income": 60000,
                "Employee Experience": 5,
                "Loan Amount": 15000,
                "Loan interest Rate": 11.2,
                "Loan percentage": 0.25,
                "Credit History": 6,
                "Credit Score": 720,
                "Gender": "Female",
                "Education": "Master",
                "Home Ownership": "Mortgage",
                "Loan Intent": "Education"
            }]
        }
    )

# ==========================================================================
# ЭНДПОИНТЫ
# ==========================================================================
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: Features):
    # by_alias=True ВАЖНО: передает в DataFrame ключи с пробелами (как ждет модель)
    data_dict = features.model_dump(by_alias=True)
    X = pd.DataFrame([data_dict])

    # Получаем вероятность и класс (классификация)
    proba = float(model.predict_proba(X)[0, 1])
    prediction = int(proba >= 0.5)
    
    return {
        "prediction": prediction,
        "probability": round(proba, 3),
    }