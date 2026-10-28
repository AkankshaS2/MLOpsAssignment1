from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Stroke Prediction API")

MODEL_PATH = Path(__file__).parent / "models" / "logistic_regression.joblib"
model = joblib.load(MODEL_PATH)


class PatientData(BaseModel):
    gender: str
    age: float = Field(ge=0, le=120)
    hypertension: int = Field(ge=0, le=1)
    heart_disease: int = Field(ge=0, le=1)
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float = Field(ge=0)
    bmi: float | None = Field(default=None, ge=0)
    smoking_status: str


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "Logistic Regression"
    }


@app.post("/predict")
def predict(patient: PatientData):
    data = pd.DataFrame([patient.model_dump()])

    prediction = int(model.predict(data)[0])
    probability = float(model.predict_proba(data)[0][1])

    return {
        "prediction": prediction,
        "stroke_probability": round(probability, 4),
        "message": (
            "Stroke predicted"
            if prediction == 1
            else "No stroke predicted"
        )
    }