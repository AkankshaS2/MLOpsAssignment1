from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    patient = {
        "gender": "Female",
        "age": 45,
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 95.5,
        "bmi": 28.0,
        "smoking_status": "never smoked"
    }

    response = client.post("/predict", json=patient)

    assert response.status_code == 200
    result = response.json()
    assert result["prediction"] in [0, 1]
    assert 0 <= result["stroke_probability"] <= 1