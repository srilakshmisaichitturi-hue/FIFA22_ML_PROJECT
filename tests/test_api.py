from fastapi.testclient import TestClient
from app.main import app

def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        assert response.json()["model_loaded"] is True

def test_predict():
    with TestClient(app) as client:
        payload = {
            "age": 25,
            "height_cm": 182.0,
            "weight_kg": 76.0,
            "pace": 78.0,
            "shooting": 72.0,
            "passing": 75.0,
            "dribbling": 77.0,
            "defending": 60.0,
            "physic": 72.0,
            "attacking_crossing": 70.0,
            "attacking_finishing": 68.0
        }
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "predicted_overall" in data
        assert 40 <= data["predicted_overall"] <= 99