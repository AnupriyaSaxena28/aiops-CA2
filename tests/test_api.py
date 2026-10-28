from fastapi.testclient import TestClient
from app import app
import json
from pathlib import Path

client = TestClient(app)
FEATURE_NAMES = json.loads((Path(__file__).resolve().parents[1] / "feature_names.json").read_text())


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "running" in response.json()["message"]


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True
    assert response.json()["feature_count"] == 57


def test_prediction_returns_expected_fields():
    response = client.post("/predict", json={"features": [0.0] * len(FEATURE_NAMES)})
    assert response.status_code == 200
    body = response.json()
    assert body["prediction"] in [0, 1]
    assert body["label"] in ["spam", "ham"]
    assert 0.0 <= body["spam_probability"] <= 1.0


def test_rejects_wrong_number_of_features():
    response = client.post("/predict", json={"features": [0.0, 1.0]})
    assert response.status_code == 422
