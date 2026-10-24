import pytest

from src.app import app


@pytest.fixture
def client():
    app.config["TESTING"] = True

    with app.test_client() as test_client:
        yield test_client


def test_homepage_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"EstateIQ" in response.data
    assert b"prediction-form" in response.data


def test_health_endpoint(client):
    response = client.get("/api/health")

    assert response.status_code == 200

    data = response.get_json()
    assert data["status"] == "ok"
    assert "model_ready" in data
    assert "dataset_ready" in data


def test_metrics_endpoint_responds(client):
    response = client.get("/api/metrics")

    assert response.status_code == 200
    assert isinstance(response.get_json(), dict)


def test_prediction_rejects_missing_fields(client):
    response = client.post("/api/predict", json={})

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_prediction_rejects_invalid_room_counts(client):
    response = client.post(
        "/api/predict",
        json={
            "bedrooms": -2,
            "bathrooms": 3,
            "toilets": 3,
            "parking_space": 2,
            "title": "Detached Duplex",
            "town": "Lekki",
            "state": "Lagos",
        },
    )

    assert response.status_code == 400
    assert "error" in response.get_json()


def test_prediction_rejects_non_json(client):
    response = client.post(
        "/api/predict",
        data="not json",
        content_type="text/plain",
    )

    assert response.status_code == 400
