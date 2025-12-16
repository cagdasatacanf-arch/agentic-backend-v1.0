from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_metadata():
    response = client.get("/api/v1/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "chat" in data["models"]
